"""
run_full_pipeline.py
====================
Full end-to-end pipeline (no Colab needed):
  1.  Load & clean data
  2.  Embed with SBERT (all-MiniLM-L6-v2)
  3.  Cluster with BERTopic
  4.  Score each cluster with meme similarity  → saves scores_df.csv
  5.  Classify via GDELT v2 artlist + dynamic percentile thresholds
  6.  Fine-tune SBERT on the labels
  7.  Save fine-tuned model → ./finetuned_event_model/

Run fresh (forces re-computation of everything):
    python3 run_full_pipeline.py --force

Run (automatically skips Steps 1-4 if scores_df.csv exists):
    python3 run_full_pipeline.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import re, sys, os, time, ast, random, json, importlib.util
import requests
import numpy as np
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FORCE_RECOMPUTE = "--force" in sys.argv
RESUME_MODE = not FORCE_RECOMPUTE and os.path.exists("scores_df.csv")
DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None
ACCELERATE_AVAILABLE = importlib.util.find_spec("accelerate") is not None
FINE_TUNING_COMPLETED = False

from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader

try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("⚠  BERTopic/UMAP/HDBSCAN not installed — pip install bertopic umap-learn hdbscan")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: clean text
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

# ─────────────────────────────────────────────────────────────────────────────
# RESUME branch — load cached files and jump to Step 5
# ─────────────────────────────────────────────────────────────────────────────
SCORES_CACHE = "scores_df.csv"
EMB_CACHE    = "embeddings_v1.npy"

if RESUME_MODE and os.path.exists(SCORES_CACHE):
    print("=" * 60)
    print("RESUME MODE — loading cached scores_df.csv")
    print("=" * 60)
    scores_df = pd.read_csv(SCORES_CACHE)
    # keywords column was serialised as string list — convert back
    scores_df["keywords"] = scores_df["keywords"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # We still need all_texts + df for the fine-tuning step
    print("  Loading raw data for fine-tuning text lookup …")
    fake_news = pd.read_csv("disaster.csv")
    sentiment = pd.read_csv("sentiment.csv", header=None)
    meme_df   = pd.read_csv("meme.csv")
    sentiment.columns = ["id", "entity", "sentiment", "text"]
    fake_texts  = fake_news["text"].fillna("").astype(str).apply(clean_text).tolist()[:10_000]
    sent_texts  = sentiment["text"].fillna("").astype(str).apply(clean_text).tolist()[:10_000]
    meme_texts  = meme_df["headline"].fillna("").astype(str).apply(clean_text).tolist()[:5_000]
    all_texts   = fake_texts + sent_texts + meme_texts
    topics_list = scores_df["topic_id"].tolist()  # placeholder; df built below

    # Rebuild cluster-to-topic mapping from scores_df alone (fine-tuning uses it)
    # We fake df from scores_df topic_ids so fine-tuning has something to pull texts from
    df = None   # signal to fine-tuning that it should use scores_df keywords directly
    print(f"  Loaded {len(scores_df)} scored clusters — skipping to Step 5\n")

else:
    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Load data
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading data …")
    print("=" * 60)

    fake_news = pd.read_csv("disaster.csv")
    sentiment = pd.read_csv("sentiment.csv", header=None)
    meme_df   = pd.read_csv("meme.csv")
    sentiment.columns = ["id", "entity", "sentiment", "text"]

    fake_texts  = fake_news["text"].fillna("").astype(str).apply(clean_text).tolist()[:10_000]
    sent_texts  = sentiment["text"].fillna("").astype(str).apply(clean_text).tolist()[:10_000]
    meme_texts  = meme_df["headline"].fillna("").astype(str).apply(clean_text).tolist()[:5_000]
    all_texts   = fake_texts + sent_texts + meme_texts
    source_labels = (["disaster"] * len(fake_texts) +
                     ["sentiment"] * len(sent_texts) +
                     ["meme"] * len(meme_texts))

    print(f"  disaster  : {len(fake_texts):,}")
    print(f"  sentiment : {len(sent_texts):,}")
    print(f"  meme      : {len(meme_texts):,}")
    print(f"  TOTAL     : {len(all_texts):,}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Embeddings
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Generating embeddings …")
    print("=" * 60)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(EMB_CACHE):
        embeddings = np.load(EMB_CACHE, allow_pickle=False)
        if embeddings.shape[0] != len(all_texts):
            print("  Cache mismatch — re-embedding …")
            embeddings = model.encode(all_texts, batch_size=64, show_progress_bar=True)
            np.save(EMB_CACHE, embeddings)
        else:
            print(f"  Loaded cached embeddings {embeddings.shape}")
    else:
        embeddings = model.encode(all_texts, batch_size=64, show_progress_bar=True)
        np.save(EMB_CACHE, embeddings)
        print(f"  Saved → {EMB_CACHE}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3 — BERTopic
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — BERTopic clustering …")
    print("=" * 60)

    if BERTOPIC_AVAILABLE:
        topic_model = BERTopic(
            umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                            random_state=42, low_memory=True),
            hdbscan_model=HDBSCAN(min_cluster_size=15, min_samples=5,
                                  prediction_data=True),
            vectorizer_model=CountVectorizer(stop_words="english",
                                             ngram_range=(1, 2), min_df=5,
                                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"),
            nr_topics="auto",
            verbose=True,
        )
        topics, _ = topic_model.fit_transform(all_texts, embeddings)
        print(f"  Topics found: {len(set(topics)) - 1}  (excl. -1 noise)")
    else:
        print("  Fallback: random topic assignment")
        topics = [random.randint(0, 29) for _ in all_texts]
        topic_model = None

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4 — Meme similarity scoring
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Meme similarity scoring …")
    print("=" * 60)

    meme_embeddings = model.encode(meme_texts, batch_size=64, show_progress_bar=True)
    df = pd.DataFrame({"text": all_texts, "topic": topics, "source": source_labels})

    STOPWORDS = {
        "the","a","an","is","it","in","on","at","to","of","and","or","for",
        "with","this","that","was","are","be","as","by","from","have","has",
        "had","not","but","they","we","you","he","she","his","her","its",
        "our","their","which","who","what","how","when","where","were","will",
        "can","just","so","if","do","did","no","up","out","my","your",
    }

    def meme_similarity_score(cluster_texts, sample=100):
        c_emb = model.encode(cluster_texts[:sample], show_progress_bar=False)
        sims  = util.cos_sim(c_emb, meme_embeddings)
        return sims.max(dim=1).values.mean().item()

    def extract_keywords(cluster_texts, top_n=8):
        from collections import Counter
        words = [w for t in cluster_texts
                 for w in re.findall(r"\b[a-z]{4,}\b", t) if w not in STOPWORDS]
        return [w for w, _ in Counter(words).most_common(top_n)]

    cluster_scores = []
    unique_topics  = sorted(t for t in df["topic"].unique() if t != -1)
    print(f"  Scoring {len(unique_topics)} topics …")

    for topic_id in unique_topics:
        cluster_texts = df[df["topic"] == topic_id]["text"].tolist()
        if len(cluster_texts) < 5:
            continue
        cluster_scores.append({
            "topic_id":   topic_id,
            "size":       len(cluster_texts),
            "meme_score": meme_similarity_score(cluster_texts),
            "keywords":   extract_keywords(cluster_texts),
        })

    scores_df = pd.DataFrame(cluster_scores)
    scores_df.to_csv(SCORES_CACHE, index=False)       # ← checkpoint for --resume
    print(f"  Scored {len(scores_df)} clusters — saved to {SCORES_CACHE}")
    print(scores_df[["topic_id", "size", "meme_score", "keywords"]].head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — GDELT v2 artlist classification
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — GDELT v2 classification …")
print("=" * 60)

# Dynamic thresholds from the actual score distribution
scores_arr     = scores_df["meme_score"].values
LOW_THRESHOLD  = float(np.percentile(scores_arr, 25))
HIGH_THRESHOLD = float(np.percentile(scores_arr, 75))

print("  Meme score distribution:")
print(f"    Min    : {scores_arr.min():.3f}")
print(f"    25th   : {LOW_THRESHOLD:.3f}   ← below = REAL EVENT candidate")
print(f"    Median : {np.median(scores_arr):.3f}")
print(f"    75th   : {HIGH_THRESHOLD:.3f}   ← above = MEME candidate")
print(f"    Max    : {scores_arr.max():.3f}")

# Gaming / pop-culture noise filter
NOISE_KEYWORDS = {
    "hearthstone", "xbox", "playstation", "fifa", "fortnite", "minecraft",
    "twitch", "stream", "gaming", "league", "legends", "pokemon", "nintendo",
    "gamer", "gameplay", "ps5", "preorder", "warzone", "valorant", "apex",
    "gta", "rockstar", "redemption", "deadred", "overwatch", "blizzard",
    "esports", "halo", "pubg", "roblox", "streamer", "twitchtv",
    "kardashian", "taylor", "swift", "bieber", "tiktok",
}

def is_gaming_noise(keywords):
    return bool(set(k.lower() for k in keywords) & NOISE_KEYWORDS)

# GDELT v2 artlist — confirmed working, with JSON caching to avoid rate limits
GDELT_CACHE_FILE = "gdelt_cache.json"
GDELT_V2_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_V1_URL = "https://api.gdeltproject.org/api/v1/search_ftxtsearch/search_ftxtsearch"
GDELT_CONNECT_TIMEOUT = 5
GDELT_READ_TIMEOUT = 20
GDELT_FAILURE_LIMIT = 5

gdelt_session = requests.Session()
gdelt_retry = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=1.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
    raise_on_status=False,
)
gdelt_adapter = HTTPAdapter(max_retries=gdelt_retry)
gdelt_session.mount("https://", gdelt_adapter)
gdelt_session.mount("http://", gdelt_adapter)

# Load cache if it exists
if os.path.exists(GDELT_CACHE_FILE):
    with open(GDELT_CACHE_FILE, "r") as f:
        gdelt_cache = json.load(f)
else:
    gdelt_cache = {}

gdelt_state = {
    "consecutive_failures": 0,
    "disabled": False,
}

def _save_gdelt_cache():
    with open(GDELT_CACHE_FILE, "w") as f:
        json.dump(gdelt_cache, f)

def _query_gdelt_v2(query, delay=1.0):
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": 50}
    time.sleep(delay)
    r = gdelt_session.get(
        GDELT_V2_URL,
        params=params,
        timeout=(GDELT_CONNECT_TIMEOUT, GDELT_READ_TIMEOUT),
    )
    if r.status_code != 200:
        return None
    return len(r.json().get("articles", []))

def _query_gdelt_v1(query, delay=1.0):
    params = {"query": query, "output": "artlist", "maxrecords": 50, "format": "json"}
    time.sleep(delay)
    r = gdelt_session.get(
        GDELT_V1_URL,
        params=params,
        timeout=(GDELT_CONNECT_TIMEOUT, GDELT_READ_TIMEOUT),
    )
    if r.status_code != 200:
        return None
    return len(r.json().get("articles", []))

def verify_gdelt(keywords, delay=1.0):
    clean_kw = [k.lower() for k in keywords if len(k) > 3 and k.isalpha()]
    if len(clean_kw) < 2:
        return 0
    query  = " ".join(clean_kw[:2])

    # Check cache first
    if query in gdelt_cache:
        return gdelt_cache[query]

    if gdelt_state["disabled"]:
        return 0

    try:
        count = _query_gdelt_v2(query, delay=delay)
        if count is None:
            count = _query_gdelt_v1(query, delay=0.2)

        if count is not None:
            gdelt_state["consecutive_failures"] = 0
            gdelt_cache[query] = count
            _save_gdelt_cache()
            return count
    except requests.exceptions.RequestException as e:
        print(f"    GDELT network error ({query}): {e}")
    except ValueError as e:
        print(f"    GDELT parse error ({query}): {e}")

    gdelt_state["consecutive_failures"] += 1
    if gdelt_state["consecutive_failures"] >= GDELT_FAILURE_LIMIT:
        gdelt_state["disabled"] = True
        print("    ⚠  Disabling live GDELT lookups after repeated failures; cached results will still be used.")
    return 0

# Sanity check
print("\n  GDELT sanity check:")
for pair in [["australia", "bushfire"], ["iran", "airplane"], ["meghan", "harry"]]:
    cnt = verify_gdelt(pair)
    print(f"    {'✅' if cnt > 0 else '❌'}  {' + '.join(pair):<30} → {cnt} articles")

# Classification loop
print("\n  Classifying topics …")
final_results = []

for i, (_, row) in enumerate(scores_df.iterrows()):
    keywords   = row["keywords"]
    meme_score = row["meme_score"]

    if is_gaming_noise(keywords):
        label, news_count = "NOISE (GAMING)", 0
    else:
        news_count = verify_gdelt(keywords, delay=0.6)
        if   news_count > 10 and meme_score < LOW_THRESHOLD:
            label = "REAL EVENT"
        elif meme_score > HIGH_THRESHOLD:
            label = "MEME"
        elif news_count > 5:
            label = "POSSIBLE EVENT"
        else:
            label = "NOISE"

    final_results.append({
        "topic_id":   row["topic_id"],
        "keywords":   keywords,
        "news_count": news_count,
        "meme_score": meme_score,
        "label":      label,
    })

    if (i + 1) % 10 == 0 or (i + 1) == len(scores_df):
        print(f"    [{i + 1:>3}/{len(scores_df)}] last: {label:<16}  kw={keywords[:3]}")

results_df = pd.DataFrame(final_results)
results_df.to_csv("fixed_approach_results.csv",    index=False)
results_df.to_csv("fixed_approach_v2_results.csv", index=False)

print("\n  ── Classification summary ──────────────────────────────")
print(results_df["label"].value_counts().to_string())

for lbl, title in [("REAL EVENT","REAL EVENTS"),("POSSIBLE EVENT","POSSIBLE EVENTS"),("MEME","TOP MEMES")]:
    sub = results_df[results_df["label"] == lbl]
    if lbl == "MEME":
        sub = sub.sort_values("meme_score", ascending=False).head(10)
    print(f"\n  === {title} ===")
    if sub.empty:
        print("    (none)")
    else:
        print(sub[["topic_id","keywords","news_count","meme_score","label"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Fine-tune SBERT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Fine-tuning SBERT …")
print("=" * 60)

# Load SBERT if not already loaded (resume path skips Steps 1-4)
if "model" not in dir():
    model = SentenceTransformer("all-MiniLM-L6-v2")

# Build per-label text lists
# In resume mode we use the keyword strings as proxy texts (lightweight)
label_texts: dict[str, list[str]] = {}
for _, row in results_df.iterrows():
    lbl = row["label"]
    kws = row["keywords"] if isinstance(row["keywords"], list) else ast.literal_eval(str(row["keywords"]))
    # Use keyword joined string as a proxy sentence when full df unavailable
    if df is not None:
        texts = df[df["topic"] == row["topic_id"]]["text"].tolist()[:20]
    else:
        texts = [" ".join(kws)] * 5   # lightweight proxy
    label_texts.setdefault(lbl, []).extend(texts)

real_texts   = label_texts.get("REAL EVENT", [])
meme_texts_l = label_texts.get("MEME", []) + label_texts.get("NOISE (GAMING)", [])
poss_texts   = label_texts.get("POSSIBLE EVENT", [])

print(f"  Texts available — REAL: {len(real_texts)}  MEME: {len(meme_texts_l)}  POSSIBLE: {len(poss_texts)}")

train_examples = []

# Positive pairs: real–real
for j in range(min(500, len(real_texts) // 2)):
    if j + 1 < len(real_texts):
        train_examples.append(InputExample(texts=[real_texts[j], real_texts[j+1]], label=0.9))

# Positive pairs: meme–meme
for j in range(min(500, len(meme_texts_l) // 2)):
    if j + 1 < len(meme_texts_l):
        train_examples.append(InputExample(texts=[meme_texts_l[j], meme_texts_l[j+1]], label=0.9))

# Negative pairs: real vs meme
n = min(1000, len(real_texts), len(meme_texts_l))
for r, m in zip(random.sample(real_texts, n), random.sample(meme_texts_l, n)):
    train_examples.append(InputExample(texts=[r, m], label=0.05))

# Medium pairs: possible event
for j in range(min(300, len(poss_texts) // 2)):
    if j + 1 < len(poss_texts):
        train_examples.append(InputExample(texts=[poss_texts[j], poss_texts[j+1]], label=0.6))

print(f"  Training pairs: {len(train_examples)}")

if not real_texts or not meme_texts_l:
    print("  ⚠  Fine-tuning skipped: need both REAL EVENT and MEME-labeled texts.")
    print("     Current labels are too one-sided for a useful contrastive training step.")
elif len(train_examples) < 10:
    print("  ⚠  Too few pairs — skipping fine-tuning.")
    print("     Tip: lower the news_count threshold (>10 → >5) in Step 5.")
elif not DATASETS_AVAILABLE:
    print("  ⚠  Fine-tuning skipped: missing optional dependency `datasets`.")
    print("     Install it in your virtualenv with: pip install datasets")
elif not ACCELERATE_AVAILABLE:
    print("  ⚠  Fine-tuning skipped: missing optional dependency `accelerate`.")
    print("     Install it in your virtualenv with: pip install 'accelerate>=1.1.0'")
else:
    ft_model         = SentenceTransformer("all-MiniLM-L6-v2")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss       = losses.CosineSimilarityLoss(model=ft_model)

    print("  Training …")
    try:
        ft_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=max(1, len(train_dataloader) // 10),
            show_progress_bar=True,
        )
    except ImportError as e:
        print(f"  ⚠  Fine-tuning skipped due to trainer dependency error: {e}")
        print("     Install the missing package in your virtualenv and rerun Step 6.")
        ft_model = None

    if ft_model is not None:
        OUTPUT_DIR = "finetuned_event_model"
        ft_model.save(OUTPUT_DIR)
        FINE_TUNING_COMPLETED = True
        print(f"\n  ✅  Fine-tuned model saved → ./{OUTPUT_DIR}/")

        if real_texts and meme_texts_l:
            r_emb     = ft_model.encode(real_texts[:5])
            m_emb     = ft_model.encode(meme_texts_l[:5])
            cross_sim = util.cos_sim(r_emb, m_emb).mean().item()
            self_sim  = util.cos_sim(r_emb, r_emb).mean().item()
            print(f"\n  Eval — Real↔Real: {self_sim:.3f}  (want high)")
            print(f"         Real↔Meme: {cross_sim:.3f}  (want low)")

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("  fixed_approach_results.csv    — classification labels")
print("  fixed_approach_v2_results.csv — same file (alias)")
print("  scores_df.csv                 — cluster scores checkpoint")
if FINE_TUNING_COMPLETED:
    print("  finetuned_event_model/        — fine-tuned SBERT")
print()
