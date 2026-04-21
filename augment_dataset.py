"""
augment_dataset.py
==================
Oversampling pipeline for the dynamic event detector.

What it does:
  1. Loads headlines from disaster.csv (REAL_EVENT) and meme.csv (FAKE_NEWS)
  2. Calls Gemini to generate 5 tweet-style + 2 meme-style versions per headline
  3. Saves a flat augmented CSV  → augmented_tweets.csv
  4. Builds balanced InputExample pairs  → training_pairs.csv
     (used by finetune_only.py / run_full_pipeline.py)

Usage:
    export GOOGLE_API_KEY="your-key-here"
    python3 augment_dataset.py
    python3 augment_dataset.py --real 300 --fake 300 --batch 10

Flags:
    --real   N   How many real-event headlines to augment  (default 200)
    --fake   N   How many fake/meme headlines to augment   (default 200)
    --batch  N   Headlines per LLM call                    (default 8)
    --out    P   Output CSV path                           (default augmented_tweets.csv)
    --force      Re-generate even if output already exists
"""

import os, sys, time, json, random, argparse, textwrap
import pandas as pd
from google import genai
from google.genai import types

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--real",  type=int, default=200, help="# real-event headlines to augment")
parser.add_argument("--fake",  type=int, default=200, help="# fake/meme headlines to augment")
parser.add_argument("--batch", type=int, default=8,   help="Headlines per LLM call")
parser.add_argument("--out",   type=str, default="augmented_tweets.csv")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

# ── API key ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyCxatU-ED-jwI_w8DpnYpsSrrWhyYDecFE")
if not API_KEY:
    sys.exit(
        "❌  Set your Gemini API key first:\n"
        "    export GOOGLE_API_KEY='your-key-here'\n"
        "    Get a free key at https://aistudio.google.com/app/apikey"
    )
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"   # fast + free-tier friendly

# ── Prompt template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = textwrap.dedent("""
You are building a training dataset for an AI model that detects real news events vs memes on social media.

Your job is to generate synthetic tweet-style posts from news headlines so the model learns that informal tweet language and formal news language can mean the same thing.

You will be given a list of news headlines. For each headline:
1. Generate 5 tweet-style versions (casual, slang, emojis, informal)
2. Generate 2 meme-style posts that sound similar in topic BUT are clearly not serious news (use internet humor, jokes, irony)
3. Classify the headline as: REAL_EVENT or FAKE_NEWS

Rules for tweet versions:
- Sound like a real person tweeting in shock/fear/anger/surprise
- Use casual language: "omg", "bro", "fr", "ngl", "lmao", "wtf", "rn"
- Include emojis naturally
- Under 20 words each
- Each tweet should sound like a DIFFERENT person
- NO hashtags or mentions

Rules for meme versions:
- Sound humorous or ironic about the topic
- Clearly not a serious reaction
- Use meme language: "no cap", "slay", "based", "cope", "ratio", "L", "W"
- Under 15 words each

Rules for classification:
- REAL_EVENT: Something that actually happened in the real world
- FAKE_NEWS: Fabricated or misleading story

Headlines:
{headlines}

Return ONLY valid JSON in this exact format, nothing else:
[
  {{
    "headline": "original headline here",
    "classification": "REAL_EVENT or FAKE_NEWS",
    "tweet_versions": [
      "tweet 1",
      "tweet 2",
      "tweet 3",
      "tweet 4",
      "tweet 5"
    ],
    "meme_versions": [
      "meme 1",
      "meme 2"
    ]
  }}
]
""").strip()

# ── Load source data ────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING SOURCE DATASETS")
print("=" * 60)

# disaster.csv — real disaster/news headlines  (col: text, target=1 means real)
disaster_df = pd.read_csv("disaster.csv")
real_headlines = (
    disaster_df[disaster_df["target"] == 1]["text"]
    .dropna()
    .str.strip()
    .unique()
    .tolist()
)
real_headlines = [h for h in real_headlines if len(h) > 20]   # drop junk
random.shuffle(real_headlines)
real_headlines = real_headlines[: args.real]

# meme.csv — sarcastic/fake news headlines  (col: headline, is_sarcastic=1 means fake)
meme_df = pd.read_csv("meme.csv")
fake_headlines = (
    meme_df[meme_df["is_sarcastic"] == 1]["headline"]
    .dropna()
    .str.strip()
    .unique()
    .tolist()
)
random.shuffle(fake_headlines)
fake_headlines = fake_headlines[: args.fake]

print(f"  Real-event headlines selected : {len(real_headlines)}")
print(f"  Fake/meme  headlines selected : {len(fake_headlines)}")

all_headlines = [("REAL_EVENT", h) for h in real_headlines] + \
                [("FAKE_NEWS",  h) for h in fake_headlines]
random.shuffle(all_headlines)


# ── LLM call helper ────────────────────────────────────────────────────────────
def _extract_json(raw: str) -> list[dict]:
    """
    Robustly extract a JSON array from LLM output that may:
    - be wrapped in markdown fences
    - be truncated mid-object (take all complete objects)
    """
    # Strip markdown fences
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        # grab the content inside the first fence pair
        for p in parts[1:]:
            if p.startswith("json"):
                raw = p[4:].strip()
                break
            elif p.strip().startswith("["):
                raw = p.strip()
                break

    # Try full parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Truncated output: collect complete objects by scanning for closing braces
    # Find the opening bracket
    start = raw.find("[")
    if start == -1:
        return []
    raw = raw[start:]

    complete = []
    depth = 0
    obj_start = None
    in_string = False
    escape = False

    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 1:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1 and obj_start is not None:
                try:
                    obj = json.loads(raw[obj_start : i + 1])
                    complete.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == "[":
            depth += 1 if depth > 0 else 1  # count the outer bracket

    return complete


def call_llm(batch: list[tuple[str, str]], retries: int = 3) -> list[dict]:
    """
    batch: list of (expected_class, headline_text)
    Returns list of parsed JSON dicts (may be shorter than batch on parse failure).
    """
    headline_block = "\n".join(f"- {h}" for _, h in batch)
    prompt = PROMPT_TEMPLATE.format(headlines=headline_block)

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    max_output_tokens=8192,   # higher limit = fewer truncations
                ),
            )
            raw = response.text
            parsed = _extract_json(raw)
            if parsed:
                return parsed
            print(f"    ⚠  No valid JSON objects extracted (attempt {attempt})")
        except Exception as e:
            print(f"    ⚠  LLM error (attempt {attempt}): {e}")

        if attempt < retries:
            time.sleep(2 ** attempt)   # exponential back-off

    return []   # give up on this batch


# ── Main generation loop ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING AUGMENTED DATA")
print("=" * 60)

if not args.force and os.path.exists(args.out):
    print(f"  '{args.out}' already exists. Use --force to regenerate.")
    rows = pd.read_csv(args.out).to_dict("records")
else:
    rows = []
    batch_size = args.batch
    total = len(all_headlines)
    errors = 0

    for i in range(0, total, batch_size):
        batch = all_headlines[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{-(-total // batch_size)}  "
              f"({i + 1}–{min(i + batch_size, total)} of {total})", end=" ", flush=True)

        results = call_llm(batch)
        if not results:
            errors += 1
            print("❌  skipped")
            continue

        for item in results:
            headline        = item.get("headline", "")
            classification  = item.get("classification", "REAL_EVENT")
            tweet_versions  = item.get("tweet_versions", [])
            meme_versions   = item.get("meme_versions", [])

            # --- tweet rows  (label = "TWEET", same class as headline)
            for tweet in tweet_versions:
                rows.append({
                    "headline":       headline,
                    "text":           tweet,
                    "text_type":      "tweet",
                    "classification": classification,
                })

            # --- meme rows  (always MEME regardless of headline class)
            for meme in meme_versions:
                rows.append({
                    "headline":       headline,
                    "text":           meme,
                    "text_type":      "meme",
                    "classification": "MEME",
                })

        print(f"✅  +{len(results)} items")
        time.sleep(0.5)   # polite rate-limiting

    print(f"\n  Done — {len(rows)} rows generated, {errors} batches failed")

    aug_df = pd.DataFrame(rows)
    aug_df.to_csv(args.out, index=False)
    print(f"  Saved → {args.out}")


# ── Show distribution ───────────────────────────────────────────────────────────
aug_df = pd.read_csv(args.out)
print("\n── Augmented dataset distribution ──────────────────────────")
print(aug_df.groupby(["classification", "text_type"]).size().to_string())
print(f"\nTotal rows : {len(aug_df)}")


# ── Build balanced training pairs for SBERT ────────────────────────────────────
print("\n" + "=" * 60)
print("BUILDING SBERT TRAINING PAIRS")
print("=" * 60)

from sentence_transformers import InputExample
import pickle

real_tweets  = aug_df[aug_df["classification"] == "REAL_EVENT"]["text"].tolist()
fake_tweets  = aug_df[aug_df["classification"] == "FAKE_NEWS"]["text"].tolist()
meme_tweets  = aug_df[aug_df["classification"] == "MEME"]["text"].tolist()

real_heads   = aug_df[aug_df["classification"] == "REAL_EVENT"]["headline"].tolist()
fake_heads   = aug_df[aug_df["classification"] == "FAKE_NEWS"]["headline"].tolist()

print(f"  REAL_EVENT texts : {len(real_tweets)}  (tweets) + {len(real_heads)} (headlines)")
print(f"  FAKE_NEWS  texts : {len(fake_tweets)}")
print(f"  MEME       texts : {len(meme_tweets)}")

train_examples = []

# 1. Positive pairs: tweet ↔ original headline (same real event)
n = min(2000, len(real_tweets), len(real_heads))
for tw, hl in zip(random.sample(real_tweets, n), random.sample(real_heads, n)):
    train_examples.append(InputExample(texts=[tw, hl], label=0.9))

# 2. Positive pairs: tweet ↔ tweet (same event class)
for j in range(min(1000, len(real_tweets) // 2)):
    if j + 1 < len(real_tweets):
        train_examples.append(InputExample(texts=[real_tweets[j], real_tweets[j+1]], label=0.85))

# 3. Negative pairs: real tweet ↔ meme tweet  (should be very dissimilar)
n_neg = min(2000, len(real_tweets), len(meme_tweets))
for tw, m in zip(random.sample(real_tweets, n_neg), random.sample(meme_tweets, n_neg)):
    train_examples.append(InputExample(texts=[tw, m], label=0.05))

# 4. Negative pairs: real tweet ↔ fake headline
n_neg2 = min(1000, len(real_tweets), len(fake_heads))
for tw, fh in zip(random.sample(real_tweets, n_neg2), random.sample(fake_heads, n_neg2)):
    train_examples.append(InputExample(texts=[tw, fh], label=0.1))

# 5. Medium pairs: fake tweet ↔ fake headline (somewhat related)
n_med = min(1000, len(fake_tweets), len(fake_heads))
for tw, fh in zip(random.sample(fake_tweets, n_med), random.sample(fake_heads, n_med)):
    train_examples.append(InputExample(texts=[tw, fh], label=0.6))

random.shuffle(train_examples)

# Save pairs as pickle (for use in fine-tuning)
PAIRS_FILE = "augmented_training_pairs.pkl"
with open(PAIRS_FILE, "wb") as f:
    pickle.dump(train_examples, f)

# Also save as human-readable CSV for inspection
pairs_csv = pd.DataFrame([
    {"text1": ex.texts[0], "text2": ex.texts[1], "label": ex.label}
    for ex in train_examples
])
pairs_csv.to_csv("training_pairs.csv", index=False)

print(f"\n  Training pairs breakdown:")
print(f"    tweet ↔ headline  (real, pos)   : {min(2000, len(real_tweets), len(real_heads))}")
print(f"    tweet ↔ tweet     (real, pos)   : {min(1000, len(real_tweets) // 2)}")
print(f"    tweet ↔ meme      (neg)         : {n_neg}")
print(f"    tweet ↔ fake hl   (neg)         : {n_neg2}")
print(f"    fake tw ↔ fake hl (med)         : {n_med}")
print(f"    ─────────────────────────────────")
print(f"    TOTAL pairs                     : {len(train_examples)}")
print(f"\n  Saved → {PAIRS_FILE}")
print(f"  Saved → training_pairs.csv")

print("\n" + "=" * 60)
print("✅  AUGMENTATION COMPLETE")
print("=" * 60)
print("""
Next step — fine-tune with the augmented pairs:

    python3 finetune_augmented.py

Or pass --augmented flag to run_full_pipeline.py (see finetune_augmented.py).
""")
