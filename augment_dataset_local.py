"""
augment_dataset_local.py
========================
Local, deterministic dataset augmentation for the dynamic event detector.

This mirrors the output contract of augment_dataset.py without using Gemini:
  1. Builds tweet-style rewrites from real and fake headlines
  2. Builds meme-style posts from the same headlines
  3. Saves augmented_tweets.csv
  4. Builds augmented_training_pairs.pkl and training_pairs.csv

Usage:
    python3 augment_dataset_local.py
    python3 augment_dataset_local.py --real 400 --fake 400 --seed 42
"""

import argparse
import pickle
import random
import re
from collections import Counter

import pandas as pd
from sentence_transformers import InputExample


parser = argparse.ArgumentParser()
parser.add_argument("--real", type=int, default=400, help="# real-event headlines to augment")
parser.add_argument("--fake", type=int, default=400, help="# fake/meme headlines to augment")
parser.add_argument("--out", type=str, default="augmented_tweets.csv", help="Augmented CSV output path")
parser.add_argument("--pairs", type=str, default="augmented_training_pairs.pkl", help="Pickle output path")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

rng = random.Random(args.seed)

SLANG_OPENERS = [
    "omg",
    "bro",
    "wtf",
    "nah",
    "yo",
    "fr",
    "ngl",
    "no way",
    "aint no way",
    "bruh",
]

REACTION_FRAGMENTS = [
    "this is wild",
    "what is going on rn",
    "actually insane",
    "cant believe this",
    "this is so messed up",
    "im shook",
    "everybody stay safe",
    "people really going through it",
    "this is heartbreaking",
    "hope everyone is okay",
]

EMOJIS_BY_TOPIC = {
    "quake": ["😱", "🌍", "💥"],
    "storm": ["⛈️", "🌧️", "😳"],
    "flood": ["🌊", "😰", "😭"],
    "fire": ["🔥", "😳", "🚨"],
    "explosion": ["💥", "🚨", "😱"],
    "crash": ["😬", "🚨", "💥"],
    "health": ["😷", "🧪", "🚨"],
    "finance": ["📉", "😬", "🤯"],
    "violence": ["😟", "🚨", "💔"],
    "default": ["😳", "😭", "🚨", "🤯"],
}

MEME_PREFIXES = [
    "bro said",
    "this headline really said",
    "me reading this like",
    "the timeline after seeing this",
    "internet gonna turn this into",
]

MEME_SUFFIXES = [
    "no cap",
    "im crying",
    "this is sending me",
    "absolute cinema",
    "what a timeline",
    "major L energy",
    "kinda wild ngl",
]

HARD_NEGATIVE_MEMES = [
    "skibidi toilet no cap lmao",
    "skibidi ohio rizz is crazy",
    "only in ohio bro im crying",
    "npc energy fr no cap",
    "gyatt level event of the year",
    "delulu timeline is undefeated",
    "bro really said sigma grindset",
    "mewing streak got broken rip",
    "this is so brainrot i cant",
    "goofy ahh moment fr",
    "caught in 4k being cringe",
    "main character syndrome update",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "this", "that", "from", "into",
    "over", "under", "after", "before", "they", "them", "their", "have", "has",
    "had", "been", "were", "will", "would", "could", "should", "about", "just",
    "more", "than", "into", "onto", "your", "while", "where", "when", "what",
    "says", "say", "amid", "over", "near", "today", "night", "across",
}

REAL_EVENT_KEYWORDS = [
    "earthquake", "quake", "storm", "hurricane", "cyclone", "flood", "flooding",
    "fire", "wildfire", "explosion", "blast", "bomb", "crash", "collision",
    "plane", "aircraft", "virus", "outbreak", "covid", "hospital", "shooting",
    "attack", "killed", "evacuation", "evacuated", "market crash", "stocks",
    "stock market", "economy", "market falls", "pandemic", "health emergency",
]

CURATED_POSITIVE_PAIRS = [
    ("markets bleeding rn sell everything 📉", "Stock market sees sharp decline", 0.95),
    ("markets are tanking rn", "Stock market sees sharp decline", 0.92),
    ("everything red on my screen rn 📉", "Global stocks tumble as markets fall", 0.92),
    ("new covid variant spreading fr", "WHO monitors new virus strain", 0.95),
    ("new variant news again 😷", "Health officials monitor new COVID variant", 0.92),
    ("virus update got me stressed", "Officials report outbreak of new virus strain", 0.9),
    ("omg ground shaking in Istanbul 😱", "6.2 magnitude earthquake strikes Turkey", 0.95),
]

CURATED_NEGATIVE_PAIRS = [
    ("skibidi toilet no cap lmao", "6.2 magnitude earthquake strikes Turkey", 0.0),
    ("this cat is literally everything 😭", "WHO declares health emergency", 0.0),
    ("this cat is literally everything 😭", "Stock market sees sharp decline", 0.0),
    ("only in ohio bro im crying", "Officials warn of severe flooding", 0.0),
    ("npc energy fr no cap", "New virus strain under investigation", 0.0),
]

TOPIC_HOOKS = {
    "quake": [
        "ground really shaking rn",
        "another earthquake update",
        "earthquake news got me stressed",
        "buildings shaking is terrifying",
    ],
    "storm": [
        "storm warnings keep getting worse",
        "weather is getting ugly rn",
        "winds are doing too much",
        "this storm path is scary",
    ],
    "flood": [
        "roads really underwater rn",
        "flooding got everything shut down",
        "water levels rising fast",
        "flood updates are scary",
    ],
    "fire": [
        "fire spread is terrifying",
        "everything burning so fast rn",
        "wildfire updates are brutal",
        "smoke everywhere rn",
    ],
    "explosion": [
        "that blast sounds insane",
        "explosion news got me sick",
        "people heard that boom everywhere",
        "blast update is terrifying",
    ],
    "crash": [
        "crash update is horrifying",
        "plane news getting worse",
        "wreck footage is brutal",
        "collision reports are scary",
    ],
    "health": [
        "new variant news again",
        "virus update got me stressed",
        "outbreak headlines keep coming",
        "public health news is scary",
    ],
    "finance": [
        "markets are tanking rn",
        "stocks getting cooked today",
        "market panic is real",
        "everything red on my screen",
    ],
    "violence": [
        "attack news is horrifying",
        "people really not safe rn",
        "violence updates got me sick",
        "this situation is terrifying",
    ],
    "default": [
        "breaking news got me shook",
        "timeline is in chaos rn",
        "this update is wild",
        "i cant process this news",
    ],
}


def clean_headline(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"' ")


def infer_topic_bucket(headline: str) -> str:
    h = headline.lower()
    if re.search(r"\b(earthquake|quake|tremor|aftershock)\b", h):
        return "quake"
    if re.search(r"\b(storm|cyclone|hurricane|rain|typhoon)\b", h):
        return "storm"
    if re.search(r"\b(flood|flooding)\b", h):
        return "flood"
    if re.search(r"\b(fire|wildfire|ablaze|burn|burning|bushfire)\b", h):
        return "fire"
    if re.search(r"\b(explosion|blast|bomb)\b", h):
        return "explosion"
    if re.search(r"\b(crash|collision|plane|aircraft)\b", h):
        return "crash"
    if re.search(r"\b(covid|virus|outbreak|health|disease|hospital|variant|pandemic)\b", h):
        return "health"
    if re.search(r"\b(market|markets|stocks|stock|dow|shares|investors|economy|recession)\b", h):
        return "finance"
    if re.search(r"\b(riot|violence|shooting|attack|killed|police)\b", h):
        return "violence"
    return "default"


def looks_like_clean_real_headline(text: str) -> bool:
    t = clean_headline(text)
    lower = t.lower()
    if len(t) < 25 or len(t) > 180:
        return False
    if any(tok in lower for tok in [" i ", " my ", " me ", " we ", " our ", " lol", " lmao", " omg", "😭", "😱", "#"]):
        return False
    if lower.count("http") or lower.count("@"):
        return False
    return any(k in lower for k in REAL_EVENT_KEYWORDS)


def headline_stub(headline: str, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9']+", headline.lower())
    return " ".join(words[:max_words])


def keyword_stub(headline: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9']+", headline.lower())
    filtered = [w for w in words if len(w) > 3 and w not in STOPWORDS]
    if not filtered:
        filtered = words[:max_words]
    counts = Counter(filtered)
    ranked = [w for w, _ in counts.most_common(max_words)]
    return " ".join(ranked[:max_words])


def shorten(text: str, limit: int = 110) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].strip()


def tweet_variants(headline: str, classification: str) -> list[str]:
    topic = infer_topic_bucket(headline)
    emojis = EMOJIS_BY_TOPIC.get(topic, EMOJIS_BY_TOPIC["default"])
    stub = headline_stub(headline)
    key_stub = keyword_stub(headline)
    base = clean_headline(headline)
    hooks = TOPIC_HOOKS.get(topic, TOPIC_HOOKS["default"])

    if classification == "REAL_EVENT":
        templates = [
            "{opener} {stub} {emoji}",
            "{opener} {reaction} {emoji}",
            "{stub} and people are still acting normal {emoji}",
            "{opener} this {topic_word} news is insane {emoji}",
            "{reaction} after seeing {stub} {emoji}",
            "{hook} {emoji}",
            "{opener} {key_stub} got me stressed {emoji}",
            "{hook} and nobody was ready for this {emoji}",
        ]
        topic_word = {
            "quake": "earthquake",
            "storm": "storm",
            "flood": "flood",
            "fire": "fire",
            "explosion": "blast",
            "crash": "crash",
            "health": "health",
            "finance": "market",
            "violence": "violence",
            "default": "breaking",
        }[topic]
    else:
        templates = [
            "{opener} {stub} is taking me out {emoji}",
            "{stub} sounds made up but here we are {emoji}",
            "{opener} this headline is so unserious {emoji}",
            "{reaction} because {stub} is crazy work {emoji}",
            "{stub} got the timeline in shambles {emoji}",
            "{opener} {key_stub} really sounds fake {emoji}",
            "{hook} but in a made up way {emoji}",
        ]
        topic_word = "headline"

    variants = []
    for idx, template in enumerate(templates):
        opener = SLANG_OPENERS[idx % len(SLANG_OPENERS)]
        reaction = REACTION_FRAGMENTS[(idx + 2) % len(REACTION_FRAGMENTS)]
        emoji = emojis[idx % len(emojis)]
        text = template.format(
            opener=opener,
            stub=stub,
            key_stub=key_stub,
            reaction=reaction,
            emoji=emoji,
            topic_word=topic_word,
            hook=hooks[idx % len(hooks)],
        )
        variants.append(shorten(text))

    # Keep one variant closer to the original wording for alignment.
    variants[0] = shorten(f"{SLANG_OPENERS[0]} {base[:85]} {emojis[0]}")
    # Deduplicate while preserving order.
    return list(dict.fromkeys(variants))


def meme_variants(headline: str) -> list[str]:
    stub = keyword_stub(headline, max_words=6)
    prefix = rng.choice(MEME_PREFIXES)
    suffix = rng.choice(MEME_SUFFIXES)
    return [
        f"{prefix} {stub} and expected us to be normal",
        f"{stub} is giving {suffix}",
        f"{stub} but make it a meme page post",
    ]


def load_sources() -> tuple[list[str], list[str]]:
    disaster_df = pd.read_csv("disaster.csv")
    real_headlines = (
        disaster_df[disaster_df["target"] == 1]["text"]
        .dropna()
        .map(clean_headline)
        .loc[lambda s: s.apply(looks_like_clean_real_headline)]
        .drop_duplicates()
        .tolist()
    )

    meme_df = pd.read_csv("meme.csv")
    fake_headlines = (
        meme_df[meme_df["is_sarcastic"] == 1]["headline"]
        .dropna()
        .map(clean_headline)
        .loc[lambda s: s.str.len() > 20]
        .drop_duplicates()
        .tolist()
    )

    rng.shuffle(real_headlines)
    rng.shuffle(fake_headlines)
    return real_headlines[: args.real], fake_headlines[: args.fake]


def build_augmented_rows(real_headlines: list[str], fake_headlines: list[str]) -> list[dict]:
    rows = []
    for classification, headlines in [("REAL_EVENT", real_headlines), ("FAKE_NEWS", fake_headlines)]:
        for headline in headlines:
            for tweet in tweet_variants(headline, classification):
                rows.append({
                    "headline": headline,
                    "text": tweet,
                    "text_type": "tweet",
                    "classification": classification,
                })
            for meme in meme_variants(headline):
                rows.append({
                    "headline": headline,
                    "text": meme,
                    "text_type": "meme",
                    "classification": "MEME",
                })
    return rows


def build_training_pairs(aug_df: pd.DataFrame) -> list[InputExample]:
    real_tweets = aug_df[aug_df["classification"] == "REAL_EVENT"]["text"].tolist()
    fake_tweets = aug_df[aug_df["classification"] == "FAKE_NEWS"]["text"].tolist()
    meme_tweets = aug_df[aug_df["classification"] == "MEME"]["text"].tolist()
    real_heads = aug_df[aug_df["classification"] == "REAL_EVENT"]["headline"].tolist()
    fake_heads = aug_df[aug_df["classification"] == "FAKE_NEWS"]["headline"].tolist()

    train_examples = []

    n = min(4000, len(real_tweets), len(real_heads))
    for tw, hl in zip(rng.sample(real_tweets, n), rng.sample(real_heads, n)):
        train_examples.append(InputExample(texts=[tw, hl], label=0.9))

    for j in range(min(2000, len(real_tweets) // 2)):
        if j + 1 < len(real_tweets):
            train_examples.append(InputExample(texts=[real_tweets[j], real_tweets[j + 1]], label=0.85))

    for j in range(min(2000, len(fake_tweets) // 2)):
        if j + 1 < len(fake_tweets):
            train_examples.append(InputExample(texts=[fake_tweets[j], fake_tweets[j + 1]], label=0.75))

    n_neg = min(4000, len(real_tweets), len(meme_tweets))
    for tw, meme in zip(rng.sample(real_tweets, n_neg), rng.sample(meme_tweets, n_neg)):
        train_examples.append(InputExample(texts=[tw, meme], label=0.05))

    n_neg2 = min(2000, len(real_tweets), len(fake_heads))
    for tw, fh in zip(rng.sample(real_tweets, n_neg2), rng.sample(fake_heads, n_neg2)):
        train_examples.append(InputExample(texts=[tw, fh], label=0.1))

    n_med = min(2000, len(fake_tweets), len(fake_heads))
    for tw, fh in zip(rng.sample(fake_tweets, n_med), rng.sample(fake_heads, n_med)):
        train_examples.append(InputExample(texts=[tw, fh], label=0.6))

    # Hard negatives: generic meme-slang should stay far from real-event headlines.
    hard_neg_count = min(3000, len(real_heads) * 2)
    sampled_heads = [rng.choice(real_heads) for _ in range(hard_neg_count)]
    sampled_memes = [rng.choice(HARD_NEGATIVE_MEMES) for _ in range(hard_neg_count)]
    for meme_text, real_hl in zip(sampled_memes, sampled_heads):
        train_examples.append(InputExample(texts=[meme_text, real_hl], label=0.0))

    # Also contrast hard-negative memes against real tweets.
    hard_tweet_count = min(3000, len(real_tweets))
    sampled_real_tweets = rng.sample(real_tweets, hard_tweet_count)
    sampled_memes = [rng.choice(HARD_NEGATIVE_MEMES) for _ in range(hard_tweet_count)]
    for meme_text, real_tw in zip(sampled_memes, sampled_real_tweets):
        train_examples.append(InputExample(texts=[meme_text, real_tw], label=0.0))

    for text1, text2, label in CURATED_POSITIVE_PAIRS + CURATED_NEGATIVE_PAIRS:
        for _ in range(200):
            train_examples.append(InputExample(texts=[text1, text2], label=label))

    rng.shuffle(train_examples)
    return train_examples


def main():
    print("=" * 60)
    print("LOCAL DATASET AUGMENTATION")
    print("=" * 60)

    real_headlines, fake_headlines = load_sources()
    print(f"  Real-event headlines selected : {len(real_headlines)}")
    print(f"  Fake/meme  headlines selected : {len(fake_headlines)}")

    rows = build_augmented_rows(real_headlines, fake_headlines)
    aug_df = pd.DataFrame(rows)
    aug_df.to_csv(args.out, index=False)

    print("\n── Augmented dataset distribution ──────────────────────────")
    print(aug_df.groupby(["classification", "text_type"]).size().to_string())
    print(f"\nTotal rows : {len(aug_df)}")
    print(f"Saved → {args.out}")

    train_examples = build_training_pairs(aug_df)
    with open(args.pairs, "wb") as f:
        pickle.dump(train_examples, f)

    pairs_csv = pd.DataFrame(
        [{"text1": ex.texts[0], "text2": ex.texts[1], "label": ex.label} for ex in train_examples]
    )
    pairs_csv.to_csv("training_pairs.csv", index=False)

    print("\n── Training pairs ──────────────────────────────────────────")
    print(f"  Total pairs : {len(train_examples)}")
    print(f"Saved → {args.pairs}")
    print("Saved → training_pairs.csv")


if __name__ == "__main__":
    main()
