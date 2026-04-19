
# ═══════════════════════════════════════════════════════════════════════════
# FIX PIPELINE — All 3 Issues Resolved
# Paste this as a NEW cell in your Colab notebook (run AFTER the existing
# cells that built scores_df and topic_model have already executed).
# ═══════════════════════════════════════════════════════════════════════════

import requests
import numpy as np
import pandas as pd
import time

# ── FIX 1: GDELT Debug + Switch to v1 ─────────────────────────────────────

def _test_gdelt_debug(query: str) -> dict:
    """
    Run once to diagnose the GDELT v2 failure.
    Returns a dict with status_code and article_count.
    """
    params = {
        "query":      query,
        "mode":       "artlist",
        "format":     "json",
        "maxrecords": 10,
    }
    try:
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params, timeout=15
        )
        print(f"  Status : {r.status_code}")
        print(f"  Preview: {r.text[:300]}")
        data = r.json()
        count = len(data.get("articles", []))
        print(f"  Articles found: {count}")
        return {"status": r.status_code, "count": count}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"status": -1, "count": 0}

print("=== GDELT v2 Diagnostic ===")
print("[australia bushfire climate]")
_test_gdelt_debug("australia bushfire climate")
print()
print("[iran airplane missile]")
_test_gdelt_debug("iran airplane missile")
print()

# ── GDELT v1 — simpler, more reliable ─────────────────────────────────────

def verify_gdelt_v1(keywords: list, delay: float = 0.5) -> int:
    """
    Query GDELT v1 full-text search.
    Uses only the top-2 alpha-only keywords (>3 chars) to avoid query errors.
    Adds a small delay to avoid 429 rate-limiting.
    Returns the number of articles found (0–50).
    """
    clean_kw = [
        k.lower() for k in keywords
        if len(k) > 3 and k.isalpha()
    ]
    if len(clean_kw) < 2:
        return 0

    query = "+".join(clean_kw[:2])
    url = (
        f"https://api.gdeltproject.org/api/v1/search_ftxtsearch/search_ftxtsearch"
        f"?query={query}&output=artlist&maxrecords=50&format=json"
    )
    try:
        time.sleep(delay)          # be polite to the API
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return len(r.json().get("articles", []))
        elif r.status_code == 429:
            print(f"  ⚠️  Rate-limited for query='{query}' — sleeping 10s")
            time.sleep(10)
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return len(r.json().get("articles", []))
    except Exception as e:
        print(f"  GDELT error for {clean_kw[:2]}: {e}")
    return 0

# Quick sanity-check for GDELT v1
print("=== GDELT v1 Sanity Check ===")
print(f"australia+bushfire  → {verify_gdelt_v1(['australia', 'bushfire'])} articles")
print(f"iran+airplane       → {verify_gdelt_v1(['iran', 'airplane'])} articles")
print(f"meghan+harry        → {verify_gdelt_v1(['meghan', 'harry'])} articles")
print()

# ── FIX 2: Dynamic Meme Thresholds ────────────────────────────────────────
# scores_df must already exist from the previous pipeline cell

print("=== Meme Score Distribution ===")
scores = scores_df["meme_score"].values
print(f"  Min:             {scores.min():.3f}")
print(f"  Max:             {scores.max():.3f}")
print(f"  Mean:            {scores.mean():.3f}")
print(f"  Median:          {np.median(scores):.3f}")
print(f"  25th percentile: {np.percentile(scores, 25):.3f}")
print(f"  75th percentile: {np.percentile(scores, 75):.3f}")
print()

LOW_THRESHOLD  = float(np.percentile(scores, 25))   # bottom 25% → REAL candidates
HIGH_THRESHOLD = float(np.percentile(scores, 75))   # top 25%    → MEME candidates

print(f"  ✅ REAL EVENT if meme_score < {LOW_THRESHOLD:.3f}")
print(f"  ✅ MEME       if meme_score > {HIGH_THRESHOLD:.3f}")
print()

# ── FIX 3 + 4: Gaming Noise Filter + Full Classification ──────────────────

NOISE_KEYWORDS = {
    # Gaming
    "hearthstone", "xbox", "playstation", "fifa", "fortnite", "minecraft",
    "twitch", "stream", "gaming", "league", "legends", "pokemon", "nintendo",
    "gamer", "gameplay", "ps5", "preorder", "warzone", "valorant", "apex",
    "gta", "rockstar", "redemption", "deadred", "overwatch", "blizzard",
    "esports", "halo", "pubg", "roblox", "streamer", "twitchtv",
    # Pop culture / social noise
    "kardashian", "taylor", "swift", "bieber", "tiktok",
}

def is_gaming_noise(keywords: list) -> bool:
    """Return True if any keyword matches the noise set."""
    return bool(set(k.lower() for k in keywords) & NOISE_KEYWORDS)


print("=== Running Fixed Classification ===")
final_results = []

for i, (_, row) in enumerate(scores_df.iterrows()):
    keywords   = row["keywords"]
    meme_score = row["meme_score"]

    if is_gaming_noise(keywords):
        # Skip GDELT call for gaming — save API quota
        label      = "NOISE (GAMING)"
        news_count = 0
    else:
        news_count = verify_gdelt_v1(keywords, delay=0.4)

        if news_count > 10 and meme_score < LOW_THRESHOLD:
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

    # Progress log every 25 topics
    if (i + 1) % 25 == 0:
        print(f"  Processed {i + 1}/{len(scores_df)} topics...")

results_df = pd.DataFrame(final_results)
results_df.to_csv("fixed_approach_v2_results.csv", index=False)

print("\n=== RESULTS SUMMARY ===")
print(results_df["label"].value_counts().to_string())
print()

print("=== REAL EVENTS ===")
real = results_df[results_df["label"] == "REAL EVENT"][
    ["topic_id", "keywords", "news_count", "meme_score", "label"]
]
print(real.to_string() if not real.empty else "  (none found — try lowering news_count threshold)")
print()

print("=== POSSIBLE EVENTS ===")
possible = results_df[results_df["label"] == "POSSIBLE EVENT"][
    ["topic_id", "keywords", "news_count", "meme_score", "label"]
]
print(possible.to_string() if not possible.empty else "  (none found)")
print()

print("=== TOP MEMES ===")
memes = results_df[results_df["label"] == "MEME"][
    ["topic_id", "keywords", "news_count", "meme_score", "label"]
].sort_values("meme_score", ascending=False).head(10)
print(memes.to_string() if not memes.empty else "  (none found)")

display(results_df.head(25))
