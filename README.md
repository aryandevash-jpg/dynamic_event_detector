# Dynamic Event Detector — Alter-Approach Branch

**Distinguish real-world events from internet memes using semantic similarity, topic clustering, and GDELT news verification.**

This branch (`alter-approach`) implements a refined multi-stage pipeline that combines embeddings, unsupervised clustering, meme similarity scoring, and dynamic thresholds to automatically classify trending topics as real events, possible events, memes, or noise.

---

## 🎯 Overview

This project tackles the problem of differentiating genuine world events from viral internet memes and noise in social media text data. The approach is **modular, scalable, and adaptable**:

1. **Text embeddings** with SentenceTransformers (all-MiniLM-L6-v2)
2. **Topic clustering** with BERTopic
3. **Meme similarity scoring** against a known meme corpus
4. **GDELT v1/v2 verification** to validate topics against news archives
5. **Dynamic thresholds** based on percentile distributions
6. **Fine-tuned SBERT** for downstream event detection tasks

---

## 📁 Repository Structure

```
alter-approach/
├── README.md                          # This file
├── run_full_pipeline.py               # Main entry point (Steps 1–6)
├── finetune_only.py                   # Standalone fine-tuning script
├── fix_pipeline.py                    # Standalone classification fixing script
├── compare_models.py                  # Model comparison utility
├── combine_phmeme.py                  # Data preprocessing utility
│
├── finetuned_event_model/             # Saved fine-tuned SBERT model
│   └── README.md                      # Model card (auto-generated)
│
├── Data files (not committed):
│   ├── disaster.csv                   # Disaster/crisis dataset (~1.6M rows)
│   ├── sentiment.csv                  # Sentiment dataset (~10M rows)
│   ├── meme.csv                       # Meme corpus (~4.5M rows)
│   ├── pheme_all.csv                  # PHEME rumor dataset
│   ├── abcnews-date-text.csv          # ABC News archive (~63M)
│   ├── embeddings_v1.npy              # Cached embeddings matrix
│   ├── scores_df.csv                  # Cluster scores checkpoint
│   ├── gdelt_cache.json               # GDELT API cache
│   └── fixed_approach_results.csv     # Final classification results
│
├── Notebooks:
│   ├── Untitled0.ipynb                # Exploratory analysis
│   └── meme_vs_event_classifier.ipynb # Classification experiments
│
└── Artifacts:
    ├── diagram-export-05-05-2026-13_31_15.svg  # Architecture diagram
    └── artifacts_meme_vs_event/               # Experiment outputs
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install sentence-transformers bertopic umap-learn hdbscan pandas numpy requests
pip install datasets accelerate  # Optional: for fine-tuning
```

### Option 1: Run Full Pipeline (Default)

```bash
python3 run_full_pipeline.py
```

This will:
1. Load disaster, sentiment, and meme datasets
2. Generate embeddings (or use cache)
3. Cluster with BERTopic
4. Score clusters by meme similarity
5. Classify via GDELT verification + dynamic thresholds
6. Fine-tune SBERT on results
7. Save all outputs

**Output files:**
- `scores_df.csv` — cluster metadata + meme scores
- `fixed_approach_results.csv` — final labels (REAL EVENT, MEME, NOISE, etc.)
- `finetuned_event_model/` — trained SentenceTransformer

### Option 2: Resume from Cached Scores

If `scores_df.csv` exists, the pipeline automatically skips Steps 1–4:

```bash
python3 run_full_pipeline.py
```

### Option 3: Force Re-compute Everything

```bash
python3 run_full_pipeline.py --force
```

### Option 4: Fine-tune Only

If you already have `fixed_approach_results.csv`:

```bash
python3 finetune_only.py
```

---

## 📊 Pipeline Stages

### Stage 1: Data Loading
- Loads three datasets:
  - **Disaster/Crisis** (first 10k texts) — real events
  - **Sentiment** (first 10k texts) — mixed content
  - **Memes** (first 5k texts) — internet cultural content

### Stage 2: Embeddings
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Caches embeddings in `embeddings_v1.npy` to avoid re-computation
- Dimension: 384D

### Stage 3: BERTopic Clustering
- UMAP dimensionality reduction (5D, 15 neighbors)
- HDBSCAN clustering (min_cluster_size=15)
- Vectorizer: English stopwords, 1–2 grams
- Extracts top keywords per cluster

### Stage 4: Meme Similarity Scoring
- Encodes meme corpus with SBERT
- For each cluster, computes cosine similarity to all memes
- Score = mean max similarity (higher = more meme-like)
- Saves scores in `scores_df.csv`

### Stage 5: GDELT Verification + Classification
- Queries GDELT v1 full-text search with top 2 keywords
- Applies **dynamic percentile thresholds**:
  - **REAL EVENT:** news_count > 10 AND meme_score < 25th percentile
  - **MEME:** meme_score > 75th percentile
  - **POSSIBLE EVENT:** news_count > 5 (intermediate)
  - **NOISE (GAMING):** matches gaming/pop-culture keywords
  - **NOISE:** everything else
- Implements rate-limiting (0.4s–0.6s delays) and caching (`gdelt_cache.json`)

### Stage 6: Fine-tuning SBERT
- Builds training pairs from labeled clusters:
  - **Same-label pairs** (REAL–REAL, MEME–MEME): label=0.9
  - **Cross-label pairs** (REAL vs MEME): label=0.05
  - **Medium pairs** (POSSIBLE EVENT): label=0.6
- CosineSimilarityLoss, 3 epochs, batch_size=16
- Evaluates on test embeddings (Real↔Real similarity vs Real↔Meme)
- Saves to `finetuned_event_model/`

---

## 🔑 Key Design Choices

### Dynamic Thresholds
Instead of fixed cutoffs (e.g., meme_score > 0.5), the pipeline uses **percentile-based thresholds** derived from the actual score distribution:
- 25th percentile = boundary between REAL candidates and NOISE
- 75th percentile = boundary between NOISE and MEME candidates

**Benefit:** Adapts to different datasets automatically.

### GDELT API Strategy
1. Starts with **v2 (faster, structured)**, falls back to **v1 (simpler, more reliable)**
2. Implements exponential backoff and caching to survive rate limits
3. Disables live queries after 5 consecutive failures (cached results still used)
4. Filters by gaming keywords to save API quota

### Gaming Noise Filter
Prevents false positives from gaming/pop-culture discussions:
- Keywords: `fortnite`, `twitch`, `league`, `kardashian`, etc.
- Topics matching these skip GDELT calls entirely

---

## 📈 Results & Evaluation

### Classification Distribution
The pipeline outputs:
- **REAL EVENT** — Topics with high news coverage and low meme similarity
- **POSSIBLE EVENT** — Topics with moderate news coverage
- **MEME** — Topics with high meme similarity
- **NOISE (GAMING)** — Gaming/celebrity topics (filtered)
- **NOISE** — Everything else

### Fine-tuning Evaluation
After training, the model is evaluated on embedding similarities:

```
Eval — Real↔Real: 0.85   (want high)
       Real↔Meme: 0.15   (want low)
```

High Real↔Real similarity + low Real↔Meme similarity = good separation.

---

## 🛠️ Utility Scripts

### `finetune_only.py`
Standalone script to fine-tune SBERT on existing `fixed_approach_results.csv`.

**Use when:**
- Rerunning classification with new parameters
- Experimenting with training hyperparameters
- Resuming fine-tuning without re-clustering

### `fix_pipeline.py`
Debugging & fixing script (also works as standalone cell in Jupyter):
- Tests GDELT v2 connectivity
- Calculates percentile thresholds
- Runs classification with detailed progress
- Exports `fixed_approach_v2_results.csv`

### `compare_models.py`
Compares base vs. fine-tuned SBERT on a test set.

### `combine_phmeme.py`
Preprocessing utility to merge PHEME dataset with meme corpus.

---

## 🔧 Configuration

Edit these constants in `run_full_pipeline.py`:

```python
# Dataset sizes (lines ~107–109)
fake_texts  = fake_news["text"].fillna("").astype(str).apply(clean_text).tolist()[:10_000]
sent_texts  = sentiment["text"].fillna("").astype(str).apply(clean_text).tolist()[:10_000]
meme_texts  = meme_df["headline"].fillna("").astype(str).apply(clean_text).tolist()[:5_000]

# BERTopic parameters (lines ~150–160)
topic_model = BERTopic(
    umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0, random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True),
    ...
)

# Classification thresholds (lines ~226–227)
LOW_THRESHOLD  = float(np.percentile(scores_arr, 25))
HIGH_THRESHOLD = float(np.percentile(scores_arr, 75))

# Fine-tuning parameters (lines ~474–476)
ft_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=max(1, len(train_dataloader) // 10),
)
```

---

## 📦 Dependencies

**Required:**
- `sentence-transformers>=2.2.0` — embeddings
- `pandas>=1.3.0` — data handling
- `numpy>=1.20.0` — numerical ops
- `requests>=2.26.0` — GDELT API calls

**Optional (for clustering):**
- `bertopic>=0.14.0` — topic modeling
- `umap-learn>=0.5.0` — dimensionality reduction
- `hdbscan>=0.8.0` — clustering algorithm
- `scikit-learn>=0.24.0` — feature extraction

**Optional (for fine-tuning):**
- `torch>=1.11.0` — PyTorch backend
- `datasets>=4.0.0` — data loading
- `accelerate>=1.1.0` — distributed training

---

## 🌐 API Notes

### GDELT API Rate Limits
- **v1**: ~1 request/second recommended (no official limit stated)
- **v2**: ~3 requests/minute (undocumented; pipeline uses 0.4–0.6s delays)
- **Strategy**: Caching + backoff + v1 fallback

### Datasets
- **disaster.csv** — https://www.kaggle.com/c/nlp-getting-started
- **sentiment.csv** — custom or Twitter sentiment dataset
- **meme.csv** — internet meme corpus (custom or from PHEME)
- **abcnews-date-text.csv** — ABC News article archive (Kaggle)

---

## 🚦 Troubleshooting

### GDELT Queries Return 0 Results
- Check your keyword length (must be >3 chars, alpha-only)
- Verify network connectivity
- Check `gdelt_cache.json` for cached failures
- Use `fix_pipeline.py` to diagnose

### Fine-tuning Skipped
- Ensure `fixed_approach_results.csv` has both "REAL EVENT" and "MEME" labels
- Install optional deps: `pip install datasets 'accelerate>=1.1.0'`
- Check that you have >10 training pairs

### Embeddings Cache Mismatch
- Delete `embeddings_v1.npy` and re-run (will re-embed all texts)
- Or use `--force` flag

### BERTopic Not Installed
- `pip install bertopic umap-learn hdbscan`
- Pipeline will fall back to random topic assignment if unavailable

---

## 📋 Sample Output

```
STEP 5 — GDELT v2 classification …
  Meme score distribution:
    Min    : 0.158
    25th   : 0.382   ← below = REAL EVENT candidate
    Median : 0.521
    75th   : 0.687   ← above = MEME candidate
    Max    : 0.891

  Classifying topics …
    [  1/142] last: MEME              kw=['fire', 'bushfire', 'australia']
    [ 11/142] last: REAL EVENT        kw=['iran', 'airplane', 'crash']
    [142/142] last: NOISE             kw=['game', 'play', 'win']

  ── Classification summary ──────────────────────────────
  NOISE          68
  REAL EVENT     32
  MEME           26
  POSSIBLE EVENT  14
  NOISE (GAMING)  2

  === REAL EVENTS ===
  topic_id  keywords                          news_count  meme_score label
       42   [iran, airplane, missile]              145       0.312    REAL EVENT
       55   [australia, bushfire, fire]             89       0.291    REAL EVENT
```

---

## 💡 Future Improvements

- [ ] Multi-lingual support (XLM-RoBERTa embeddings)
- [ ] Real-time streaming via social media APIs
- [ ] Interactive web dashboard for classification review
- [ ] Reinforcement learning from human feedback
- [ ] Temporal dynamics (trend velocity, decay rate)
- [ ] Integration with fact-checking APIs (ClaimBuster, etc.)

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{alter_approach_2026,
  title={Dynamic Event Detector — Alter-Approach Branch},
  author={Aryan Devash},
  year={2026},
  url={https://github.com/aryandevash-jpg/dynamic_event_detector/tree/alter-approach}
}
```

---

## 📄 License

[Specify your license here — e.g., MIT, Apache 2.0, etc.]

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit with clear messages
4. Submit a pull request

---

## 📧 Contact

For questions or feedback, reach out to **@aryandevash-jpg** or open an issue on GitHub.

---

**Last updated:** 2026-05-05 | Branch: `alter-approach`
