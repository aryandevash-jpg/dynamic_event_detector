"""
finetune_augmented.py
=====================
Fine-tunes SBERT using the rich training pairs from augment_dataset.py.

This replaces the keyword-proxy approach in finetune_only.py with real
tweet↔headline contrastive pairs, which directly fixes the model's weakness
of not connecting informal tweet language to formal news language.

Usage:
    python3 finetune_augmented.py
    python3 finetune_augmented.py --epochs 5 --batch 32 --out my_model/
"""

import os, sys, random, argparse, pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pairs",  default="augmented_training_pairs.pkl",
                    help="Pickle file from augment_dataset.py")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch",  type=int, default=16)
parser.add_argument("--warmup", type=float, default=0.1,
                    help="Fraction of steps used for warm-up")
parser.add_argument("--out",    default="finetuned_event_model",
                    help="Directory to save the fine-tuned model")
args = parser.parse_args()

# ── Load pairs ─────────────────────────────────────────────────────────────────
if not os.path.exists(args.pairs):
    sys.exit(
        f"❌  '{args.pairs}' not found.\n"
        "    Run augment_dataset.py first:\n"
        "        python3 augment_dataset.py"
    )

print("=" * 60)
print("LOADING TRAINING PAIRS")
print("=" * 60)

with open(args.pairs, "rb") as f:
    train_examples: list[InputExample] = pickle.load(f)

random.shuffle(train_examples)

# Quick distribution summary
labels = [ex.label for ex in train_examples]
pos = sum(1 for l in labels if l >= 0.7)
neg = sum(1 for l in labels if l <= 0.2)
med = len(labels) - pos - neg
print(f"  Total pairs  : {len(train_examples)}")
print(f"  Positive (≥0.7) : {pos}")
print(f"  Medium   (0.2–0.7) : {med}")
print(f"  Negative (≤0.2) : {neg}")

# ── Validation split (10%) ─────────────────────────────────────────────────────
split = int(len(train_examples) * 0.9)
train_set = train_examples[:split]
val_set   = train_examples[split:]
print(f"\n  Train : {len(train_set)}  |  Val : {len(val_set)}")

# ── Model + DataLoader ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINE-TUNING SBERT")
print("=" * 60)

ft_model         = SentenceTransformer("all-MiniLM-L6-v2")
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch)
train_loss       = losses.CosineSimilarityLoss(model=ft_model)
warmup_steps     = max(1, int(len(train_dataloader) * args.epochs * args.warmup))

print(f"  Model       : all-MiniLM-L6-v2")
print(f"  Epochs      : {args.epochs}")
print(f"  Batch size  : {args.batch}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Output      : {args.out}/")

ft_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=args.epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
    output_path=args.out,
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION ON HELD-OUT PAIRS")
print("=" * 60)

correct = 0
for ex in val_set[:200]:
    e1 = ft_model.encode(ex.texts[0])
    e2 = ft_model.encode(ex.texts[1])
    pred_sim = util.cos_sim(e1, e2).item()
    expected_high = ex.label >= 0.7
    predicted_high = pred_sim >= 0.5
    if expected_high == predicted_high:
        correct += 1

accuracy = correct / min(200, len(val_set))
print(f"  Threshold accuracy (sim≥0.5 = positive): {accuracy:.1%}  ({correct}/{ min(200, len(val_set))})")

# Tweet↔headline spot check
print("\n── Spot check: tweet ↔ news headline similarity ───────────────")
test_cases = [
    ("omg ground shaking in Istanbul 😱",       "6.2 magnitude earthquake strikes Turkey",   "HIGH"),
    ("markets bleeding rn sell everything 📉",  "Stock market sees sharp decline",            "HIGH"),
    ("new covid variant spreading fr",          "WHO monitors new virus strain",             "HIGH"),
    ("skibidi toilet no cap lmao",              "6.2 magnitude earthquake strikes Turkey",   "LOW"),
    ("this cat is literally everything 😭",     "WHO declares health emergency",             "LOW"),
]

print(f"  {'SCORE':>6}  EXPECTED  TEXT PAIR")
print("  " + "-" * 70)
for tweet, headline, expected in test_cases:
    e1  = ft_model.encode(tweet)
    e2  = ft_model.encode(headline)
    sim = util.cos_sim(e1, e2).item()
    ok  = "✅" if (sim >= 0.5) == (expected == "HIGH") else "❌"
    print(f"  {sim:>6.3f}  {expected:<8}  {ok}  {tweet[:45]}")

# ── Save ───────────────────────────────────────────────────────────────────────
ft_model.save(args.out)
print(f"\n✅  Fine-tuned model saved → ./{args.out}/")
print("\nTest it anytime with:")
print("    python3 compare_models.py")
