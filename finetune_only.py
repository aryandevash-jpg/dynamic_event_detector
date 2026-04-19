import os, random, importlib.util, ast
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader

DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None
ACCELERATE_AVAILABLE = importlib.util.find_spec("accelerate") is not None

print("Loading results...")
results_df = pd.read_csv("fixed_approach_results.csv")
df = None

print("STEP 6 — Fine-tuning SBERT …")
model = SentenceTransformer("all-MiniLM-L6-v2")

label_texts = {}
for _, row in results_df.iterrows():
    lbl = row["label"]
    kws = row["keywords"] if isinstance(row["keywords"], list) else ast.literal_eval(str(row["keywords"]))
    
    texts = [" ".join(kws)] * 5   # lightweight proxy
    label_texts.setdefault(lbl, []).extend(texts)

real_texts   = label_texts.get("REAL EVENT", [])
meme_texts_l = label_texts.get("MEME", []) + label_texts.get("NOISE (GAMING)", [])
poss_texts   = label_texts.get("POSSIBLE EVENT", [])

print(f"  Texts available — REAL: {len(real_texts)}  MEME: {len(meme_texts_l)}  POSSIBLE: {len(poss_texts)}")

train_examples = []
for j in range(min(500, len(real_texts) // 2)):
    if j + 1 < len(real_texts):
        train_examples.append(InputExample(texts=[real_texts[j], real_texts[j+1]], label=0.9))
for j in range(min(500, len(meme_texts_l) // 2)):
    if j + 1 < len(meme_texts_l):
        train_examples.append(InputExample(texts=[meme_texts_l[j], meme_texts_l[j+1]], label=0.9))
n = min(1000, len(real_texts), len(meme_texts_l))
for r, m in zip(random.sample(real_texts, n), random.sample(meme_texts_l, n)):
    train_examples.append(InputExample(texts=[r, m], label=0.05))
for j in range(min(300, len(poss_texts) // 2)):
    if j + 1 < len(poss_texts):
        train_examples.append(InputExample(texts=[poss_texts[j], poss_texts[j+1]], label=0.6))

print(f"  Training pairs: {len(train_examples)}")

if not real_texts or not meme_texts_l:
    print("  ⚠  Fine-tuning skipped: need both REAL EVENT and MEME-labeled texts.")
elif len(train_examples) < 10:
    print("  ⚠  Too few pairs — skipping fine-tuning.")
elif not DATASETS_AVAILABLE:
    print("  ⚠  Fine-tuning skipped: missing optional dependency `datasets`.")
elif not ACCELERATE_AVAILABLE:
    print("  ⚠  Fine-tuning skipped: missing optional dependency `accelerate`.")
else:
    ft_model         = SentenceTransformer("all-MiniLM-L6-v2")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss       = losses.CosineSimilarityLoss(model=ft_model)

    print("  Training …")
    ft_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=max(1, len(train_dataloader) // 10),
        show_progress_bar=True,
    )

    OUTPUT_DIR = "finetuned_event_model"
    ft_model.save(OUTPUT_DIR)
    print(f"\n  ✅  Fine-tuned model saved → ./{OUTPUT_DIR}/")
