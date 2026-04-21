from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = BASE_DIR / "fixed_approach_v2_results.csv"
DEFAULT_MODEL_DIR = BASE_DIR / "finetuned_event_model_v2"
FALLBACK_MODEL_DIR = BASE_DIR / "finetuned_event_model"


def resolve_model_path() -> str:
    env_model = os.environ.get("MODEL_DIR", "")
    if env_model:
        candidate = BASE_DIR / env_model
        if candidate.exists():
            return str(candidate)
        if Path(env_model).exists():
            return env_model
    if DEFAULT_MODEL_DIR.exists():
        return str(DEFAULT_MODEL_DIR)
    if FALLBACK_MODEL_DIR.exists():
        return str(FALLBACK_MODEL_DIR)
    raise FileNotFoundError("No fine-tuned model directory found for inference.")


@dataclass
class TopicRecord:
    topic_id: int
    label: str
    keywords: list[str]
    news_count: int
    meme_score: float
    topic_text: str
    event_name: str


class EventDetector:
    def __init__(self, model_path: str | None = None, results_path: str | Path = DEFAULT_RESULTS_PATH):
        self.model_path = model_path or resolve_model_path()
        self.results_path = Path(results_path)
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        self.model = SentenceTransformer(self.model_path)
        self.topics = self._load_topics(self.results_path)
        self.topic_embeddings = self.model.encode([topic.topic_text for topic in self.topics])

    def _load_topics(self, results_path: Path) -> list[TopicRecord]:
        df = pd.read_csv(results_path)
        topics: list[TopicRecord] = []
        for _, row in df.iterrows():
            keywords = row["keywords"]
            if isinstance(keywords, str):
                keywords = ast.literal_eval(keywords)
            keywords = [str(k).strip() for k in keywords if str(k).strip()]
            topic_text = ", ".join(keywords[:8])
            event_name = " ".join(word.title() for word in keywords[:3]) if keywords else f"Topic {int(row['topic_id'])}"
            topics.append(
                TopicRecord(
                    topic_id=int(row["topic_id"]),
                    label=str(row["label"]),
                    keywords=keywords,
                    news_count=int(row["news_count"]),
                    meme_score=float(row["meme_score"]),
                    topic_text=topic_text,
                    event_name=event_name,
                )
            )
        return topics

    def detect(self, text: str, top_k: int = 5) -> dict:
        query_embedding = self.model.encode(text)
        sims = util.cos_sim(query_embedding, self.topic_embeddings)[0].tolist()

        scored = []
        for topic, similarity in zip(self.topics, sims):
            scored.append(
                {
                    "topic_id": topic.topic_id,
                    "event_name": topic.event_name,
                    "label": topic.label,
                    "keywords": topic.keywords,
                    "news_count": topic.news_count,
                    "meme_score": round(topic.meme_score, 4),
                    "similarity": round(float(similarity), 4),
                    "topic_text": topic.topic_text,
                }
            )

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        best = scored[0]
        confidence = self._confidence_label(best["similarity"])
        return {
            "query": text,
            "prediction": {
                **best,
                "confidence": confidence,
            },
            "top_matches": scored[:top_k],
        }

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.55:
            return "medium"
        return "low"

    def stats(self) -> dict:
        label_counts: dict[str, int] = {}
        for topic in self.topics:
            label_counts[topic.label] = label_counts.get(topic.label, 0) + 1
        return {
            "topic_count": len(self.topics),
            "label_counts": label_counts,
            "model_path": self.model_path,
            "results_path": str(self.results_path),
        }
