from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

from inference import EventDetector

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DEFAULT_MODEL_DIR = BASE_DIR / "finetuned_event_model_v2"
FALLBACK_MODEL_DIR = BASE_DIR / "finetuned_event_model"
DEFAULT_MODEL_NAME = os.environ.get("MODEL_DIR", "")


def resolve_model_path() -> str:
    if DEFAULT_MODEL_NAME:
        candidate = BASE_DIR / DEFAULT_MODEL_NAME
        if candidate.exists():
            return str(candidate)
        if Path(DEFAULT_MODEL_NAME).exists():
            return DEFAULT_MODEL_NAME
    if DEFAULT_MODEL_DIR.exists():
        return str(DEFAULT_MODEL_DIR)
    if FALLBACK_MODEL_DIR.exists():
        return str(FALLBACK_MODEL_DIR)
    raise FileNotFoundError(
        "No fine-tuned model directory found. Expected 'finetuned_event_model_v2/' or 'finetuned_event_model/'."
    )


MODEL_PATH = resolve_model_path()
model = SentenceTransformer(MODEL_PATH)
detector = EventDetector(model_path=MODEL_PATH)

app = FastAPI(title="Dynamic Event Detector API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimilarityRequest(BaseModel):
    text1: str = Field(..., min_length=1, description="First text to compare")
    text2: str = Field(..., min_length=1, description="Second text to compare")


class BatchItem(BaseModel):
    label: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class MatchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text to compare against candidate items")
    candidates: list[BatchItem] = Field(..., min_length=1)


class DetectRequest(BaseModel):
    tweet: str = Field(..., min_length=1, description="Tweet or short text to classify into an event topic")


def compute_similarity(text1: str, text2: str) -> float:
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return float(util.cos_sim(emb1, emb2).item())


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "topic_count": detector.stats()["topic_count"],
    }


@app.post("/similarity")
def similarity(payload: SimilarityRequest) -> dict:
    score = compute_similarity(payload.text1, payload.text2)
    return {
        "text1": payload.text1,
        "text2": payload.text2,
        "similarity": round(score, 4),
    }


@app.post("/match")
def match(payload: MatchRequest) -> dict:
    if not payload.candidates:
        raise HTTPException(status_code=400, detail="At least one candidate is required.")

    query_embedding = model.encode(payload.query)
    results = []
    for item in payload.candidates:
        candidate_embedding = model.encode(item.text)
        score = float(util.cos_sim(query_embedding, candidate_embedding).item())
        results.append({
            "label": item.label,
            "text": item.text,
            "similarity": round(score, 4),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "query": payload.query,
        "best_match": results[0],
        "results": results,
    }


@app.post("/detect")
def detect(payload: DetectRequest) -> dict:
    return detector.detect(payload.tweet)


@app.get("/stats")
def stats() -> dict:
    return detector.stats()


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
