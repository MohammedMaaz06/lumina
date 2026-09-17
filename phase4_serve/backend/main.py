"""
Lumina — Phase 4 serving backend.

*** MOCK MODE ***
Phase 1 (contrastive pretraining) hasn't produced a usable checkpoint yet,
so /search currently returns deterministic, clearly-fake results instead
of running real CLIP-style inference. This lets the frontend and API
contract get built and tested now, without blocking on training.

To go live once a real checkpoint exists:
  1. Implement `real_search()` below — load the trained model from
     phase1_pretrain/model.py, embed the query text, compare against
     precomputed image embeddings (FAISS or a simple matrix, per the
     original project plan).
  2. Flip USE_MOCK to False.
Nothing else in this file or the frontend needs to change — the response
shape is identical between mock and real mode.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import time

USE_MOCK = True  # flip to False once real_search() is implemented

app = FastAPI(title="Lumina Search API", version="0.1.0-mock")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    image_id: str
    caption: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mock: bool
    results: list[SearchResult]
    latency_ms: float


# A small fixed pool of fake "images" so mock results are stable and
# obviously placeholder — swap for real Flickr30k images once wired up.
MOCK_POOL = [
    ("mock_001", "A person walking a dog in a park."),
    ("mock_002", "Two children playing with a red ball."),
    ("mock_003", "A city street at night with neon signs."),
    ("mock_004", "A mountain landscape with a lake in the foreground."),
    ("mock_005", "A chef preparing food in a busy kitchen."),
    ("mock_006", "A group of friends laughing at a cafe table."),
    ("mock_007", "A cyclist riding through an empty road at sunrise."),
    ("mock_008", "A dog catching a frisbee mid-air in a field."),
]


def mock_search(query: str, top_k: int) -> list[SearchResult]:
    """
    Deterministic fake ranking: hash the query + each caption to produce
    a stable pseudo-similarity score, so the same query always returns
    the same order (useful for demoing/testing the UI) without pretending
    to be a real model.
    """
    scored = []
    for image_id, caption in MOCK_POOL:
        h = hashlib.sha256((query + caption).encode()).hexdigest()
        fake_score = int(h[:8], 16) / 0xFFFFFFFF  # -> [0, 1]
        scored.append(SearchResult(image_id=image_id, caption=caption, score=round(fake_score, 4)))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def real_search(query: str, top_k: int) -> list[SearchResult]:
    """
    TODO once Phase 1 checkpoint exists:
      - load model via phase1_pretrain.model.build_model()
      - load trained weights from a checkpoint path
      - tokenize `query`, get text embedding
      - compare against precomputed image embeddings (build these once,
        offline, over the Flickr30k subset — don't embed images per-request)
      - return top_k by cosine similarity
    """
    raise NotImplementedError("real_search() not implemented yet — no trained checkpoint available")


@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock" if USE_MOCK else "real"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    start = time.perf_counter()
    results = mock_search(req.query, req.top_k) if USE_MOCK else real_search(req.query, req.top_k)
    latency_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(query=req.query, mock=USE_MOCK, results=results, latency_ms=round(latency_ms, 2))
