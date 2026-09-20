"""
Lumina - Phase 4 serving backend.

Real search mode: loads a precomputed image embedding index (built by
phase1_pretrain/precompute_embeddings.py from the trained checkpoint),
embeds each text query at request time, and returns the top-k images by
cosine similarity. Falls back to mock mode automatically if the index
file or model dependencies are not available, so the API never crashes
if something is missing - it just tells you clearly via /health.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import os
import time

USE_MOCK = False

INDEX_PATH = os.environ.get("LUMINA_INDEX_PATH", os.path.join(os.path.dirname(__file__), "image_index.pt"))

app = FastAPI(title="Lumina Search API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    scored = []
    for image_id, caption in MOCK_POOL:
        h = hashlib.sha256((query + caption).encode()).hexdigest()
        fake_score = int(h[:8], 16) / 0xFFFFFFFF
        scored.append(SearchResult(image_id=image_id, caption=caption, score=round(fake_score, 4)))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


_real_state = {"loaded": False, "available": False, "error": None}


def _try_load_real():
    if _real_state["loaded"]:
        return
    _real_state["loaded"] = True
    try:
        import sys
        phase1_dir = os.path.join(os.path.dirname(__file__), "..", "..", "phase1_pretrain")
        sys.path.insert(0, os.path.abspath(phase1_dir))
        import torch
        from model import build_model
        from dataset import get_tokenizer

        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"Image index not found at {INDEX_PATH}. "
                f"Run phase1_pretrain/precompute_embeddings.py on Kaggle and copy "
                f"image_index.pt here, or set LUMINA_INDEX_PATH."
            )

        index = torch.load(INDEX_PATH, map_location="cpu")
        tokenizer = get_tokenizer()
        model = build_model(vocab_size=tokenizer.vocab_size)
        checkpoint_path = os.environ.get("LUMINA_CHECKPOINT_PATH")
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            raise FileNotFoundError(
                "LUMINA_CHECKPOINT_PATH not set or file missing - real_search needs "
                "the trained checkpoint (not just the image index) to embed text "
                "queries with the SAME weights used to build the index."
            )
        model.eval()

        _real_state.update({
            "available": True,
            "torch": torch,
            "model": model,
            "tokenizer": tokenizer,
            "image_ids": index["image_ids"],
            "captions": index["captions"],
            "embeddings": index["embeddings"],
        })
    except Exception as e:
        _real_state["available"] = False
        _real_state["error"] = str(e)


def real_search(query: str, top_k: int) -> list[SearchResult]:
    _try_load_real()
    if not _real_state["available"]:
        raise RuntimeError(f"Real search unavailable: {_real_state['error']}")

    torch = _real_state["torch"]
    model = _real_state["model"]
    tokenizer = _real_state["tokenizer"]

    with torch.no_grad():
        enc = tokenizer(query, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        _img_feat_unused, text_feat = model(
            torch.zeros(1, 3, 128, 128),
            enc["input_ids"],
        )
        sims = (text_feat @ _real_state["embeddings"].t()).squeeze(0)
        top = torch.topk(sims, k=min(top_k, sims.shape[0]))

    results = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        results.append(SearchResult(
            image_id=_real_state["image_ids"][idx],
            caption=_real_state["captions"][idx],
            score=round(score, 4),
        ))
    return results


@app.get("/health")
def health():
    if USE_MOCK:
        return {"status": "ok", "mode": "mock"}
    _try_load_real()
    if _real_state["available"]:
        return {"status": "ok", "mode": "real", "index_size": len(_real_state["image_ids"])}
    return {"status": "degraded", "mode": "mock (real unavailable)", "error": _real_state["error"]}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    start = time.perf_counter()
    used_mock = USE_MOCK
    try:
        if USE_MOCK:
            results = mock_search(req.query, req.top_k)
        else:
            results = real_search(req.query, req.top_k)
    except Exception:
        results = mock_search(req.query, req.top_k)
        used_mock = True
    latency_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(query=req.query, mock=used_mock, results=results, latency_ms=round(latency_ms, 2))
