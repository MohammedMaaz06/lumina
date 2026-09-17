# Lumina — Phase 4 Serving (currently: mock mode)

FastAPI backend + React frontend for the multimodal search demo.

**Status: MOCK MODE.** Phase 1 pretraining hasn't produced a checkpoint
yet, so `/search` returns deterministic placeholder results instead of
real model inference. The response shape is identical to what real mode
will return — see `backend/main.py` for exactly what to implement in
`real_search()` once a checkpoint exists. Nothing else needs to change.

The frontend shows a visible "MOCK MODE" banner whenever `mock: true`
comes back from the API, so there's no risk of mistaking placeholder
results for real ones during a demo.

## Running locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
cd frontend
npm install
npm run dev
```
Then open the printed localhost URL. The dev server proxies `/search`
and `/health` to `localhost:8000` (see `vite.config.js`).

## API

`POST /search`
```json
{ "query": "a dog in a park", "top_k": 5 }
```
→
```json
{
  "query": "a dog in a park",
  "mock": true,
  "results": [
    { "image_id": "mock_001", "caption": "...", "score": 0.59 }
  ],
  "latency_ms": 0.08
}
```

`GET /health` → `{ "status": "ok", "mode": "mock" }`
