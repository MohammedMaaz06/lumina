# Progress Log

Daily log of training progress, decisions, and blockers. One entry per
commit — this doubles as your build log for the README/LinkedIn post
later, and as your streak record.

Format per entry:
```
## Day N — YYYY-MM-DD
- What I did:
- Metric (if applicable): loss / R@1 / R@5 / R@10
- Blocker or note:
```

---

## Day 1 — (fill in date)
- What I did: Repo scaffolded — model.py (ViT-Tiny + text transformer),
  dataset.py (Flickr30k subset loader), train.py (checkpointed training
  loop), evaluate.py (Recall@K). Ran smoke tests locally (model.py,
  dataset.py both execute cleanly on CPU with dummy/small data).
- Metric: N/A (pre-training)
- Blocker or note: Need to attach Flickr30k dataset on Kaggle and run
  Cell 3/4 sanity checks before starting Cell 5 training.

## Day 2 — 2026-09-11
- What I did: Attached Flickr30k dataset on Kaggle (dharmaputra13/flickr30k), GPU T4 x2
  confirmed active. Hit two issues and fixed both:
  1. First attached input was someone's empty notebook output, not a real dataset —
     caught it with `find /kaggle/input`, removed it, re-searched filtered to Datasets.
  2. Real dataset's captions.txt is comma-delimited with header
     `image_name,comment_number,comment` (not the pipe-delimited results.csv format
     assumed originally) — updated dataset.py to match, and made image path lookup
     walk the images folder once instead of hardcoding folder nesting.
- Metric: N/A (about to run Cell 3/4 sanity checks against the real dataset)
- Blocker or note: DATA_ROOT is now /kaggle/input/flickr30k/Flickr30k — confirm this
  matches your mount path before running dataset.py.

## Day 3 — 2026-09-16
- What I did: Second dataset attempt (dharmaputra13/flickr30k) was also an empty
  notebook-output, not a real dataset — confirmed via `find`. Removed it, found and
  attached the real one: adityajn105/flickr30k (Kaggle Dataset, not Notebook),
  28,803/30,003 images downloaded successfully. Updated dataset.py: DATA_ROOT ->
  /kaggle/input/datasets/adityajn105/flickr30k, IMAGES_SUBDIR -> "Images", and made
  the captions parser handle this dataset's 2-column (image,caption) format instead
  of assuming the 3-column format from before.
- Metric: N/A (about to re-run dataset.py sanity check against real data)
- Blocker or note: Also had to enable Kaggle's Internet toggle (was off by default,
  caused git clone to fail with DNS resolution errors) and re-select GPU T4 x2 after
  session restarts reset it a couple times.

## Day 4 — 2026-09-17
- What I did: Resumed training session on Kaggle. Learned the hard way that
  /kaggle/working does not persist between sessions unless "Save Version" is
  clicked before ending a session — previous session's ~340 training steps were
  lost since it wasn't saved. Restarted training from scratch with the same
  config (batch_size=64, max_hours=4.5). Confirmed loss is dropping again from
  the same starting point (~4.15 at step 20), same healthy trend as before.
- Metric: loss 4.15 at step 20 (training restarted, in progress)
- Blocker or note: Going forward, always click Save Version before ending any
  Kaggle session, even mid-training, so checkpoints in /kaggle/working actually
  persist to the next session.

## Day 5 — 2026-09-17
- What I did: Started Phase 4 serving layer in parallel with ongoing Phase 1
  training. Built FastAPI backend (phase4_serve/backend/main.py) with a
  clearly-labeled MOCK MODE search endpoint — deterministic hash-based fake
  ranking, same response shape real inference will use later. Built React
  frontend (Vite) with a search UI that shows a visible "MOCK MODE" warning
  banner whenever the API reports mock results, so demo results can never be
  mistaken for real model output. Verified end-to-end locally: backend starts,
  /search and /health both return correctly.
- Metric: N/A (serving layer, not training)
- Blocker or note: real_search() in backend/main.py is a documented stub —
  swap it in once Phase 1 has a usable checkpoint. Frontend/backend contract
  won't need to change.

## Day 6 - 2026-09-18
- What I did: Phase 1 pretraining completed all 20 epochs (9360 steps) on Kaggle. Loss dropped from 4.15 (random-init baseline) to ~0.77 by the final epochs. Ran real held-out evaluation with evaluate.py against the final checkpoint (ckpt_epoch20_step9360.pt) on 2000 held-out image-caption pairs never seen during training.
- Metric: Image-to-Text Retrieval - R@1: 22.95%, R@5: 55.70%, R@10: 71.00%. Text-to-Image Retrieval - R@1: 21.15%, R@5: 55.55%, R@10: 71.70%. Random chance baseline is ~0.05% R@1 - model is ~450x better than chance.
- Blocker or note: Phase 1 is functionally complete. Next: implement real_search() in phase4_serve/backend/main.py using this checkpoint, swap USE_MOCK to False, and precompute image embeddings. Also worth starting Phase 2 in parallel.

## Day 8 - 2026-09-21
- What I did: Cleaned up repo hygiene - node_modules and package-lock.json had accidentally been committed, removed from git tracking and added to .gitignore. Located the real ckpt_epoch20_step9360.pt checkpoint on Kaggle after a few tries (input file listing was inconsistent across session refreshes, resolved by re-running find fresh each time). Verified image_index.pt is valid - 500 real Flickr30k embeddings, correct shape.
- Metric: N/A
- Blocker or note: Copying ckpt_epoch20_step9360.pt to /kaggle/working/ for download, then need to set LUMINA_CHECKPOINT_PATH locally and verify real_search() returns real results end-to-end with the actual trained model.

## Day 9 - 2026-09-26
- What I did: Wired up and fully verified real_search() end-to-end with the actual trained checkpoint. Found and fixed a real bug: the original image_index.pt was stale/mismatched (likely built during an incomplete Flickr30k download session), giving weak scores (0.10-0.18) and poor retrieval on exact caption queries. Diagnosed this by comparing live-server results against a fresh Recall@K computation on the same image pool (got 41%/81% R@1/R@5 vs near-random behavior on the server), which proved the checkpoint logic itself was correct but the index was bad. Regenerated image_index.pt fresh from the same checkpoint and re-tested.
- Metric: After the fix, 3 of 4 exact-caption test queries correctly retrieved their own image (2 at rank #1), with similarity scores of 0.68-0.81 (up from 0.10-0.18 with the stale index). Real search latency ~10-15ms after model warmup.
- Blocker or note: Phase 4 mock-to-real transition is now complete and verified. Frontend already built - next step is confirming the actual UI shows real results correctly end-to-end, then move toward Phase 2 or polish/README updates reflecting the working full-stack demo.

## Day 9 (continued) - full-stack verification
- What I did: Confirmed the fix works end-to-end through the actual browser UI, not just curl. Started backend + frontend together, ran a real search through Lumina's search box for an exact indexed caption - got the correct image back at rank #1 with score 0.81, no MOCK MODE banner, recent searches and results toolbar all working correctly against real data.
- Metric: Top-1 exact match confirmed visually in the UI (score 0.8077). First-request latency was ~9.5s (one-time model/tokenizer load cost), expected to drop to ~10-20ms on subsequent searches.
- Blocker or note: Full-stack demo (Phase 1 trained model -> Phase 4 serving -> real UI) is genuinely complete and working. This is a solid milestone to screenshot/record for the portfolio.
