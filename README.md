# Lumina — Pretrain → Align → Compress → Serve

An end-to-end multimodal (vision + text) pipeline built from scratch:
contrastive pretraining of a vision-language model, task alignment,
compression (distillation + quantization), and advanced serving
(mixture-of-experts routing + speculative decoding) deployed on
Kubernetes.

Most portfolios fine-tune an existing model behind an API. This repo
does the full pipeline: train the alignment from scratch, evaluate it
rigorously, shrink it for deployment, and serve it with production
inference techniques.

## Status

🚧 **Phase 1: Pretraining — in progress.** See [PROGRESS.md](./PROGRESS.md)
for the daily build log.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Contrastive pretraining (ViT-Tiny + text transformer, CLIP-style) from scratch on Flickr30k subset | 🚧 In progress |
| 2 | Task alignment — fine-tune for captioning / VQA | ⬜ Not started |
| 3 | Compression — knowledge distillation + INT8/GGUF quantization | ⬜ Not started |
| 4 | Advanced serving — MoE expert routing + speculative decoding, deployed on Kubernetes with autoscaling | ⬜ Not started |

## Phase 1 architecture

- **Image encoder**: ViT-Tiny, trained from scratch (~5.7M params) —
  `embed_dim=192, depth=6, heads=3, patch=16, img_size=128`
- **Text encoder**: small Transformer encoder, trained from scratch
  (~20-25M params) — `embed_dim=256, depth=6, heads=4, max_len=64`
  (uses a pretrained *tokenizer* for vocab only — no pretrained weights)
- **Objective**: symmetric InfoNCE (CLIP-style contrastive loss)
- **Data**: Flickr30k subset (~30k image-caption pairs, first caption
  per image), sized to finish within Kaggle's free 30 hrs/week GPU budget
- **Eval**: zero-shot image-to-text and text-to-image Recall@1/5/10 on
  a held-out split

## Why this scope

Compute is a real constraint (Kaggle free tier, 30 hrs/week), so model
and dataset sizes are deliberately capped to what can actually finish
training — see [notebooks/kaggle_notebook_skeleton.md](./notebooks/kaggle_notebook_skeleton.md)
for the exact session/checkpointing workflow. A finished small model
beats an abandoned large one.

## Repo structure

```
phase1_pretrain/    model.py, dataset.py, train.py, evaluate.py
phase2_align/       (captioning/VQA fine-tuning — Phase 2)
phase3_compress/    (distillation + quantization — Phase 3)
phase4_serve/       (MoE routing, speculative decoding, K8s deploy — Phase 4)
notebooks/           Kaggle notebook skeleton with checkpointing workflow
checkpoints/          (gitignored — checkpoints are too large for git)
PROGRESS.md           daily build log
```

## Running Phase 1

See [notebooks/kaggle_notebook_skeleton.md](./notebooks/kaggle_notebook_skeleton.md)
for the full Kaggle workflow. Locally (CPU smoke test only, not real training):

```bash
cd phase1_pretrain
python model.py      # architecture + param count sanity check
python dataset.py    # dataset loading sanity check
```
