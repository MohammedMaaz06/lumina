# Kaggle Notebook Skeleton — Phase 1 Pretraining

Copy each section below into its own Kaggle notebook cell, in order.
Run this **same notebook every session** — it resumes from the latest
checkpoint automatically, so a 4-5hr session picks up exactly where
the last one stopped.

---

### Cell 1 — Setup
```python
!pip install -q transformers

import os
os.environ["DATA_ROOT"] = "/kaggle/input/flickr30k"        # adjust to your attached dataset's path
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"

!nvidia-smi --query-gpu=name,memory.total --format=csv
```

### Cell 2 — Clone your repo (once pushed to GitHub)
```python
!git clone https://github.com/MohammedMaaz06/lumina.git
%cd lumina/phase1_pretrain
```

### Cell 3 — Sanity check the model before burning GPU hours
```python
!python model.py
```
Expect: total params printed, loss close to `ln(batch_size)` on random init.

### Cell 4 — Sanity check the dataset loads
```python
!python dataset.py
```
Expect: sample count printed, tensor shapes for one image/caption pair.

### Cell 5 — Train (this is the cell you re-run every session)
```python
!python train.py --max_hours 4.5 --batch_size 64 --num_epochs 20
```
- `--max_hours` should stay under Kaggle's session limit with margin
  (leave time to save + for the notebook to shut down cleanly).
- Checkpoints land in `/kaggle/working/checkpoints/`. **Kaggle wipes
  `/kaggle/working` between sessions unless you save it as a Dataset
  output or commit the notebook** — see Cell 6.

### Cell 6 — Persist checkpoints across sessions
At the end of each session, before it ends:
```python
# Kaggle auto-saves everything under /kaggle/working when you
# "Save Version" — do this at the end of every session, or your
# checkpoints are lost and Cell 5 restarts from scratch next time.
```
In the Kaggle UI: **Save Version → Save & Run All (or Quick Save)**
before closing the notebook. On the next session, re-run Cells 1-4,
then re-download your last saved version's `/kaggle/working/checkpoints/`
by re-attaching the previous notebook output as an input dataset.

### Cell 7 — Evaluate (run after a few epochs to check real progress)
```python
!python evaluate.py --checkpoint /kaggle/working/checkpoints/ckpt_epoch2_step600.pt
```
Track Recall@1/5/10 over time — this is your real progress signal,
not just the training loss curve.

### Cell 8 — Push checkpoints/logs back to GitHub (small files only)
```python
# Don't push large .pt checkpoint files to GitHub directly (repo bloat).
# Instead: log your Recall@K numbers + loss curve screenshots into a
# markdown file per session, and push THAT daily for your streak.
!git add ../PROGRESS.md
!git commit -m "day N: epoch X, R@1=Y%, R@5=Z%"
!git push
```
