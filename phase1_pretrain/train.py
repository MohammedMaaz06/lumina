"""
Phase 1: Training loop with checkpointing.

Designed around Kaggle's session constraints: saves a checkpoint every
N steps AND at the end of every epoch, and can resume from the latest
checkpoint automatically. Run this same script every session — it
picks up where it left off.
"""

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from model import build_model, clip_contrastive_loss
from dataset import Flickr30kSubset, get_tokenizer

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/checkpoints")
CHECKPOINT_EVERY_STEPS = 200
LOG_EVERY_STEPS = 20


def get_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not ckpts:
        return None
    # filenames are ckpt_epoch{E}_step{S}.pt — sort by epoch then step
    def sort_key(fname):
        parts = fname.replace(".pt", "").split("_")
        epoch = int(parts[1].replace("epoch", ""))
        step = int(parts[2].replace("step", ""))
        return (epoch, step)
    ckpts.sort(key=sort_key)
    return os.path.join(checkpoint_dir, ckpts[-1])


def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch}_step{step}.pt")
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"[checkpoint] saved {path}")
    return path


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    latest = get_latest_checkpoint(checkpoint_dir)
    if latest is None:
        print("[checkpoint] none found, starting from scratch")
        return 0, 0
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[checkpoint] resumed from {latest} (epoch={ckpt['epoch']}, step={ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]


def train(
    max_hours=4.5,          # stay under Kaggle session limits with margin
    batch_size=64,
    lr=3e-4,
    num_epochs=20,
    data_root=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    tokenizer = get_tokenizer()
    ds_kwargs = {"tokenizer": tokenizer}
    if data_root:
        ds_kwargs["data_root"] = data_root
    dataset = Flickr30kSubset(**ds_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)
    print(f"[setup] dataset size={len(dataset)}, steps/epoch={len(loader)}")

    model = build_model(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    start_epoch, global_step = load_checkpoint(model, optimizer, CHECKPOINT_DIR, device)

    start_time = time.time()
    time_budget_seconds = max_hours * 3600

    model.train()
    for epoch in range(start_epoch, num_epochs):
        for batch_idx, (images, input_ids) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)

            optimizer.zero_grad()
            image_features, text_features = model(images, input_ids)
            loss = clip_contrastive_loss(image_features, text_features, model.logit_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0:
                elapsed = time.time() - start_time
                print(f"[epoch {epoch}] step {global_step} | loss {loss.item():.4f} | "
                      f"elapsed {elapsed/60:.1f}min")

            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(model, optimizer, epoch, global_step, CHECKPOINT_DIR)

            # time-based stop: always exit cleanly with a checkpoint, never
            # get killed mid-step by Kaggle's session limit
            if time.time() - start_time > time_budget_seconds:
                print("[time budget] reached max_hours for this session, saving and stopping")
                save_checkpoint(model, optimizer, epoch, global_step, CHECKPOINT_DIR)
                return

        # end-of-epoch checkpoint regardless of step count
        save_checkpoint(model, optimizer, epoch + 1, global_step, CHECKPOINT_DIR)

    print("[done] reached num_epochs, training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_hours", type=float, default=4.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    train(
        max_hours=args.max_hours,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        data_root=args.data_root,
    )
