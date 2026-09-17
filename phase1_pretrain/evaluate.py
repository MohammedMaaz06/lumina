"""
Phase 1: Evaluation — image-to-text and text-to-image Recall@K.

This is the metric that actually proves the model learned something
(vs. just "loss went down"). Run this against a held-out split that
was never seen during training.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from model import build_model
from dataset import Flickr30kSubset, get_tokenizer


@torch.no_grad()
def compute_embeddings(model, loader, device):
    model.eval()
    all_img_feats, all_txt_feats = [], []
    for images, input_ids in loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        img_feat, txt_feat = model(images, input_ids)
        all_img_feats.append(img_feat.cpu())
        all_txt_feats.append(txt_feat.cpu())
    return torch.cat(all_img_feats), torch.cat(all_txt_feats)


def recall_at_k(sim_matrix, k_values=(1, 5, 10)):
    """
    sim_matrix: (N, N) similarity, row i = query i, correct match = column i.
    Returns dict {k: recall@k} assuming one correct match per query
    (true for our 1-caption-per-image subset).
    """
    N = sim_matrix.shape[0]
    ranks = torch.argsort(sim_matrix, dim=1, descending=True)  # (N, N)
    targets = torch.arange(N).unsqueeze(1)                     # (N, 1)
    correct_rank_pos = (ranks == targets).float().argmax(dim=1)  # position of correct match

    results = {}
    for k in k_values:
        results[f"R@{k}"] = (correct_rank_pos < k).float().mean().item() * 100
    return results


def evaluate(checkpoint_path, data_root=None, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    ds_kwargs = {"tokenizer": tokenizer, "train": False, "max_samples": 2000}
    if data_root:
        ds_kwargs["data_root"] = data_root
    dataset = Flickr30kSubset(**ds_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(vocab_size=tokenizer.vocab_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch={ckpt['epoch']}, step={ckpt['step']}")

    img_feats, txt_feats = compute_embeddings(model, loader, device)
    sim = img_feats @ txt_feats.t()  # (N, N)

    print("\n--- Image-to-Text Retrieval ---")
    i2t = recall_at_k(sim)
    for k, v in i2t.items():
        print(f"{k}: {v:.2f}%")

    print("\n--- Text-to-Image Retrieval ---")
    t2i = recall_at_k(sim.t())
    for k, v in t2i.items():
        print(f"{k}: {v:.2f}%")

    return {"image_to_text": i2t, "text_to_image": t2i}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    evaluate(args.checkpoint, data_root=args.data_root, batch_size=args.batch_size)
