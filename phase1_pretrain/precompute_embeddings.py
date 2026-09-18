"""
Phase 1 -> Phase 4 bridge: precompute image embeddings from the trained
checkpoint, so real-time search only has to embed the text query (fast)
instead of re-embedding every image per request (slow, wasteful).

Run this once on Kaggle (needs GPU + the dataset), download the output
file, and drop it into phase4_serve/backend/ for real_search() to load.

Output: a single .pt file containing:
  - image_ids: list[str]        (filenames)
  - captions: list[str]         (their ground-truth captions, for display)
  - embeddings: FloatTensor     (N, proj_dim), L2-normalized
"""

import argparse
import torch
from torch.utils.data import DataLoader

from model import build_model
from dataset import Flickr30kSubset, get_tokenizer


@torch.no_grad()
def precompute(checkpoint_path, output_path, max_images=500, batch_size=64, data_root=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    ds_kwargs = {"tokenizer": tokenizer, "train": False, "max_samples": max_images}
    if data_root:
        ds_kwargs["data_root"] = data_root
    dataset = Flickr30kSubset(**ds_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(vocab_size=tokenizer.vocab_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch={ckpt['epoch']}, step={ckpt['step']}")

    all_embeddings = []
    all_image_ids = [name for name, _caption in dataset.samples[:max_images]]
    all_captions = [caption for _name, caption in dataset.samples[:max_images]]

    for images, _input_ids in loader:
        images = images.to(device)
        img_feat, _txt_feat = model(images, torch.zeros(images.shape[0], 1, dtype=torch.long, device=device))
        all_embeddings.append(img_feat.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Computed embeddings: {embeddings.shape}")

    torch.save({
        "image_ids": all_image_ids,
        "captions": all_captions,
        "embeddings": embeddings,
    }, output_path)
    print(f"Saved index to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="/kaggle/working/image_index.pt")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    precompute(args.checkpoint, args.output, args.max_images, args.batch_size, args.data_root)
