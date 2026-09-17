"""
Phase 1: Dataset loading for contrastive pretraining.

Uses a subset of Flickr30k (image, caption) pairs. On Kaggle, add the
"Flickr30k" or "Flickr8k" dataset via Add Data, then point DATA_ROOT
at it. This loader intentionally caps dataset size (see MAX_SAMPLES)
so a full pass fits inside a single Kaggle session.
"""

import os
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# This dataset's layout (adityajn105/flickr30k on Kaggle):
#   /kaggle/input/datasets/adityajn105/flickr30k/Images/*.jpg
#   /kaggle/input/datasets/adityajn105/flickr30k/captions.txt  (header: image,caption)
DATA_ROOT = os.environ.get("DATA_ROOT", "/kaggle/input/datasets/adityajn105/flickr30k")
IMAGES_SUBDIR = "Images"
CAPTIONS_FILE = "captions.txt"

MAX_SAMPLES = 30_000  # subset cap — keeps an epoch finishable within a Kaggle session
IMG_SIZE = 128


def build_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                              std=[0.26862954, 0.26130258, 0.27577711]),
    ])


class Flickr30kSubset(Dataset):
    """
    Each item: one (image, caption) pair. Flickr30k has ~5 captions/image;
    we keep only the first caption per image to control dataset size and
    avoid the same image dominating a batch.
    """

    def __init__(self, data_root=DATA_ROOT, tokenizer=None, max_len=64,
                 max_samples=MAX_SAMPLES, train=True):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, IMAGES_SUBDIR)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = build_transform(train)

        # Build a filename -> full path lookup by walking the images dir once.
        # Handles both flickr30k_images/*.jpg and flickr30k_images/flickr30k_images/*.jpg
        # style nesting without hardcoding which one this dataset uses.
        self.image_paths = {}
        for root, _dirs, files in os.walk(self.images_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths[fname] = os.path.join(root, fname)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found under {self.images_dir}. "
                f"Check DATA_ROOT / IMAGES_SUBDIR match the attached dataset's real layout."
            )

        captions_path = os.path.join(data_root, CAPTIONS_FILE)
        self.samples = []
        seen_images = set()

        with open(captions_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            header = next(reader, None)  # e.g. ['image', 'caption'] or ['image_name', 'comment_number', 'comment']
            two_col_format = header is not None and len(header) == 2

            for row in reader:
                if not row:
                    continue
                if two_col_format:
                    # format: image,caption  (caption itself may contain commas)
                    image_name = row[0].strip()
                    caption = ",".join(row[1:]).strip()
                else:
                    # format: image_name,comment_number,comment
                    if len(row) < 3:
                        continue
                    image_name = row[0].strip()
                    caption = row[2].strip()

                if not image_name or not caption:
                    continue
                if image_name in seen_images:
                    continue  # keep first caption only per image
                if image_name not in self.image_paths:
                    continue  # caption references an image we don't have on disk
                seen_images.add(image_name)
                self.samples.append((image_name, caption))
                if len(self.samples) >= max_samples:
                    break

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples loaded from {captions_path}. "
                f"Check DATA_ROOT and that the Flickr30k dataset is attached."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]
        img_path = self.image_paths[image_name]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.tokenizer is not None:
            enc = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
        else:
            # placeholder if tokenizer not wired up yet — caller must tokenize separately
            input_ids = caption

        return image, input_ids


def get_tokenizer():
    """
    Uses a pretrained tokenizer (not a pretrained model — just its
    vocab/tokenization rules) so we don't have to train a tokenizer
    from scratch. This does NOT use pretrained weights.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")


if __name__ == "__main__":
    tok = get_tokenizer()
    ds = Flickr30kSubset(tokenizer=tok, max_samples=100)
    print(f"Loaded {len(ds)} samples")
    img, ids = ds[0]
    print(f"image shape: {img.shape}, input_ids shape: {ids.shape}")
