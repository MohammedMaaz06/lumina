"""
Phase 1: Multimodal contrastive pretraining architecture.

Image encoder : ViT-Tiny  (~5-6M params, from scratch)
Text encoder  : small Transformer encoder (~20M params, from scratch)
Objective     : CLIP-style contrastive alignment (InfoNCE)

Everything here is intentionally sized to train from scratch on a
single Kaggle T4/P100 within a 30 hrs/week budget. Do not scale these
numbers up without checking training time first.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vision encoder: ViT-Tiny
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Splits an image into patches and linearly projects each to embed_dim."""

    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)                 # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x


class ViTTiny(nn.Module):
    """
    ViT-Tiny: small enough to train from scratch on a T4 in a few hours/epoch.

    Default config (~5.7M params):
      embed_dim=192, depth=6, heads=3, mlp_ratio=4, patch=16, img_size=128
    """

    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        proj_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, proj_dim)  # projection to shared embedding space

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)                 # (B, N+1, D)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:, 0]                                     # take CLS token
        return F.normalize(self.head(cls_out), dim=-1)        # (B, proj_dim), L2-normalized


# ---------------------------------------------------------------------------
# Text encoder: small Transformer
# ---------------------------------------------------------------------------

class TextTransformer(nn.Module):
    """
    Small text transformer encoder (~20-25M params depending on vocab size).

    Default config: embed_dim=256, depth=6, heads=4, max_len=64
    """

    def __init__(
        self,
        vocab_size=30522,   # default: bert-base-uncased tokenizer vocab size
        max_len=64,
        embed_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        proj_dim=256,
        dropout=0.0,
        pad_token_id=0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, proj_dim)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pad_mask = input_ids.eq(self.pad_token_id)            # (B, L), True where padded
        x = self.token_embed(input_ids) + self.pos_embed[:, :L, :]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # mean-pool over non-pad tokens
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(self.head(pooled), dim=-1)         # (B, proj_dim), L2-normalized


# ---------------------------------------------------------------------------
# Joint model + contrastive loss
# ---------------------------------------------------------------------------

class MultimodalContrastiveModel(nn.Module):
    """Wraps the vision + text encoders and exposes a shared logit_scale."""

    def __init__(self, vision_encoder: ViTTiny, text_encoder: TextTransformer, init_logit_scale=math.log(1 / 0.07)):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

    def forward(self, images, input_ids):
        image_features = self.vision_encoder(images)   # (B, proj_dim)
        text_features = self.text_encoder(input_ids)    # (B, proj_dim)
        return image_features, text_features


def clip_contrastive_loss(image_features, text_features, logit_scale):
    """
    Symmetric InfoNCE loss (CLIP-style).
    image_features, text_features: (B, proj_dim), already L2-normalized.
    """
    scale = logit_scale.exp()
    logits_per_image = scale * image_features @ text_features.t()   # (B, B)
    logits_per_text = logits_per_image.t()

    B = image_features.shape[0]
    targets = torch.arange(B, device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2


def build_model(vocab_size=30522):
    vision_encoder = ViTTiny(
        img_size=128, patch_size=16, embed_dim=192, depth=6, num_heads=3, proj_dim=256
    )
    text_encoder = TextTransformer(
        vocab_size=vocab_size, max_len=64, embed_dim=256, depth=6, num_heads=4, proj_dim=256
    )
    model = MultimodalContrastiveModel(vision_encoder, text_encoder)
    return model


if __name__ == "__main__":
    # quick smoke test — run this locally before touching Kaggle GPU hours
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")

    dummy_images = torch.randn(4, 3, 128, 128)
    dummy_ids = torch.randint(0, 30522, (4, 64))
    dummy_ids[:, 40:] = 0  # simulate padding

    img_feat, txt_feat = model(dummy_images, dummy_ids)
    loss = clip_contrastive_loss(img_feat, txt_feat, model.logit_scale)
    print(f"image_features: {img_feat.shape}, text_features: {txt_feat.shape}")
    print(f"loss (random init, should be ~ln(B)): {loss.item():.4f}")
