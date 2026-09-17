"""
check_clip_pipeline.py
======================
Decisive sanity check that your training preprocessing matches what
FashionCLIP expects — catches normalization AND channel-order mismatches
that let training succeed but quietly suppress the accuracy ceiling.

It runs each sample image two ways and compares the resulting image
embeddings:

  (A) YOUR pipeline : load npy (0-255 RGB) -> /255 -> CLIP-norm -> CHW
                      -> model.get_image_features
  (B) REFERENCE     : CLIPProcessor(images=PIL_image) -> model.get_image_features

If (A) and (B) agree (cosine ~0.99), your preprocessing is exactly right.
A low cosine means a residual mismatch — most often channel order (BGR/RGB)
or a normalization scale error.

Two extra probes:
  * Semantic  : "a shoe with a buckle" vs "a plain shoe" — should score
                higher on buckled shoes.
  * Channel   : "a red shoe" vs "a blue shoe" — a decisive RGB/BGR test,
                because a channel swap literally swaps red<->blue.

Usage
-----
    pip install transformers torch pillow numpy

    python src/check_clip_pipeline.py `
        --npy-dir   "E:/fashion-data/npy/shoes_embellishments/01" `
        --image-dir "E:/fashion-data/01-RAW/shoes_emb_md_300" `
        --n 8

--image-dir is needed only for the reference path (it re-reads the raw
image via PIL). Stems are matched between the two dirs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL = "patrickjohncyh/fashion-clip"


def your_pipeline_tensor(npy_path: Path) -> torch.Tensor:
    """Reproduce dataset.py's fixed preprocessing exactly."""
    img = np.load(npy_path).astype(np.float32)      # HxWx3, 0-255, RGB
    img = img / 255.0                               # the fix
    img = (img - CLIP_MEAN) / (CLIP_STD + 1e-8)
    return torch.from_numpy(img).permute(2, 0, 1)   # CxHxW


def find_raw(image_dir: Path, stem: str) -> Path | None:
    for p in image_dir.rglob(f"{stem}.*"):
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


@torch.no_grad()
def _clip_image_embed(model, feats):
    """
    Normalize a CLIP image embedding robustly.
    Your CLIP build returns BaseModelOutputWithPooling from get_image_features.
    Its pooler_output is already the 512-d projected embedding — detect by
    dimension and use directly; never re-apply visual_projection.
    """
    if torch.is_tensor(feats):
        print(f"  [debug] tensor path, shape={feats.shape}, norm={feats.norm():.4f}")
        return F.normalize(feats, dim=-1)
    # Object path — try image_embeds first (CLIPOutput from full forward())
    if getattr(feats, "image_embeds", None) is not None:
        print(f"  [debug] image_embeds path, norm={feats.image_embeds.norm():.4f}")
        return F.normalize(feats.image_embeds, dim=-1)
    # pooler_output: already projected if dim matches out_features
    pooled = feats.pooler_output
    print(f"  [debug] pooler path, shape={pooled.shape}, norm={pooled.norm():.4f}")
    out_dim = model.visual_projection.out_features
    if pooled.shape[-1] == out_dim:
        return F.normalize(pooled, dim=-1)          # already 512 → use directly
    return F.normalize(model.visual_projection(pooled), dim=-1)  # raw → project


@torch.no_grad()
def embed_pipeline(model, tensor, device):
    feats = model.get_image_features(pixel_values=tensor.unsqueeze(0).to(device))
    return _clip_image_embed(model, feats)


@torch.no_grad()
def embed_reference(model, processor, pil, device):
    inputs = processor(images=pil, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    return _clip_image_embed(model, feats)


@torch.no_grad()
def text_probe(model, processor, pil, pos, neg, device) -> float:
    """
    P(pos > neg) computed from image/text embeddings directly.
    Does not rely on logits_per_image so avoids the object-return issue.
    """
    img_inputs = processor(images=pil, return_tensors="pt").to(device)
    img_emb = _clip_image_embed(model, model.get_image_features(**img_inputs))  # [1,D]

    txt_inputs = processor(text=[pos, neg], return_tensors="pt", padding=True).to(device)
    txt_feats = model.get_text_features(
        input_ids=txt_inputs["input_ids"],
        attention_mask=txt_inputs["attention_mask"],
    )
    if not torch.is_tensor(txt_feats):
        txt_feats = getattr(txt_feats, "text_embeds", None) or txt_feats.pooler_output
    txt_emb = F.normalize(txt_feats, dim=-1)  # [2,D]

    scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else torch.tensor(100.0)
    sims = (img_emb @ txt_emb.T).squeeze(0) * scale
    probs = torch.softmax(sims, dim=0)
    return float(probs[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npy-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=8, help="Number of sample images")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()

    npys = sorted(args.npy_dir.glob("*.npy"))[: args.n]
    if not npys:
        raise SystemExit(f"No .npy files in {args.npy_dir}")

    print(f"\nFeature-parity check on {len(npys)} images")
    print(f"{'stem':<28} {'cos(pipeline,ref)':>18}")
    print("-" * 48)
    cosines = []
    for npy in npys:
        raw = find_raw(args.image_dir, npy.stem)
        if raw is None:
            print(f"{npy.stem:<28} {'(no raw image)':>18}")
            continue
        pil = Image.open(raw).convert("RGB")
        a = embed_pipeline(model, your_pipeline_tensor(npy), device)
        b = embed_reference(model, processor, pil, device)
        cos = float(F.cosine_similarity(a, b).item())
        cosines.append(cos)
        print(f"{npy.stem:<28} {cos:>18.4f}")

    if cosines:
        mean_cos = sum(cosines) / len(cosines)
        print("-" * 48)
        print(f"{'MEAN':<28} {mean_cos:>18.4f}")
        verdict = ("PASS — preprocessing matches CLIP" if mean_cos > 0.98 else
                   "SUSPECT — small mismatch, check resize/interp" if mean_cos > 0.90 else
                   "FAIL — likely channel order or scale mismatch")
        print(f"\nVerdict: {verdict}")
        print("  (padding/resize differences from CLIP's own resize cause a tiny\n"
              "   gap; >0.98 means your normalization and channel order are correct.)")

    # Semantic + channel probes on the first raw image found
    for npy in npys:
        raw = find_raw(args.image_dir, npy.stem)
        if raw:
            pil = Image.open(raw).convert("RGB")
            print("\nSemantic/channel probes on:", npy.stem)
            print(f"  P('a shoe with a buckle' > 'a plain shoe') = "
                  f"{text_probe(model, processor, pil, 'a shoe with a buckle', 'a plain shoe', device):.3f}")
            print(f"  P('a red shoe' > 'a blue shoe')            = "
                  f"{text_probe(model, processor, pil, 'a red shoe', 'a blue shoe', device):.3f}")
            print("  (the color probe is a decisive RGB/BGR check — swapped\n"
                  "   channels invert red<->blue and flip this probability.)")
            break


if __name__ == "__main__":
    main()