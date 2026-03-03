"""
audit_zeroshot.py
=================
Zero-shot multi-label attribute audit using FashionCLIP (or plain CLIP).

For each image it computes a cosine-similarity score for every attribute
using contrastive prompt pairs (positive vs negative description).
Each attribute is independent → naturally multi-label.

The image encoder runs ONCE per image; text embeddings are cached.
On CPU with 600 images and 10 attributes this takes ~3-5 minutes.

Usage
-----
    pip install transformers torch Pillow pandas tqdm

    # From raw images (jpg/png/webp):
    python src/audit_zeroshot.py \
        --image-dir  "E:/fashion-data/01-RAW/jackets_women_zalando" \
        --out-csv    "E:/fashion-data/csv/audit_zeroshot.csv" \
        --threshold  0.5

    # From preprocessed .npy files:
    python src/audit_zeroshot.py \
        --npy-dir    "E:/fashion-data/npy/jackets/01" \
        --out-csv    "E:/fashion-data/csv/audit_zeroshot.csv"

    # Override model (default: patrickjohncyh/fashion-clip):
    python src/audit_zeroshot.py \
        --image-dir  "..." \
        --model      "openai/clip-vit-large-patch14"

Output
------
    audit_zeroshot.csv  — one row per image, columns:
        name, <attr>_score, <attr>_pred, ...

    audit_summary.txt   — per-attribute stats printed to console + saved
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Attribute definitions
# ---------------------------------------------------------------------------
# Each attribute has:
#   "pos": text describing the garment WHEN the attribute IS present
#   "neg": text describing the garment when the attribute is ABSENT
#
# Contrastive pairs give much more reliable scores than single prompts.
# Tweak these freely — prompt wording is the main lever for zero-shot quality.
# ---------------------------------------------------------------------------

ATTRIBUTES: dict[str, dict[str, str]] = {
    # ── Global / silhouette ──────────────────────────────────────────────
    "ankle_length": {
        "pos": "a full-length coat or jacket reaching down to the ankles",
        "neg": "a short or hip-length jacket",
    },
    "asymmetric": {
        "pos": "a jacket with an asymmetric hem, collar or zip closure",
        "neg": "a jacket with a symmetric, even silhouette",
    },
    "cropped": {
        "pos": "a cropped jacket or bolero ending above the hips or at the waist",
        "neg": "a hip-length or longer jacket",
    },
    # ── Mid-scale ────────────────────────────────────────────────────────
    "belted": {
        "pos": "a jacket or coat with a visible belt, sash or waist tie",
        "neg": "a jacket with no belt or waist tie",
    },
    "drawstring": {
        "pos": "a jacket with a visible drawstring at the hood, hem or waist",
        "neg": "a jacket with no drawstring",
    },
    "epaulette": {
        "pos": "a jacket with epaulettes or structured shoulder strap details",
        "neg": "a jacket with plain shoulders and no epaulettes",
    },
    # ── Local / fine-grained ─────────────────────────────────────────────
    "chest_pocket": {
        "pos": "a jacket with a visible chest pocket on the front",
        "neg": "a jacket with no chest pocket",
    },
    # ── Embellishments ────────────────────────────────────────────────────
    "embroidery": {
        "pos": "a garment decorated with visible embroidery, stitched patterns or needlework",
        "neg": "a plain garment with no embroidery",
    },
    "eyelet": {
        "pos": "a garment with visible eyelets, grommets or punched hole details",
        "neg": "a garment with no eyelet or grommet details",
    },
    "feather": {
        "pos": "a garment trimmed or decorated with feathers",
        "neg": "a garment with no feather trim or decoration",
    },
    "flower": {
        "pos": "a garment with floral appliqué, fabric flowers or 3D flower embellishments",
        "neg": "a garment with no floral appliqué or fabric flowers",
    },
    "fringe": {
        "pos": "a garment with visible fringe, tassel trim or frayed edge decoration",
        "neg": "a garment with no fringe or tassel trim",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL    = "patrickjohncyh/fashion-clip"
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_from_image_file(path: Path, size: int = 224) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"  WARNING: could not load {path.name}: {e}")
        return None


def load_from_npy(path: Path) -> Image.Image | None:
    try:
        arr = np.load(path).astype(np.uint8)   # HxWx3 float32 → uint8
        return Image.fromarray(arr)
    except Exception as e:
        print(f"  WARNING: could not load {path.name}: {e}")
        return None


def collect_images(
    image_dir: Path | None,
    npy_dir: Path | None,
) -> list[tuple[str, Image.Image]]:
    """Return list of (stem, PIL Image) pairs."""
    items = []

    if image_dir is not None:
        paths = sorted(
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Found {len(paths)} images in {image_dir}")
        for p in paths:
            img = load_from_image_file(p)
            if img is not None:
                items.append((p.stem, img))

    elif npy_dir is not None:
        paths = sorted(npy_dir.glob("*.npy"))
        print(f"Found {len(paths)} .npy files in {npy_dir}")
        for p in paths:
            img = load_from_npy(p)
            if img is not None:
                items.append((p.stem, img))

    return items


# ---------------------------------------------------------------------------
# CLIP inference
# ---------------------------------------------------------------------------

def encode_texts(
    processor: CLIPProcessor,
    model: CLIPModel,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Returns L2-normalised text embeddings, shape (N, D)."""
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return F.normalize(embeddings, dim=-1)


def encode_image(
    processor: CLIPProcessor,
    model: CLIPModel,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Returns L2-normalised image embedding, shape (1, D)."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return F.normalize(embedding, dim=-1)


def contrastive_score(
    image_emb: torch.Tensor,   # (1, D)
    pos_emb: torch.Tensor,     # (1, D)
    neg_emb: torch.Tensor,     # (1, D)
) -> float:
    """
    Softmax over [pos_sim, neg_sim] → probability the positive prompt matches.
    This is the standard CLIP zero-shot scoring approach.
    """
    sim_pos = (image_emb @ pos_emb.T).squeeze().item()   # scalar
    sim_neg = (image_emb @ neg_emb.T).squeeze().item()   # scalar
    # Temperature-free softmax (CLIP's logit_scale not needed for relative ranking)
    exp_pos = np.exp(sim_pos)
    exp_neg = np.exp(sim_neg)
    return float(exp_pos / (exp_pos + exp_neg))


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

def run_audit(
    image_dir: Path | None,
    npy_dir: Path | None,
    out_csv: Path,
    model_name: str,
    threshold: float,
    attributes: dict,
    batch_size: int = 32,
) -> pd.DataFrame:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Model  : {model_name}")
    print(f"Threshold: {threshold}\n")

    # Load model
    print("Loading model...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    print("Model loaded.\n")

    # Pre-encode all text prompts (cached — only done once)
    print("Encoding text prompts...")
    attr_names  = list(attributes.keys())
    pos_embeddings = {}
    neg_embeddings = {}
    for attr, prompts in attributes.items():
        pos_embeddings[attr] = encode_texts(processor, model, [prompts["pos"]], device)
        neg_embeddings[attr] = encode_texts(processor, model, [prompts["neg"]], device)
    print(f"  {len(attr_names)} attributes encoded.\n")

    # Load images
    images = collect_images(image_dir, npy_dir)
    if not images:
        raise ValueError("No images found. Check --image-dir or --npy-dir.")
    print(f"Running audit on {len(images)} images...\n")

    # Score each image
    rows = []
    for stem, pil_img in tqdm(images, desc="Scoring"):
        image_emb = encode_image(processor, model, pil_img, device)

        row: dict = {"name": stem}
        for attr in attr_names:
            score = contrastive_score(
                image_emb,
                pos_embeddings[attr],
                neg_embeddings[attr],
            )
            row[f"{attr}_score"] = round(score, 4)
            row[f"{attr}_pred"]  = int(score >= threshold)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    return df


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, attributes: dict, threshold: float, out_path: Path) -> None:
    attr_names = list(attributes.keys())
    lines = []

    lines.append("=" * 60)
    lines.append(f"ZERO-SHOT AUDIT SUMMARY  (threshold={threshold})")
    lines.append(f"Images evaluated : {len(df)}")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"{'Attribute':<20} {'Predicted +ve':>14} {'% positive':>11} {'Mean score':>11} {'Signal?':>8}")
    lines.append("-" * 68)

    for attr in attr_names:
        pred_col  = f"{attr}_pred"
        score_col = f"{attr}_score"
        n_pos     = df[pred_col].sum()
        pct       = 100 * n_pos / len(df)
        mean_sc   = df[score_col].mean()

        # Heuristic signal assessment
        # Good signal: mean score is clearly above or below 0.5
        # Ambiguous: mean score near 0.5, or variance is very low
        score_std = df[score_col].std()
        if mean_sc > 0.65 or mean_sc < 0.35:
            signal = "STRONG"
        elif score_std < 0.04:
            signal = "FLAT ⚠"
        elif score_std < 0.07:
            signal = "WEAK ⚠"
        else:
            signal = "OK"

        lines.append(f"  {attr:<18} {int(n_pos):>14} {pct:>10.1f}% {mean_sc:>11.3f} {signal:>8}")

    lines.append("")
    lines.append("Signal guide:")
    lines.append("  STRONG : mean score far from 0.5 — attribute well-separated in embedding space")
    lines.append("  OK     : reasonable variance — worth fine-tuning")
    lines.append("  WEAK   : low variance — prompts may need rewording, or attribute is rare")
    lines.append("  FLAT   : almost no variance — CLIP may not distinguish this attribute at all")
    lines.append("")
    lines.append("Recommended next steps:")
    lines.append("  1. Inspect STRONG attributes visually — they are your easiest wins")
    lines.append("  2. For WEAK/FLAT: try rewording prompts (see ATTRIBUTES dict in this script)")
    lines.append("  3. Use the _score columns in Label Studio as pre-annotations to speed up labelling")
    lines.append("  4. Attributes where pct_positive < 2% are likely rare — verify a few images manually")

    summary_text = "\n".join(lines)
    print(summary_text)

    out_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSummary saved → {out_path}")


def export_labelstudio_preannotations(
    df: pd.DataFrame,
    attributes: dict,
    threshold: float,
    out_path: Path,
    image_dir: Path | None,
) -> None:
    """
    Export pre-annotations in Label Studio JSON format.
    Each image gets predicted labels pre-filled, ready for human review.
    This halves your annotation time.
    """
    attr_names = list(attributes.keys())
    tasks = []

    for _, row in df.iterrows():
        predicted_labels = [
            attr for attr in attr_names
            if row[f"{attr}_pred"] == 1
        ]

        # Build image URL — adjust prefix to match your Label Studio storage config
        if image_dir is not None:
            image_url = f"/data/local-files/?d={image_dir}/{row['name']}.jpg"
        else:
            image_url = f"/data/local-files/?d={row['name']}.jpg"

        task = {
            "data": {
                "image": image_url,
                "name": row["name"],
            },
            "annotations": [
                {
                    "result": [
                        {
                            "type": "choices",
                            "value": {"choices": predicted_labels},
                            "from_name": "attributes",
                            "to_name": "image",
                        }
                    ],
                    "ground_truth": False,
                    "lead_time": 0,
                }
            ] if predicted_labels else [],
        }
        tasks.append(task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    print(f"Label Studio pre-annotations → {out_path}  ({len(tasks)} tasks)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Zero-shot multi-label attribute audit using FashionCLIP"
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-dir", type=Path,
                     help="Directory containing raw images (searched recursively)")
    src.add_argument("--npy-dir",   type=Path,
                     help="Directory containing preprocessed .npy files")

    p.add_argument("--out-csv",    type=Path, required=True,
                   help="Output CSV path  (e.g. E:/fashion-data/csv/audit_zeroshot.csv)")
    p.add_argument("--model",      type=str,  default=DEFAULT_MODEL,
                   help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                   help="Score threshold for positive prediction (default: 0.5)")
    p.add_argument("--labelstudio", action="store_true",
                   help="Also export a Label Studio pre-annotation JSON")

    args = p.parse_args()

    df = run_audit(
        image_dir  = args.image_dir,
        npy_dir    = args.npy_dir,
        out_csv    = args.out_csv,
        model_name = args.model,
        threshold  = args.threshold,
        attributes = ATTRIBUTES,
    )

    summary_path = args.out_csv.parent / "audit_summary.txt"
    print_summary(df, ATTRIBUTES, args.threshold, summary_path)

    if args.labelstudio:
        ls_path = args.out_csv.parent / "labelstudio_preannotations.json"
        export_labelstudio_preannotations(
            df, ATTRIBUTES, args.threshold, ls_path, args.image_dir
        )


if __name__ == "__main__":
    main()
