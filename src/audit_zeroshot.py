"""
audit_zeroshot.py
=================
Zero-shot multi-label attribute audit using FashionCLIP (or plain CLIP).

For each image it computes a cosine-similarity score for every attribute
using contrastive prompt pairs (positive vs negative description).
Each attribute is independent → naturally multi-label.

The image encoder runs ONCE per image; text embeddings are cached.
On CPU with 600 images and 10 attributes this takes ~3-5 minutes.
With 47 attributes the inner loop takes longer — expect ~15-20 min on CPU
for 300 images. Use --domain to restrict to a subset during development.

Prompts are loaded from YAML, not hardcoded:
    taxonomies/vlm/prompts/embellishments.yaml           (shared, all categories)
    taxonomies/vlm/prompts/model_details/<category>.yaml (one per category)

Usage
-----
    pip install transformers torch Pillow pandas pyyaml tqdm

    # Full audit for a category — model_details[shoes] + shared embellishments:
    python src/audit_zeroshot.py `
        --image-dir  "E:/fashion-data/01-RAW/shoes_emb_md_300" `
        --category   shoes `
        --out-csv    "E:/fashion-data/csv/audit_zeroshot_shoes.csv" `
        --threshold  0.5

    # Embellishments only (category not needed — they're shared):
    python src/audit_zeroshot.py `
        --image-dir  "E:/fashion-data/01-RAW/shoes_emb_md_300" `
        --domain     embellishments `
        --out-csv    "E:/fashion-data/csv/audit_zeroshot_emb.csv"

    # model_details only for one category:
    python src/audit_zeroshot.py `
        --image-dir  "E:/fashion-data/01-RAW/rtw_300" `
        --category   ready-to-wear `
        --domain     model_details `
        --out-csv    "E:/fashion-data/csv/audit_zeroshot_rtw_md.csv"

    # From preprocessed .npy files:
    python src/audit_zeroshot.py `
        --npy-dir    "E:/fashion-data/npy/embellishments/01" `
        --category   shoes `
        --out-csv    "E:/fashion-data/csv/audit_zeroshot_shoes.csv"

    # Also export Label Studio pre-annotation JSON:
    python src/audit_zeroshot.py `
        --image-dir  "..." `
        --category   shoes `
        --out-csv    "..." `
        --labelstudio

Output
------
    audit_zeroshot.csv  — one row per image, columns:
        name, <attr>_score, <attr>_pred, ...

    audit_summary.txt   — per-attribute stats printed to console + saved

    labelstudio_preannotations.json  — (if --labelstudio) ready for import
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Prompt loading — YAML-driven, category-aware
# ---------------------------------------------------------------------------
# Prompts live in taxonomies/vlm/prompts/ :
#
#   prompts/embellishments.yaml              ← SHARED across all categories
#   prompts/model_details/<category>.yaml    ← one per category
#
# For a given --category, the active attribute set is:
#   model_details[category]  +  embellishments (with any per-category override)
#
# You can also restrict to one domain with --domain {embellishments,model_details}.
#
# YAML schema
# -----------
#   embellishments.yaml:
#       attributes:
#           <attr>: {pos: "...", neg: "...", tier: A|B|C}
#
#   model_details/<category>.yaml:
#       category: <name>
#       attributes:
#           <attr>: {pos: "...", neg: "..."}
#       embellishments_overrides:        # optional
#           <attr>: {pos: "...", neg: "..."}
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS_ROOT = Path(__file__).parent.parent / "taxonomies" / "vlm" / "prompts"


def _strip_meta(attr_dict: dict) -> dict:
    """Keep only pos/neg for scoring; drop tier and any other metadata."""
    return {"pos": attr_dict["pos"], "neg": attr_dict["neg"]}


def load_embellishments(prompts_root: Path) -> dict[str, dict[str, str]]:
    """Load the shared embellishments prompt set (pos/neg only)."""
    path = Path(prompts_root) / "embellishments.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Shared embellishments file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {name: _strip_meta(a) for name, a in raw["attributes"].items()}


def load_model_details(category: str, prompts_root: Path) -> dict[str, dict[str, str]]:
    """Load a category's model_details prompts, plus any embellishment overrides."""
    path = Path(prompts_root) / "model_details" / f"{category}.yaml"
    if not path.exists():
        available = sorted(
            p.stem for p in (Path(prompts_root) / "model_details").glob("*.yaml")
        )
        raise FileNotFoundError(
            f"No model_details prompts for category '{category}' at {path}\n"
            f"Available categories: {available}"
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    md = {name: _strip_meta(a) for name, a in (raw.get("attributes") or {}).items()}
    overrides = {
        name: _strip_meta(a)
        for name, a in (raw.get("embellishments_overrides") or {}).items()
    }
    return md, overrides


def build_attributes(
    category: str | None,
    domain: str | None,
    prompts_root: Path,
) -> dict[str, dict[str, str]]:
    """
    Assemble the active attribute set for this run.

    category : which model_details set to load (required unless domain==embellishments)
    domain   : optional restriction — 'embellishments' or 'model_details'
    """
    embellishments = load_embellishments(prompts_root)

    # Embellishments-only run needs no category
    if domain == "embellishments":
        return dict(embellishments)

    if category is None:
        raise ValueError(
            "--category is required (e.g. shoes, bags, ready-to-wear) "
            "unless --domain embellishments is used."
        )

    model_details, overrides = load_model_details(category, prompts_root)

    if domain == "model_details":
        return dict(model_details)

    # Full run: model_details + embellishments, with category overrides applied.
    # Prefix keys to prevent name collisions between the two domains
    # (e.g. 'zip-closure' in model_details vs any embellishment).
    merged = dict(embellishments)
    merged.update(overrides)               # category-specific embellishment wording
    for name, a in model_details.items():
        merged[name] = a                   # model_details never collide after review
    return merged


def load_embellishment_tiers(prompts_root: Path) -> dict[str, str]:
    """
    Return {attr_name: tier} from embellishments.yaml.
    Attributes with no tier default to 'B'. Used by pipeline.py to apply
    per-tier thresholds and to skip Tier-A attributes served by the
    trained classifier instead of zero-shot.
    """
    path = Path(prompts_root) / "embellishments.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {name: a.get("tier", "B") for name, a in raw["attributes"].items()}


# ---------------------------------------------------------------------------
# Reusable scoring API — imported by pipeline.py
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL    = "patrickjohncyh/fashion-clip"
DEFAULT_THRESHOLD = 0.5

# Default per-tier thresholds. Tier C is rare → require higher confidence to
# suppress false positives; Tier B uses the standard 0.5. Tier A is normally
# served by the trained classifier, not zero-shot, so it is excluded by
# default (see exclude_tiers).
DEFAULT_TIER_THRESHOLDS = {"A": 0.5, "B": 0.5, "C": 0.7}


def load_clip(model_name: str, device: "torch.device"):
    """Load a FashionCLIP model + processor once, for reuse across images."""
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def score_images_zeroshot(
    image_paths:     list[Path],
    attributes:      dict[str, dict[str, str]],
    processor:       "CLIPProcessor",
    model:           "CLIPModel",
    device:          "torch.device",
    tiers:           dict[str, str] | None = None,
    tier_thresholds: dict[str, float] | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    Score raw images against a set of contrastive attribute prompts.

    Returns one dict per image:
        {"name": stem,
         "<attr>_score": float, "<attr>_pred": 0/1, ...}

    This is the function pipeline.py calls for its zero-shot tier. It mirrors
    the standalone run_audit() loop but takes an already-loaded model and a
    caller-supplied attribute set, and applies per-tier thresholds so Tier-C
    rare attributes use a stricter cutoff than Tier-B.

    tiers            : {attr: 'A'|'B'|'C'}; controls which threshold applies.
    tier_thresholds  : override DEFAULT_TIER_THRESHOLDS if given.
    default_threshold: used for any attribute with no tier entry.
    """
    tiers           = tiers or {}
    tier_thresholds = tier_thresholds or DEFAULT_TIER_THRESHOLDS
    attr_names      = list(attributes.keys())

    def _threshold_for(attr: str) -> float:
        tier = tiers.get(attr)
        if tier is None:
            return default_threshold
        return tier_thresholds.get(tier, default_threshold)

    rows: list[dict] = []
    for path in tqdm(image_paths, desc="Zero-shot scoring"):
        img = load_from_image_file(path)
        if img is None:
            # Emit empty row so downstream merge keeps the stem
            row = {"name": path.stem}
            for attr in attr_names:
                row[f"{attr}_score"] = ""
                row[f"{attr}_pred"]  = ""
            rows.append(row)
            continue

        row = {"name": path.stem}
        for attr in attr_names:
            score = contrastive_score_joint(
                processor, model, img,
                pos_text=attributes[attr]["pos"],
                neg_text=attributes[attr]["neg"],
                device=device,
            )
            row[f"{attr}_score"] = round(score, 4)
            row[f"{attr}_pred"]  = int(score >= _threshold_for(attr))
        rows.append(row)

    return rows


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
# FashionCLIP must be called via model(**inputs) with BOTH image and texts
# passed together. The model internally applies its learned temperature scale
# to logits_per_image. Calling get_text_features/get_image_features separately
# produces unscaled embeddings that are nearly indistinguishable — hence the
# flat 0.49-0.50 scores we saw before.
# ---------------------------------------------------------------------------

def contrastive_score_joint(
    processor: CLIPProcessor,
    model: CLIPModel,
    image: Image.Image,
    pos_text: str,
    neg_text: str,
    device: torch.device,
) -> float:
    """
    Pass image + [pos_text, neg_text] through the full model jointly.
    Returns softmax probability for the positive text (0..1).

    This uses model(**inputs).logits_per_image which applies the model's
    internal logit_scale — the correct approach for FashionCLIP and all
    standard HuggingFace CLIP checkpoints.
    """
    inputs = processor(
        text=[pos_text, neg_text],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image shape: (1, 2) — similarity to [pos, neg]
    probs = outputs.logits_per_image.softmax(dim=1)  # (1, 2)
    return float(probs[0, 0].item())   # probability of positive text


def run_diagnostics(
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    image_dir: Path | None,
) -> None:
    """
    Sanity-check: FashionCLIP should separate a fashion item from an
    unrelated object. Category-agnostic so it works for any domain.
    """
    print("Running embedding diagnostics...")

    if image_dir is not None:
        img_paths = [p for p in image_dir.rglob("*")
                     if p.suffix.lower() in IMAGE_EXTENSIONS][:1]
        if img_paths:
            pil = Image.open(img_paths[0]).convert("RGB")

            score = contrastive_score_joint(
                processor, model, pil,
                pos_text="a photo of a fashion product on a plain background",
                neg_text="a photo of a landscape or a piece of furniture",
                device=device,
            )
            print(f"  image vs [fashion / non-fashion] → fashion prob = {score:.4f}  (expect > 0.70)")

    print()


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

    run_diagnostics(processor, model, device, image_dir)

    attr_names = list(attributes.keys())

    # Load images
    images = collect_images(image_dir, npy_dir)
    if not images:
        raise ValueError("No images found. Check --image-dir or --npy-dir.")
    print(f"Running audit on {len(images)} images, {len(attr_names)} attributes...\n")

    # Score each image — joint forward pass per attribute (correct FashionCLIP usage)
    rows = []
    for stem, pil_img in tqdm(images, desc="Scoring"):
        row: dict = {"name": stem}
        for attr in attr_names:
            score = contrastive_score_joint(
                processor, model, pil_img,
                pos_text=attributes[attr]["pos"],
                neg_text=attributes[attr]["neg"],
                device=device,
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
    p.add_argument("--category", type=str, default=None,
                   help=(
                       "Fashion category — selects which model_details prompt file "
                       "to load, e.g. shoes, bags, ready-to-wear, luggage. "
                       "Required unless --domain embellishments."
                   ))
    p.add_argument("--domain", type=str, default=None,
                   choices=["embellishments", "model_details"],
                   help=(
                       "Restrict audit to one domain only. "
                       "Default: run model_details[category] + shared embellishments."
                   ))
    p.add_argument("--prompts-root", type=Path, default=DEFAULT_PROMPTS_ROOT,
                   help=f"Path to prompts directory (default: {DEFAULT_PROMPTS_ROOT})")
    p.add_argument("--model",      type=str,  default=DEFAULT_MODEL,
                   help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                   help="Score threshold for positive prediction (default: 0.5)")
    p.add_argument("--labelstudio", action="store_true",
                   help="Also export a Label Studio pre-annotation JSON")

    args = p.parse_args()

    # Assemble the active attribute set from YAML
    active_attributes = build_attributes(
        category     = args.category,
        domain       = args.domain,
        prompts_root = args.prompts_root,
    )
    scope = args.domain or f"model_details[{args.category}] + embellishments"
    print(f"Prompt scope  : {scope} ({len(active_attributes)} attributes)")

    df = run_audit(
        image_dir  = args.image_dir,
        npy_dir    = args.npy_dir,
        out_csv    = args.out_csv,
        model_name = args.model,
        threshold  = args.threshold,
        attributes = active_attributes,
    )

    summary_path = args.out_csv.parent / "audit_summary.txt"
    print_summary(df, active_attributes, args.threshold, summary_path)

    if args.labelstudio:
        ls_path = args.out_csv.parent / "labelstudio_preannotations.json"
        export_labelstudio_preannotations(
            df, active_attributes, args.threshold, ls_path, args.image_dir
        )


if __name__ == "__main__":
    main()
