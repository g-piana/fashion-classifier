"""
pipeline.py
===========
Single entry point for end-to-end image classification.

Given a domain (taxonomy) and a directory of raw images, this script:
  1. Preprocesses raw images → .npy cache  (skips existing)
  2. Runs stage-1 category classification
  3. Routes each image to its stage-2 subcategory classifier
  4. Writes a structured output CSV

It reads everything from:
  - taxonomies/<domain>.yaml   (vocabulary, model names, runs)
  - weights/*/manifest.json    (class order, backbone, image size)

No Hydra dependency at runtime — this is the entry point for the
Django management command and the Lambda handler.

Usage
-----
    # Full run — preprocess + inference
    python src/pipeline.py `
        --domain    shoes `
        --image-dir "E:/fashion-data/01-RAW/nillab_01/photo" `
        --out-csv   "E:/fashion-data/csv/pipeline_shoes.csv"

    # Skip preprocessing (npy already exists)
    python src/pipeline.py `
        --domain        shoes `
        --image-dir     "E:/fashion-data/01-RAW/nillab_01/photo" `
        --npy-dir       "E:/fashion-data/npy/shoes_category/01" `
        --out-csv       "E:/fashion-data/csv/pipeline_shoes.csv" `
        --skip-preprocess

    # Single known category — skip stage-1, run stage-2 directly
    python src/pipeline.py `
        --domain        shoes `
        --image-dir     "E:/fashion-data/01-RAW/nillab_01/photo" `
        --npy-dir       "E:/fashion-data/npy/shoes_category/01" `
        --out-csv       "E:/fashion-data/csv/pipeline_shoes.csv" `
        --known-category heeled-shoes-women `
        --skip-preprocess

    # Brand fine-tuned checkpoints
    python src/pipeline.py `
        --domain shoes `
        --image-dir "E:/fashion-data/01-RAW/gucci_ss25" `
        --out-csv   "E:/fashion-data/csv/pipeline_shoes_gucci.csv" `
        --brand     gucci

    # Dry run — show what would happen without running inference
    python src/pipeline.py --domain shoes --image-dir "..." --out-csv "..." --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

# Make src/ importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from manifest import read_manifest
from registry import (
    CheckpointInfo,
    TaxonomyCheckpoints,
    load_taxonomy,
    resolve_taxonomy_checkpoints,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}
DEFAULT_TAXONOMIES_ROOT = Path(__file__).parent.parent / "taxonomies"
DEFAULT_WEIGHTS_ROOT    = Path("E:/fashion-data/weights")
DEFAULT_NPY_ROOT        = Path("E:/fashion-data/npy")


# ---------------------------------------------------------------------------
# Image preprocessing — mirrors preprocess.py exactly
# ---------------------------------------------------------------------------

def _pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w == h:
        return img
    size   = max(w, h)
    result = np.full((size, size, 3), 255, dtype=img.dtype)
    result[(size - h) // 2:(size - h) // 2 + h,
           (size - w) // 2:(size - w) // 2 + w] = img
    return result


def preprocess_images(
    image_dir:  Path,
    npy_dir:    Path,
    image_size: int = 224,
) -> list[str]:
    """
    Preprocess raw images from image_dir into npy_dir.
    Skips images already on disk (safe to call repeatedly).

    Returns list of stems successfully on disk after the run.
    """
    npy_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in image_dir.rglob("*") if p.suffix in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    print(f"\nPreprocessing {len(images)} images → {npy_dir}")
    new_count = skip_count = error_count = 0

    for img_path in images:
        out_path = npy_dir / f"{img_path.stem}.npy"
        if out_path.exists():
            skip_count += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: cannot read {img_path.name}")
            error_count += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _pad_to_square(img)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        np.save(out_path, img.astype(np.uint8))
        new_count += 1

    print(f"  New: {new_count}  Skipped: {skip_count}  Errors: {error_count}")

    input_stems = {p.stem for p in image_dir.rglob("*") if p.suffix in IMAGE_EXTENSIONS}
    stems = sorted(s for s in input_stems if (npy_dir / f"{s}.npy").exists())
    print(f"  Stems from input dir : {len(input_stems)}")
    print(f"  Available in npy_dir : {len(stems)}")
    return stems


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _NpyDataset(Dataset):
    def __init__(self, stems: list[str], npy_root: Path, normalization: np.ndarray):
        self.stems    = stems
        self.npy_root = Path(npy_root)
        self.mean     = normalization[0].astype(np.float32)
        self.std      = normalization[1].astype(np.float32)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img  = np.load(self.npy_root / f"{stem}.npy").astype(np.float32)
        img  = (img - self.mean) / (self.std + 1e-8)
        return stem, torch.from_numpy(img).permute(2, 0, 1)


def _collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


# ---------------------------------------------------------------------------
# Model loading — plain PyTorch, no Lightning, no Hydra
# ---------------------------------------------------------------------------

def _build_model(num_classes: int) -> nn.Module:
    model    = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_checkpoint(ckpt_info: CheckpointInfo, device: torch.device) -> nn.Module:
    """
    Load a checkpoint using manifest metadata.
    Strips the Lightning 'model.' prefix from state_dict keys.
    """
    model = _build_model(num_classes=len(ckpt_info.classes))

    ckpt  = torch.load(str(ckpt_info.ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"]

    cleaned = {
        k[len("model."):]: v
        for k, v in state.items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys loading {ckpt_info.ckpt_path}: {missing}")

    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inference(
    stems:      list[str],
    npy_dir:    Path,
    ckpt_info:  CheckpointInfo,
    device:     torch.device,
    batch_size: int = 64,
) -> dict[str, tuple[str, float]]:
    """
    Run single-label inference over stems.
    Reads class list from manifest (ckpt_info.classes) — not from config.

    Returns dict: stem → (predicted_class, confidence)
    """
    normalization = np.load(str(ckpt_info.norm_path))
    model         = _load_checkpoint(ckpt_info, device)

    dataset = _NpyDataset(stems, npy_dir, normalization)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = _collate,
    )

    results: dict[str, tuple[str, float]] = {}

    with torch.no_grad():
        for batch_stems, images in loader:
            logits      = model(images.to(device))
            probs       = torch.softmax(logits, dim=1)
            pred_idxs   = probs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()
            for stem, idx, conf in zip(batch_stems, pred_idxs, confidences):
                results[stem] = (ckpt_info.classes[idx], round(float(conf), 4))

    return results


# ---------------------------------------------------------------------------
# Cascade inference — taxonomy-driven
# ---------------------------------------------------------------------------

def run_cascade(
    stems:        list[str],
    npy_dir:      Path,
    resolved:     TaxonomyCheckpoints,
    device:       torch.device,
    batch_size:   int = 64,
    known_category: str | None = None,
) -> pd.DataFrame:
    """
    Run the full taxonomy cascade over stems.

    If known_category is set, stage-1 is skipped entirely and all images
    are routed directly to that category's stage-2 model. This is the
    Django admin upload path: user specifies category, we run subcategory only.

    Returns DataFrame with columns:
        name, category, category_conf, subcategory, subcategory_conf
    """
    # ── Stage 1 ────────────────────────────────────────────────────────
    if known_category:
        print(f"\nStage-1 skipped — known category: {known_category}")
        s1_results = {stem: (known_category, 1.0) for stem in stems}
    else:
        print(f"\nStage 1 — category classifier"
              f"  ({resolved.stage1.model_name}/{resolved.stage1.run})")
        print(f"  Classes : {resolved.stage1.classes}")
        s1_results = run_inference(
            stems      = stems,
            npy_dir    = npy_dir,
            ckpt_info  = resolved.stage1,
            device     = device,
            batch_size = batch_size,
        )

    # Group by predicted category
    by_category: dict[str, list[str]] = defaultdict(list)
    for stem, (cat, _) in s1_results.items():
        by_category[cat].append(stem)

    if not known_category:
        print("\n  Stage-1 distribution:")
        for cat in sorted(by_category):
            print(f"    {cat:<35} {len(by_category[cat]):>5} images")

    # ── Stage 2 ────────────────────────────────────────────────────────
    s2_results: dict[str, tuple[str, float]] = {}

    if not resolved.stage2_map:
        # Single-stage domain (e.g. jackets) — no subcategory models
        print("\n  No stage-2 models configured for this domain.")
        for stem in stems:
            s2_results[stem] = ("", 0.0)
    else:
        for category, group_stems in sorted(by_category.items()):
            if category not in resolved.stage2_map:
                print(f"\n  WARN: no stage-2 model for '{category}' — marking unknown")
                for stem in group_stems:
                    s2_results[stem] = ("unknown", 0.0)
                continue

            sub_ckpt = resolved.stage2_map[category]
            print(f"\n  Stage 2 [{category}]"
                  f"  model={sub_ckpt.model_name}/{sub_ckpt.run}"
                  f"  n={len(group_stems)}")
            print(f"    Classes : {sub_ckpt.classes}")

            available = [s for s in group_stems if (npy_dir / f"{s}.npy").exists()]
            if len(available) < len(group_stems):
                print(f"    WARN: {len(group_stems) - len(available)} stems missing .npy")

            if available:
                preds = run_inference(
                    stems      = available,
                    npy_dir    = npy_dir,
                    ckpt_info  = sub_ckpt,
                    device     = device,
                    batch_size = batch_size,
                )
                s2_results.update(preds)

            for stem in group_stems:
                if stem not in s2_results:
                    s2_results[stem] = ("unknown", 0.0)

    # ── Assemble output ────────────────────────────────────────────────
    rows = []
    for stem in stems:
        cat,    cat_conf = s1_results.get(stem, ("unknown", 0.0))
        subcat, sub_conf = s2_results.get(stem, ("",        0.0))
        rows.append({
            "name":             stem,
            "category":         cat,
            "category_conf":    cat_conf,
            "subcategory":      subcat,
            "subcategory_conf": sub_conf,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fashion classifier pipeline — preprocess + cascade inference"
    )
    p.add_argument("--domain",      required=True,
                   help="Taxonomy domain, e.g. shoes, jackets")
    p.add_argument("--image-dir",   type=Path, required=True,
                   help="Directory of raw images to classify")
    p.add_argument("--out-csv",     type=Path, required=True,
                   help="Output predictions CSV path")
    p.add_argument("--npy-dir",     type=Path, default=None,
                   help="Explicit npy directory (default: derived from taxonomy npy_run)")
    p.add_argument("--brand",       default=None,
                   help="Brand name for fine-tuned checkpoint resolution")
    p.add_argument("--known-category", default=None,
                   help="Skip stage-1 and route all images to this category directly")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing — npy files must already exist in --npy-dir")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--dry-run",     action="store_true",
                   help="Resolve config and validate paths without running inference")
    p.add_argument("--taxonomies-root", type=Path, default=DEFAULT_TAXONOMIES_ROOT)
    p.add_argument("--weights-root",    type=Path, default=DEFAULT_WEIGHTS_ROOT)
    p.add_argument("--npy-root",        type=Path, default=DEFAULT_NPY_ROOT)
    args = p.parse_args()

    t_start = time.perf_counter()

    # ── 1. Load taxonomy and resolve checkpoints ────────────────────────
    print(f"\n{'='*60}")
    print(f"Pipeline — domain: {args.domain}"
          + (f"  brand: {args.brand}" if args.brand else ""))
    print(f"{'='*60}")

    taxonomy = load_taxonomy(args.domain, args.taxonomies_root)
    resolved = resolve_taxonomy_checkpoints(
        taxonomy,
        weights_root = args.weights_root,
        brand        = args.brand,
    )

    # ── 2. Resolve npy directory ────────────────────────────────────────
    npy_dir = args.npy_dir or (
        args.npy_root / taxonomy.stage1.model_name / "infer" / args.image_dir.name
    )

    print(f"\nImage dir  : {args.image_dir}")
    print(f"NPY dir    : {npy_dir}")
    print(f"Output CSV : {args.out_csv}")

    if args.dry_run:
        print("\n[DRY RUN] All paths resolved successfully — skipping inference.")
        print(f"  Stage-1 classes : {resolved.stage1.classes}")
        for cat, sub in resolved.stage2_map.items():
            print(f"  Stage-2 [{cat}] : {sub.classes}")
        return

    # ── 3. Preprocess ───────────────────────────────────────────────────
    if args.skip_preprocess:
        if not npy_dir.exists():
            raise FileNotFoundError(
                f"--skip-preprocess set but npy directory not found: {npy_dir}"
            )
        input_stems = {p.stem for p in args.image_dir.rglob("*")
                    if p.suffix in IMAGE_EXTENSIONS}
        stems = sorted(s for s in input_stems if (npy_dir / f"{s}.npy").exists())
        missing = input_stems - set(stems)
        if missing:
            print(f"  WARN: {len(missing)} input images have no .npy — "
                f"remove --skip-preprocess to build them")
        print(f"Preprocessing skipped — {len(stems)} stems matched from input dir")
        
    else:
        stems = preprocess_images(
            image_dir  = args.image_dir,
            npy_dir    = npy_dir,
            image_size = taxonomy.image_size,
        )

    if not stems:
        raise ValueError(f"No .npy files available in {npy_dir}")

    # ── 4. Cascade inference ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    df = run_cascade(
        stems           = stems,
        npy_dir         = npy_dir,
        resolved        = resolved,
        device          = device,
        batch_size      = args.batch_size,
        known_category  = args.known_category,
    )

    # ── 5. Write output ─────────────────────────────────────────────────
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Done — {len(df)} images in {elapsed:.1f}s")
    print(f"Output : {args.out_csv}")
    if "category" in df.columns and df["category"].nunique() > 0:
        print(f"\nCategory distribution:")
        print(df["category"].value_counts().to_string())
    if "subcategory" in df.columns and df["subcategory"].ne("").any():
        print(f"\nSubcategory distribution (top 15):")
        print(df["subcategory"].value_counts().head(15).to_string())
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
