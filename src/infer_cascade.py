"""
infer_cascade.py
================
Two-stage cascading inference for shoes (or any domain with a
category → subcategory hierarchy).

Stage 1 : classify every image into a top-level category
           (e.g. heeled-shoes-women, sandals-women, …)
Stage 2 : for each category group, run the matching subcategory
           classifier (e.g. pump, mule, slingback, …)

Output CSV columns
------------------
    name               : image stem
    category           : stage-1 prediction
    category_conf      : stage-1 softmax confidence
    subcategory        : stage-2 prediction
    subcategory_conf   : stage-2 softmax confidence

Config-driven — all paths resolved from the standard filesystem
convention; no hard-coded directories.

Usage
-----
    # Minimal — uses all defaults from conf/cascade/shoes.yaml
    python src/infer_cascade.py cascade=shoes

    # Override npy dir and output CSV
    python src/infer_cascade.py `
        cascade=shoes `
        cascade_infer.npy_dir="E:/fashion-data/npy/shoes_category/01" `
        cascade_infer.out_csv="E:/fashion-data/csv/predictions_shoes_cascade.csv"

    # Limit batch size (useful on small GPUs)
    python src/infer_cascade.py cascade=shoes cascade_infer.batch_size=32
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from collections import defaultdict

from model import FashionClassifier


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NpyDataset(Dataset):
    """Loads preprocessed .npy files, normalises, returns (stem, tensor)."""

    def __init__(self, stems: list[str], npy_root: Path, normalization: np.ndarray):
        self.stems    = stems
        self.npy_root = Path(npy_root)
        self.mean     = normalization[0]   # (3,)
        self.std      = normalization[1]   # (3,)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img  = np.load(self.npy_root / f"{stem}.npy").astype(np.float32)
        img  = (img - self.mean) / (self.std + 1e-8)
        img  = torch.from_numpy(img).permute(2, 0, 1)   # HWC → CHW
        return stem, img


def _collate(batch):
    stems  = [b[0] for b in batch]
    images = torch.stack([b[1] for b in batch])
    return stems, images


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path, device: torch.device) -> FashionClassifier:
    """Load a FashionClassifier checkpoint onto device."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = FashionClassifier.load_from_checkpoint(str(ckpt_path))
    model.eval()
    return model.to(device)


def run_inference(
    stems: list[str],
    npy_dir: Path,
    norm_path: Path,
    ckpt_path: Path,
    classes: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, tuple[str, float]]:
    """
    Run single-label inference over `stems`.

    Returns
    -------
    dict  stem -> (predicted_class, confidence)
    """
    normalization = np.load(norm_path)
    model         = _load_model(ckpt_path, device)

    dataset = NpyDataset(stems, npy_dir, normalization)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,          # 0 avoids multiprocessing issues on Windows
        shuffle=False,
        collate_fn=_collate,
    )

    results: dict[str, tuple[str, float]] = {}

    with torch.no_grad():
        for batch_stems, images in loader:
            images      = images.to(device)
            logits      = model(images)
            probs       = torch.softmax(logits, dim=1)
            pred_idxs   = probs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()

            for stem, pred_idx, conf in zip(batch_stems, pred_idxs, confidences):
                results[stem] = (classes[pred_idx], round(float(conf), 4))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:

    # ------------------------------------------------------------------ #
    # 1. Resolve cascade config and infer overrides
    # ------------------------------------------------------------------ #
    if "cascade" not in config:
        raise ValueError(
            "No cascade config found.\n"
            "Run with:  python src/infer_cascade.py cascade=shoes"
        )

    cascade_cfg    = config.cascade
    infer_cfg      = config.get("cascade_infer", {})
    root           = Path(config.filesystem.root)
    batch_size     = infer_cfg.get("batch_size", 64)
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1         = cascade_cfg.stage1
    stage2_map     = cascade_cfg.stage2_map

    # Stage-1 paths
    # npy comes from infer_run (e.g. "infer"), weights/norm from run (e.g. "01")
    npy_run     = stage1.get("infer_run", stage1.run)
    s1_npy_dir  = Path(infer_cfg.get("npy_dir", "") or
                       root / config.filesystem.npy / stage1.name / npy_run)
    s1_wts_dir  = root / config.filesystem.weights / stage1.name / stage1.run
    s1_ckpt     = s1_wts_dir / "best.ckpt"
    s1_norm     = s1_wts_dir / "normalization.npy"
    s1_classes  = list(stage1.classes)

    # Output CSV
    out_csv = Path(infer_cfg.get("out_csv", "") or
                   root / config.filesystem.csv / "predictions_shoes_cascade.csv")

    print(f"\n{'='*60}")
    print(f"Cascade inference — shoes domain")
    print(f"Device       : {device}")
    print(f"Stage-1 model: {stage1.name}  (run {stage1.run})")
    print(f"Stage-1 npy  : {s1_npy_dir}")
    print(f"Output CSV   : {out_csv}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 2. Collect stems from the stage-1 npy directory
    # ------------------------------------------------------------------ #
    if not s1_npy_dir.exists():
        raise FileNotFoundError(
            f"Stage-1 npy directory not found: {s1_npy_dir}\n"
            "Run preprocess.py for shoes_category first."
        )

    stems = sorted(p.stem for p in s1_npy_dir.glob("*.npy"))
    if not stems:
        raise ValueError(f"No .npy files found in {s1_npy_dir}")
    print(f"Stage 1 — found {len(stems)} images\n")

    # ------------------------------------------------------------------ #
    # 3. Stage 1: category classification
    # ------------------------------------------------------------------ #
    print(f"Running stage-1 classifier ({stage1.name}) …")
    s1_results = run_inference(
        stems      = stems,
        npy_dir    = s1_npy_dir,
        norm_path  = s1_norm,
        ckpt_path  = s1_ckpt,
        classes    = s1_classes,
        batch_size = batch_size,
        device     = device,
    )

    # Group stems by predicted category
    by_category: dict[str, list[str]] = defaultdict(list)
    for stem, (cat, _conf) in s1_results.items():
        by_category[cat].append(stem)

    print("\nStage-1 distribution:")
    for cat in sorted(by_category):
        print(f"  {cat:<35} {len(by_category[cat]):>5} images")

    # ------------------------------------------------------------------ #
    # 4. Stage 2: subcategory classification (per category group)
    # ------------------------------------------------------------------ #
    s2_results: dict[str, tuple[str, float]] = {}

    for category, group_stems in sorted(by_category.items()):

        if category not in stage2_map:
            print(f"\n  WARNING: no stage-2 model configured for '{category}' — skipping subcategory")
            for stem in group_stems:
                s2_results[stem] = ("unknown", 0.0)
            continue

        sub_cfg    = stage2_map[category]
        sub_name   = sub_cfg.name
        sub_run    = sub_cfg.run
        sub_classes = list(sub_cfg.classes)

        sub_wts_dir = root / config.filesystem.weights / sub_name / sub_run
        sub_ckpt    = sub_wts_dir / "best.ckpt"
        sub_norm    = sub_wts_dir / "normalization.npy"

        # Stage-2 npy: these images were preprocessed under the category model's
        # npy dir (same images, same stems — npy files are shared via stage-1 dir)
        sub_npy_dir = s1_npy_dir   # stems already exist here from stage-1 preprocess

        print(f"\n  Stage 2 [{category}]  →  model: {sub_name}  ({len(group_stems)} images)")

        # Verify only the stems that actually have a .npy in the shared dir
        available = [s for s in group_stems if (sub_npy_dir / f"{s}.npy").exists()]
        missing   = len(group_stems) - len(available)
        if missing:
            print(f"    WARNING: {missing} stems have no .npy — will be marked unknown")

        if available:
            group_preds = run_inference(
                stems      = available,
                npy_dir    = sub_npy_dir,
                norm_path  = sub_norm,
                ckpt_path  = sub_ckpt,
                classes    = sub_classes,
                batch_size = batch_size,
                device     = device,
            )
            s2_results.update(group_preds)

        for stem in group_stems:
            if stem not in s2_results:
                s2_results[stem] = ("unknown", 0.0)

    # ------------------------------------------------------------------ #
    # 5. Assemble and save output CSV
    # ------------------------------------------------------------------ #
    rows = []
    for stem in stems:
        cat,     cat_conf  = s1_results.get(stem,  ("unknown", 0.0))
        subcat,  sub_conf  = s2_results.get(stem,  ("unknown", 0.0))
        rows.append({
            "name":             stem,
            "category":         cat,
            "category_conf":    cat_conf,
            "subcategory":      subcat,
            "subcategory_conf": sub_conf,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Done.  {len(df)} rows written → {out_csv}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nSubcategory distribution (top 20):")
    print(df["subcategory"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
