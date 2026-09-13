"""
scripts/backfill_manifests.py
=============================
Generates manifest.json for all existing checkpoints that were trained
before the manifest system was introduced.

Values are read from the corresponding conf/category/*.yaml files, which
are the authoritative source for existing runs. After running this script,
all existing checkpoints are first-class citizens of the new registry.

Run once from the repo root:

    python scripts/backfill_manifests.py

Dry-run (prints what would be written, writes nothing):

    python scripts/backfill_manifests.py --dry-run

Skip already-manifested checkpoints silently (default behaviour — safe
to run multiple times).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from manifest import backfill_manifest

# ---------------------------------------------------------------------------
# Registry of existing checkpoints
# ---------------------------------------------------------------------------
# Edit this table when you add new domains / runs.
# Fields:
#   wts_path        — relative to E:/fashion-data/weights/
#   domain          — config.category.name value
#   classes         — MUST match training order in conf/category/<name>.yaml
#   label_type      — "single" or "multi"
#   backbone        — "resnet50", "resnet18", or "clip_vit"
#   image_size      — 224 (default) or 336 (CLIP runs)
#   run             — folder name under weights/{domain}/
#   val_metric_name — metric monitored during training
#   val_metric_value— best val score (check MLflow or eval/ folder)
#   scope           — None for top-level, parent category for subcategory models
#   parent_run      — None unless fine-tuned from another run

CHECKPOINTS = [
    # ── Jackets ─────────────────────────────────────────────────────────
    dict(
        wts_path="jackets/02",
        domain="jackets",
        classes=["biker", "blazer", "bomber", "fur jacket", "parka"],
        label_type="single",
        backbone="resnet50",
        image_size=224,
        run="01",
        val_metric_name="val_f1",
        val_metric_value=0.966,   # fill from MLflow / eval folder if known
        scope=None,
        parent_run=None,
    ),


]

    # ── Shoes — top-level category ───────────────────────────────────────
    # dict(
    #     wts_path="shoes_category/01",
    #     domain="shoes_category",
    #     classes=[
    #         "boots-and-booties-women",
    #         "flat-shoes-women",
    #         "heeled-shoes-women",
    #         "sandals-women",
    #         "sneakers-women",
    #     ],
    #     label_type="single",
    #     backbone="resnet50",
    #     image_size=224,
    #     run="01",
    #     val_metric_name="val_f1",
    #     val_metric_value=None,
    #     scope=None,
    #     parent_run=None,
    # ),

    # ── Shoes — subcategory models ───────────────────────────────────────
    # Add one entry per subcategory classifier.
    # CRITICAL: classes must match the order in conf/category/<name>.yaml
    # exactly — this is what the manifest system exists to freeze.

    # dict(
    #     wts_path="boots_sub/01",
    #     domain="boots_sub",
    #     classes=["ankle-boot", "boot", "lace-up-boot", "over-the-knee"],
    #     label_type="single",
    #     backbone="resnet50",
    #     image_size=224,
    #     run="01",
    #     val_metric_name="val_f1",
    #     val_metric_value=None,
    #     scope="boots-and-booties-women",   # parent category this model handles
    #     parent_run=None,
    # ),

    # Add heeled_sub, flat_sub, sandals_sub, sneakers_sub entries here
    # following the same pattern once you confirm class orders from their YAMLs.
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Backfill manifest.json for existing checkpoints")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without writing anything")
    p.add_argument("--weights-root", type=Path,
                   default=Path("E:/fashion-data/weights"),
                   help="Root weights directory (default: E:/fashion-data/weights)")
    args = p.parse_args()

    print(f"Weights root : {args.weights_root}")
    print(f"Dry run      : {args.dry_run}")
    print(f"Checkpoints  : {len(CHECKPOINTS)}\n")

    written = 0
    skipped = 0
    missing = 0

    for entry in CHECKPOINTS:
        wts_path = args.weights_root / entry["wts_path"]

        if not wts_path.exists():
            print(f"  ✗  {wts_path} — directory not found, skipping")
            missing += 1
            continue

        if not (wts_path / "best.ckpt").exists():
            print(f"  ✗  {wts_path} — no best.ckpt, skipping")
            missing += 1
            continue

        manifest_path = wts_path / "manifest.json"
        if manifest_path.exists():
            print(f"  –  {wts_path} — manifest already exists, skipping")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would write manifest to {wts_path}")
            print(f"            domain={entry['domain']}  "
                  f"classes={entry['classes']}  "
                  f"backbone={entry['backbone']}")
            written += 1
            continue

        backfill_manifest(
            wts_path=wts_path,
            domain=entry["domain"],
            classes=entry["classes"],
            label_type=entry["label_type"],
            backbone=entry["backbone"],
            image_size=entry["image_size"],
            run=entry["run"],
            val_metric_name=entry.get("val_metric_name", "val_f1"),
            val_metric_value=entry.get("val_metric_value"),
            scope=entry.get("scope"),
            brand=entry.get("brand"),
            parent_run=entry.get("parent_run"),
        )
        written += 1

    print(f"\n{'='*50}")
    print(f"Written : {written}")
    print(f"Skipped : {skipped}  (manifest already existed)")
    print(f"Missing : {missing}  (no checkpoint found)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
