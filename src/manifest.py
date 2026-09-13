"""
manifest.py
===========
Writes and reads manifest.json files alongside trained checkpoints.

A manifest makes each checkpoint self-describing — it carries the class
list (in training order), backbone, image size, label type, and val
metrics so that inference code never needs to read Hydra configs.

This eliminates the class-order bug: the manifest is the single source
of truth for the class list, written once at train time and read at
every inference.

Public API
----------
    write_manifest(wts_path, config, trainer, checkpoint_cb)
        Called at the end of train.py — writes manifest.json.

    read_manifest(wts_path)
        Called by inference code — returns manifest as a dict.

    backfill_manifest(wts_path, category_cfg, model_backbone, image_size,
                      val_metric_name, val_metric_value)
        Called by scripts/backfill_manifests.py for existing checkpoints.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_manifest(
    wts_path: Path,
    config: DictConfig,
    trainer,           # lightning.Trainer — for best metric value
    checkpoint_cb,     # lightning ModelCheckpoint callback
) -> dict:
    """
    Build and write manifest.json into wts_path.
    Called immediately after trainer.fit() in train.py.

    Returns the manifest dict (useful for printing / testing).
    """
    best_score = checkpoint_cb.best_model_score
    val_metric = config.training.checkpoint_metric   # e.g. "val_f1" or "val_loss"

    # Resolve parent_run from finetune_from (may be None)
    parent_run = config.training.get("finetune_from", None)

    manifest: dict[str, Any] = {
        # ── Identity ───────────────────────────────────────────────────────
        "domain":    config.category.name,   # e.g. "jackets", "shoes_category"
        "task":      config.category.name,   # same for now; B will split these
        "scope":     None,                   # set for subcategory models (e.g. "heeled-shoes")
        "run":       config.data.run,        # e.g. "01"
        "brand":     None,                   # set for brand fine-tuned models

        # ── Architecture ───────────────────────────────────────────────────
        "backbone":   config.model.backbone,
        "image_size": config.data.get("image_size", 224),
        "label_type": config.category.label_type,   # "single" | "multi"

        # ── Label vocabulary (THE source of truth for class order) ─────────
        "classes":     list(config.category.classes),
        "num_classes": len(config.category.classes),

        # ── Training provenance ────────────────────────────────────────────
        "parent_run": parent_run,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),

        # ── Validation metrics ────────────────────────────────────────────
        "val_metric_name":  val_metric,
        "val_metric_value": round(float(best_score), 4) if best_score is not None else None,

        # ── File pointers (relative to this directory) ────────────────────
        "checkpoint":    "best.ckpt",
        "normalization": "normalization.npy",
    }

    out_path = Path(wts_path) / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nManifest written → {out_path}")
    print(f"  domain      : {manifest['domain']}")
    print(f"  backbone    : {manifest['backbone']}")
    print(f"  label_type  : {manifest['label_type']}")
    print(f"  classes     : {manifest['classes']}")
    print(f"  {val_metric}: {manifest['val_metric_value']}")

    return manifest


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_manifest(wts_path: Path) -> dict:
    """
    Load and return manifest.json from wts_path.
    Raises FileNotFoundError with a clear message if missing.
    """
    path = Path(wts_path) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in {wts_path}\n"
            f"Run scripts/backfill_manifests.py for existing checkpoints, "
            f"or retrain to generate it automatically."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Backfill (for checkpoints trained before this change)
# ---------------------------------------------------------------------------

def backfill_manifest(
    wts_path: Path,
    *,
    domain: str,
    classes: list[str],
    label_type: str,
    backbone: str,
    image_size: int,
    run: str,
    val_metric_name: str  = "val_f1",
    val_metric_value: float | None = None,
    scope: str | None = None,
    brand: str | None = None,
    parent_run: str | None = None,
) -> dict:
    """
    Write a manifest for an existing checkpoint that predates this system.
    All values must be supplied explicitly — there is no config to read.

    Returns the manifest dict.
    """
    wts_path = Path(wts_path)
    out_path = wts_path / "manifest.json"

    if out_path.exists():
        print(f"  Skipping {wts_path} — manifest already exists")
        return json.loads(out_path.read_text())

    if not (wts_path / "best.ckpt").exists():
        print(f"  Skipping {wts_path} — no best.ckpt found")
        return {}

    manifest: dict[str, Any] = {
        "domain":    domain,
        "task":      domain,
        "scope":     scope,
        "run":       run,
        "brand":     brand,
        "backbone":  backbone,
        "image_size": image_size,
        "label_type": label_type,
        "classes":    list(classes),
        "num_classes": len(classes),
        "parent_run": parent_run,
        "trained_at": None,   # unknown for backfilled checkpoints
        "val_metric_name":  val_metric_name,
        "val_metric_value": round(float(val_metric_value), 4) if val_metric_value is not None else None,
        "checkpoint":    "best.ckpt",
        "normalization": "normalization.npy",
        "backfilled":    True,   # flag so we know this wasn't written by train.py
    }

    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  ✓  {wts_path} → manifest.json  (classes: {classes})")
    return manifest
