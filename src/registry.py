"""
registry.py
===========
Resolves taxonomy definitions and checkpoint paths from the
taxonomies/ directory and the weights/ directory on disk.

This module is the bridge between:
  - taxonomies/*.yaml  (what models exist and what their vocabularies are)
  - weights/*/*/manifest.json  (where trained checkpoints live)

It has NO dependency on Hydra — it reads plain YAML and JSON.
This is intentional: the pipeline orchestrator and the Lambda handler
both use this module without needing the full training config stack.

Public API
----------
    load_taxonomy(domain, taxonomies_root)
        Load and return a TaxonomyConfig for the given domain.

    resolve_checkpoint(model_name, run, weights_root, brand=None)
        Return the CheckpointInfo for a specific model+run, reading
        its manifest.json. Raises if manifest is missing.

    resolve_taxonomy_checkpoints(taxonomy, weights_root, brand=None)
        Resolve all checkpoints referenced by a taxonomy in one call.
        Returns a TaxonomyCheckpoints with stage1 + stage2_map filled.

Usage
-----
    from registry import load_taxonomy, resolve_taxonomy_checkpoints
    from pathlib import Path

    taxonomy = load_taxonomy("shoes", Path("taxonomies"))
    resolved = resolve_taxonomy_checkpoints(
        taxonomy,
        weights_root=Path("E:/fashion-data/weights"),
    )

    # stage-1
    print(resolved.stage1.classes)      # frozen class list from manifest
    print(resolved.stage1.ckpt_path)    # Path to best.ckpt
    print(resolved.stage1.norm_path)    # Path to normalization.npy

    # stage-2 for a specific category
    sub = resolved.stage2_map["heeled-shoes-women"]
    print(sub.classes)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """Raw stage definition as read from a taxonomy YAML."""
    model_name:   str
    run:          str
    npy_run:      str
    label_column: str
    csv_file:     str
    classes:      list[str]


@dataclass
class TaxonomyConfig:
    """Full taxonomy definition loaded from taxonomies/<domain>.yaml."""
    domain:      str
    image_size:  int
    backbone:    str
    stage1:      StageConfig
    stage2_map:  dict[str, StageConfig]   # category_label → StageConfig


@dataclass
class CheckpointInfo:
    """A resolved checkpoint — manifest read, paths verified."""
    model_name:   str
    run:          str
    npy_run:      str
    ckpt_path:    Path
    norm_path:    Path
    manifest:     dict
    # Pulled from manifest for convenience
    classes:      list[str]
    label_type:   str
    image_size:   int
    backbone:     str
    brand:        Optional[str]


@dataclass
class TaxonomyCheckpoints:
    """All resolved checkpoints for a taxonomy, ready for inference."""
    domain:     str
    stage1:     CheckpointInfo
    stage2_map: dict[str, CheckpointInfo] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load taxonomy
# ---------------------------------------------------------------------------

def load_taxonomy(domain: str, taxonomies_root: Path) -> TaxonomyConfig:
    """
    Load taxonomies/<domain>.yaml and return a TaxonomyConfig.

    Parameters
    ----------
    domain          : e.g. "shoes", "jackets"
    taxonomies_root : Path to the taxonomies/ directory
    """
    path = Path(taxonomies_root) / f"{domain}.yaml"
    if not path.exists():
        available = [p.stem for p in Path(taxonomies_root).glob("*.yaml")]
        raise FileNotFoundError(
            f"No taxonomy found for domain '{domain}' at {path}\n"
            f"Available domains: {available}"
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    stage1_raw = raw["stage1"]
    stage1 = StageConfig(
        model_name   = stage1_raw["model_name"],
        run          = str(stage1_raw["run"]),
        npy_run      = str(stage1_raw.get("npy_run", stage1_raw["run"])),
        label_column = stage1_raw["label_column"],
        csv_file     = stage1_raw["csv_file"],
        classes      = list(stage1_raw["classes"]),
    )

    stage2_map: dict[str, StageConfig] = {}
    for category, sub_raw in (raw.get("stage2_map") or {}).items():
        stage2_map[category] = StageConfig(
            model_name   = sub_raw["model_name"],
            run          = str(sub_raw["run"]),
            npy_run      = str(sub_raw.get("npy_run", sub_raw["run"])),
            label_column = sub_raw["label_column"],
            csv_file     = sub_raw["csv_file"],
            classes      = list(sub_raw["classes"]),
        )

    return TaxonomyConfig(
        domain     = raw["domain"],
        image_size = int(raw.get("image_size", 224)),
        backbone   = raw.get("backbone", "resnet50"),
        stage1     = stage1,
        stage2_map = stage2_map,
    )


# ---------------------------------------------------------------------------
# Resolve a single checkpoint
# ---------------------------------------------------------------------------

def resolve_checkpoint(
    model_name:   str,
    run:          str,
    npy_run:      str,
    weights_root: Path,
    brand:        Optional[str] = None,
) -> CheckpointInfo:
    """
    Locate the checkpoint directory for model_name/run (or brand variant),
    read its manifest.json, and return a CheckpointInfo.

    Brand resolution: if brand is given and weights/{model_name}/{run}_{brand}/
    exists, that directory is preferred over the base run. This keeps brand
    fine-tunes isolated without a separate registry entry.

    Raises
    ------
    FileNotFoundError  if the directory, manifest, ckpt, or norm is missing.
    """
    weights_root = Path(weights_root)

    # Brand variant path: weights/{model_name}/{run}_{brand}/
    # e.g. weights/heeled_shoes_sub/02_gucci/
    if brand:
        brand_dir = weights_root / model_name / f"{run}_{brand}"
        if brand_dir.exists():
            wts_dir = brand_dir
        else:
            # Fall back to base run silently
            wts_dir = weights_root / model_name / run
    else:
        wts_dir = weights_root / model_name / run

    # Validate manifest
    manifest_path = wts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {wts_dir}\n"
            f"Run scripts/backfill_manifests.py or retrain to generate it."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ckpt_path = wts_dir / manifest.get("checkpoint", "best.ckpt")
    norm_path = wts_dir / manifest.get("normalization", "normalization.npy")

    for p, label in [(ckpt_path, "Checkpoint"), (norm_path, "Normalization")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    return CheckpointInfo(
        model_name = model_name,
        run        = run,
        npy_run    = npy_run,
        ckpt_path  = ckpt_path,
        norm_path  = norm_path,
        manifest   = manifest,
        classes    = manifest["classes"],      # ← authoritative order from manifest
        label_type = manifest["label_type"],
        image_size = manifest["image_size"],
        backbone   = manifest["backbone"],
        brand      = manifest.get("brand"),
    )


# ---------------------------------------------------------------------------
# Resolve all checkpoints for a taxonomy
# ---------------------------------------------------------------------------

def resolve_taxonomy_checkpoints(
    taxonomy:     TaxonomyConfig,
    weights_root: Path,
    brand:        Optional[str] = None,
) -> TaxonomyCheckpoints:
    """
    Resolve all checkpoints referenced by a taxonomy in one call.
    Validates that every manifest exists before returning — fails fast
    rather than discovering a missing model mid-inference.

    Parameters
    ----------
    taxonomy     : loaded via load_taxonomy()
    weights_root : root weights directory
    brand        : optional brand name for fine-tuned variant resolution
    """
    weights_root = Path(weights_root)

    print(f"\nResolving checkpoints for domain '{taxonomy.domain}'"
          + (f"  [brand: {brand}]" if brand else ""))

    # Stage 1
    stage1_ckpt = resolve_checkpoint(
        model_name   = taxonomy.stage1.model_name,
        run          = taxonomy.stage1.run,
        npy_run      = taxonomy.stage1.npy_run,
        weights_root = weights_root,
        brand        = brand,
    )
    print(f"  stage1  : {taxonomy.stage1.model_name}/{stage1_ckpt.run}"
          f"  classes={stage1_ckpt.classes}")

    # Stage 2 — resolve each subcategory model
    stage2_map: dict[str, CheckpointInfo] = {}
    for category, sub_cfg in taxonomy.stage2_map.items():
        sub_ckpt = resolve_checkpoint(
            model_name   = sub_cfg.model_name,
            run          = sub_cfg.run,
            npy_run      = sub_cfg.npy_run,
            weights_root = weights_root,
            brand        = brand,
        )
        stage2_map[category] = sub_ckpt
        print(f"  stage2  : {sub_cfg.model_name}/{sub_ckpt.run}"
              f"  [{category}]  classes={sub_ckpt.classes}")

    return TaxonomyCheckpoints(
        domain     = taxonomy.domain,
        stage1     = stage1_ckpt,
        stage2_map = stage2_map,
    )


# ---------------------------------------------------------------------------
# CLI — quick validation tool
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Validate that all taxonomy checkpoints resolve correctly"
    )
    p.add_argument("domain",         help="Domain name, e.g. shoes")
    p.add_argument("--taxonomies",   default="taxonomies",
                   help="Path to taxonomies/ directory (default: taxonomies/)")
    p.add_argument("--weights-root", default="E:/fashion-data/weights",
                   help="Path to weights root directory")
    p.add_argument("--brand",        default=None,
                   help="Optional brand name for fine-tuned variant resolution")
    args = p.parse_args()

    try:
        taxonomy = load_taxonomy(args.domain, Path(args.taxonomies))
        resolved = resolve_taxonomy_checkpoints(
            taxonomy,
            weights_root = Path(args.weights_root),
            brand        = args.brand,
        )
        print(f"\n✓  All checkpoints resolved for domain '{args.domain}'")
        print(f"   Stage-1 classes : {resolved.stage1.classes}")
        print(f"   Stage-2 models  : {list(resolved.stage2_map.keys())}")
    except Exception as e:
        print(f"\n✗  {e}", file=sys.stderr)
        sys.exit(1)
