"""
build_infer_csv.py
==================
Scans a folder of images and writes a minimal CSV with stem names only —
no real labels. Used to drive preprocess.py for unlabelled inference images.

The placeholder label written to the CSV is irrelevant for preprocessing;
it is only there because preprocess.py expects the column to exist.

Supports flat and recursive (nested subfolders) layouts via --recursive.

Usage
-----
    # Flat folder of images
    python src/build_infer_csv.py `
        --src "E:/fashion-data/01-RAW/nillab_01/infer" `
        --csv "E:/fashion-data/csv/labels_shoes_infer.csv"

    # Recursive — images nested inside subfolders (any depth)
    python src/build_infer_csv.py `
        --src "E:/fashion-data/01-RAW/nillab_01/infer" `
        --csv "E:/fashion-data/csv/labels_shoes_infer.csv" `
        --recursive

    # Override label column name or placeholder value
    python src/build_infer_csv.py `
        --src "E:/fashion-data/01-RAW/nillab_01/infer" `
        --csv "E:/fashion-data/csv/labels_shoes_infer.csv" `
        --label-col "Class" `
        --placeholder "unknown"
"""

import argparse
import csv
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    p = argparse.ArgumentParser(
        description="Build a stub inference CSV (no real labels) from an image folder."
    )
    p.add_argument("--src",         type=Path, required=True,
                   help="Folder containing images to run inference on.")
    p.add_argument("--csv",         type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--label-col",   default="Class",
                   help="Label column name (default: Class). Must match category config.")
    p.add_argument("--placeholder", default="unknown",
                   help="Placeholder label value (default: unknown). Value is ignored by preprocess.py.")
    p.add_argument("--recursive",   action="store_true",
                   help="Scan subfolders recursively (any depth). Use when images are nested.")
    args = p.parse_args()

    src = args.src.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")

    # Collect images
    if args.recursive:
        images = sorted(
            f for f in src.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
    else:
        images = sorted(
            f for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )

    if not images:
        raise ValueError(
            f"No images found in {src}"
            + (" (searched recursively)" if args.recursive else
               " — try adding --recursive if images are in subfolders")
        )

    # Warn on duplicate stems (can happen with recursive scan across subfolders)
    stems_seen: set[str] = set()
    duplicates = 0
    rows = []
    for img in images:
        stem = img.stem
        if stem in stems_seen:
            print(f"  WARNING: duplicate stem '{stem}' — skipping {img}")
            duplicates += 1
            continue
        stems_seen.add(stem)
        rows.append({"name": stem, args.label_col: args.placeholder})

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", args.label_col])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows → {args.csv}")
    if duplicates:
        print(f"Skipped {duplicates} duplicate stems.")


if __name__ == "__main__":
    main()
