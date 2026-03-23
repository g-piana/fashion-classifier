"""
build_infer_csv.py
==================
Scans a flat folder of images and writes a minimal CSV with just the
stem names — no labels. Used to drive preprocess.py for unlabelled
inference images.

Usage
-----
    python src/build_infer_csv.py \
        --src "E:/fashion-data/00-RAW/photo_or_draw_raw/infer" \
        --csv "E:/fashion-data/csv/labels_photo_or_draw_infer.csv" \
        --label-col "Class"
"""

import argparse
import csv
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--label-col", default="Class")
    args = p.parse_args()

    images = sorted(
        f for f in args.src.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    if not images:
        raise ValueError(f"No images found in {args.src}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", args.label_col])
        writer.writeheader()
        for img in images:
            # Placeholder label — preprocess only uses the stem,
            # label content is irrelevant for npy generation
            writer.writerow({"name": img.stem, args.label_col: "sandal"})

    print(f"Written {len(images)} rows → {args.csv}")


if __name__ == "__main__":
    main()