"""
build_custom_csv.py
===================
Scans a folder of class subfolders (images already have unique names)
and writes a CSV ready for the training pipeline.

Expected input layout:
    jackets_img/
        biker/          parka_jacket_011.jpg  ...
        blazer/         ...
        bomber/         ...
        fur_jacket/     ...
        parka/          ...

Output CSV columns:
    name   : image stem (no extension) — used to find the .npy file
    Class  : subfolder name (= label)

Usage
-----
    python src/build_custom_csv.py \
        --src  "E:/fashion-data/01-RAW/jackets_img" \
        --csv  "E:/fashion-data/csv/labels_jackets_custom.csv" \
        --label-col "Class"
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Map subfolder name -> label written to CSV.
# If a subfolder name already matches the desired label, no entry needed.
# Add overrides here if your folder names differ from the label strings.
LABEL_OVERRIDES: dict[str, str] = {
    "fur_jacket": "fur jacket",   # folder name -> label with space
}


def build_csv(src: Path, csv_path: Path, label_col: str) -> None:
    src = src.resolve()
    csv_path = csv_path.resolve()

    print(f"\nScanning : {src}")
    print(f"CSV      : {csv_path}\n")

    rows: list[dict] = []
    class_counts: dict[str, int] = defaultdict(int)
    stems_seen: set[str] = set()

    for subfolder in sorted(src.iterdir()):
        if not subfolder.is_dir():
            continue

        label = LABEL_OVERRIDES.get(subfolder.name, subfolder.name)
        images = [
            f for f in sorted(subfolder.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]

        if not images:
            print(f"  SKIP (empty): {subfolder.name}")
            continue

        print(f"  [{label}]  {len(images)} images  (from {subfolder.name}/)")

        for img_path in images:
            stem = img_path.stem
            if stem in stems_seen:
                print(f"    WARNING: duplicate stem '{stem}' — skipping")
                continue
            stems_seen.add(stem)
            rows.append({"name": stem, label_col: label})
            class_counts[label] += 1

    print(f"\nClass summary:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label:15s}: {count:>5} images")
    print(f"  {'TOTAL':15s}: {sum(class_counts.values()):>5} images")

    if not rows:
        print("\nERROR: No images found. Check --src path.")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", label_col])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written → {csv_path}  ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training CSV from pre-organized class subfolders"
    )
    parser.add_argument("--src", required=True,
                        help="Root folder containing one subfolder per class")
    parser.add_argument("--csv", required=True,
                        help="Output CSV path")
    parser.add_argument("--label-col", default="Class",
                        help="Column name for the label (default: Class)")
    args = parser.parse_args()

    build_csv(
        src=Path(args.src),
        csv_path=Path(args.csv),
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()