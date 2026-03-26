"""
build_custom_csv.py
===================
Scans a folder of class subfolders (images already have unique names)
and writes a CSV ready for the training pipeline.

Standard layout (flat — default):
    jackets_img/
        biker/          parka_jacket_011.jpg  ...
        blazer/         ...

Two-level layout (use --recursive):
    categories/
        heeled-shoes-women/
            pump/       img_001.jpg ...
            mule/       img_002.jpg ...
        sandals-women/
            ...

With --recursive the label is the TOP-LEVEL subfolder name (e.g.
"heeled-shoes-women"), and images are collected from all nested
subfolders beneath it.

Duplicate stems
---------------
A "duplicate stem" means the same filename (e.g. C0S6047.jpg) exists in
more than one subfolder under the same class. In --recursive mode all
those copies would produce identical CSV rows (same name, same label),
so only the first occurrence is kept. This is correct behaviour — the
.npy file is keyed by stem, so multiple copies of the same image add no
new training signal.

Use --show-duplicates to print the full paths of every duplicate so you
can decide whether to clean up the source data.

Output CSV columns:
    name   : image stem (no extension) — used to find the .npy file
    Class  : subfolder name (= label)

Usage
-----
    # Flat (original behaviour):
    python src/build_custom_csv.py `
      --src  "E:/fashion-data/01-RAW/nillab_01/categories/heeled-shoes-women" `
      --csv  "E:/fashion-data/csv/labels_heeled_shoes_sub.csv" `
      --label-col "Class"

    # Two-level — top-level folder name becomes the label:
    python src/build_custom_csv.py `
      --src  "E:/fashion-data/01-RAW/nillab_01/categories" `
      --csv  "E:/fashion-data/csv/labels_shoes_category.csv" `
      --label-col "Class" `
      --recursive

    # Show full paths of duplicates for investigation:
    python src/build_custom_csv.py `
      --src  "E:/fashion-data/01-RAW/nillab_01/categories" `
      --csv  "E:/fashion-data/csv/labels_shoes_category.csv" `
      --label-col "Class" `
      --recursive `
      --show-duplicates
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Map subfolder name -> label written to CSV.
# Add overrides here if your folder names differ from the desired label strings.
LABEL_OVERRIDES: dict[str, str] = {
    "fur_jacket": "fur jacket",   # folder name -> label with space
}


def build_csv(
    src: Path,
    csv_path: Path,
    label_col: str,
    recursive: bool,
    show_duplicates: bool,
) -> None:
    src = src.resolve()
    csv_path = csv_path.resolve()

    print(f"\nScanning : {src}")
    print(f"CSV      : {csv_path}")
    print(f"Recursive: {recursive}\n")

    rows: list[dict] = []
    class_counts: dict[str, int] = defaultdict(int)

    # stem -> first Path where it was seen (scoped per top-level subfolder)
    # Key: (label, stem) so the same stem in different categories is fine
    seen: dict[tuple[str, str], Path] = {}
    # duplicates: list of (stem, label, first_path, duplicate_path)
    duplicates: list[tuple[str, str, Path, Path]] = []

    for subfolder in sorted(src.iterdir()):

        if not subfolder.is_dir() or subfolder.stem == "infer":
            continue

        label = LABEL_OVERRIDES.get(subfolder.name, subfolder.name)

        if recursive:
            images = [
                f for f in sorted(subfolder.rglob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            ]
        else:
            images = [
                f for f in sorted(subfolder.iterdir())
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            ]

        if not images:
            print(f"  SKIP (empty): {subfolder.name}")
            continue

        # Count before dedup to show total found
        found = len(images)
        added = 0

        for img_path in images:
            stem = img_path.stem
            key  = (label, stem)

            if key in seen:
                duplicates.append((stem, label, seen[key], img_path))
                continue

            seen[key] = img_path
            rows.append({"name": stem, label_col: label})
            class_counts[label] += 1
            added += 1

        skipped = found - added
        if skipped:
            print(
                f"  [{label}]  {added} images  (from {subfolder.name}/)  "
                f"— {skipped} duplicate stem(s) skipped (same image in multiple subfolders)"
            )
        else:
            print(f"  [{label}]  {added} images  (from {subfolder.name}/)")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\nClass summary:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label:30s}: {count:>5} images")
    print(f"  {'TOTAL':30s}: {sum(class_counts.values()):>5} images")

    if duplicates:
        print(f"\nDuplicate stems: {len(duplicates)} total")
        print(
            "  (These are images whose filename appears in more than one subfolder\n"
            "   under the same class. They produce identical CSV rows and are safely\n"
            "   deduplicated — no training signal is lost.)"
        )
        if show_duplicates:
            print("\n  Duplicate details (stem | class | kept path → duplicate path):")
            for stem, label, first, dup in duplicates:
                # Show paths relative to src for readability
                try:
                    first_rel = first.relative_to(src)
                    dup_rel   = dup.relative_to(src)
                except ValueError:
                    first_rel, dup_rel = first, dup
                print(f"    {stem}  [{label}]")
                print(f"      kept : {first_rel}")
                print(f"      skip : {dup_rel}")
        else:
            print("  Run with --show-duplicates to see the full paths.")

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
    parser.add_argument("--recursive", action="store_true",
                        help=(
                            "Collect images from ALL nested subfolders under each "
                            "class folder. Label = top-level subfolder name. "
                            "Use this for two-level layouts (category/subcategory/images)."
                        ))
    parser.add_argument("--show-duplicates", action="store_true",
                        help=(
                            "Print the full path of every duplicate stem so you can "
                            "investigate whether the source data needs cleaning."
                        ))
    args = parser.parse_args()

    build_csv(
        src=Path(args.src),
        csv_path=Path(args.csv),
        label_col=args.label_col,
        recursive=args.recursive,
        show_duplicates=args.show_duplicates,
    )


if __name__ == "__main__":
    main()
