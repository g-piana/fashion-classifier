"""
find_duplicate_stems.py
=======================
Scans a two-level folder structure (category / subcategory / images)
and finds image stems that appear in MORE THAN ONE subcategory under
the same category.

Since subcategories are mutually exclusive, any stem appearing in
multiple subcategory folders is a data error — the same product image
has been placed in two competing labels.

Output CSV columns
------------------
    stem          : image filename without extension
    category      : top-level folder name
    occurrences   : how many subcategory folders contain this stem
    subcat_1 … N  : the subcategory folder names where it was found
    path_1 … N    : the full paths of each copy

Usage
-----
    python src/find_duplicate_stems.py `
        --src "E:/fashion-data/01-RAW/nillab_01/categories" `
        --csv "E:/fashion-data/csv/duplicate_stems.csv"

    # Limit to one category
    python src/find_duplicate_stems.py `
        --src "E:/fashion-data/01-RAW/nillab_01/categories" `
        --csv "E:/fashion-data/csv/duplicate_stems.csv" `
        --category flat-shoes-women
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_duplicates(
    src: Path,
    category_filter: str | None,
) -> list[dict]:
    """
    Returns a list of rows, one per duplicate stem.
    Each row has: stem, category, occurrences, subcat_1..N, path_1..N
    """
    src = src.resolve()
    all_rows = []

    categories = sorted(
        d for d in src.iterdir()
        if d.is_dir() and d.name != "infer"
    )

    if category_filter:
        categories = [c for c in categories if c.name == category_filter]
        if not categories:
            raise ValueError(f"Category '{category_filter}' not found in {src}")

    for cat_dir in categories:
        cat_name = cat_dir.name

        # Build: stem -> list of (subcategory_name, full_path)
        stem_map: dict[str, list[tuple[str, Path]]] = defaultdict(list)

        subcategories = sorted(d for d in cat_dir.iterdir() if d.is_dir())
        for subcat_dir in subcategories:
            for img_path in sorted(subcat_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTS:
                    stem_map[img_path.stem].append((subcat_dir.name, img_path))

        # Keep only stems that appear more than once
        duplicates = {
            stem: occurrences
            for stem, occurrences in stem_map.items()
            if len(occurrences) > 1
        }

        if duplicates:
            print(f"  [{cat_name}]  {len(duplicates)} duplicate stem(s)")
        else:
            print(f"  [{cat_name}]  no duplicates")

        for stem, occurrences in sorted(duplicates.items()):
            row: dict = {
                "stem":        stem,
                "category":    cat_name,
                "occurrences": len(occurrences),
            }
            for i, (subcat, path) in enumerate(occurrences, start=1):
                row[f"subcat_{i}"] = subcat
                row[f"path_{i}"]   = str(path)
            all_rows.append(row)

    return all_rows


def write_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        print("\nNo duplicates found — nothing to write.")
        return

    # Determine max occurrences to build dynamic column headers
    max_occ = max(r["occurrences"] for r in rows)
    fieldnames = ["stem", "category", "occurrences"]
    for i in range(1, max_occ + 1):
        fieldnames += [f"subcat_{i}", f"path_{i}"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDuplicate report written → {csv_path}  ({len(rows)} rows)")
    print("Next step: run quarantine_duplicates.py to move them for manual review.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Find image stems present in more than one subcategory folder"
    )
    p.add_argument("--src",      type=Path, required=True,
                   help="Root categories folder (contains one subfolder per category)")
    p.add_argument("--csv",      type=Path, required=True,
                   help="Output CSV path for the duplicate report")
    p.add_argument("--category", type=str,  default=None,
                   help="Limit scan to a single category folder (default: all)")
    args = p.parse_args()

    print(f"\nScanning: {args.src.resolve()}\n")
    rows = find_duplicates(args.src.resolve(), args.category)

    total = len(rows)
    total_files = sum(r["occurrences"] for r in rows)
    print(f"\nSummary: {total} duplicate stem(s), {total_files} total file copies involved")

    write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
