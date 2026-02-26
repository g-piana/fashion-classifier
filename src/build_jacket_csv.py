"""
build_jacket_csv.py  v2
=======================
Scans a DeepFashion image root directory, maps folder names to jacket
subcategory labels using keyword rules, copies images with unique
global filenames, and writes a CSV ready for the training pipeline.

Usage
-----
    python build_jacket_csv.py \
        --src   "E:/fashion-data/01-RAW/img/img" \
        --dst   "E:/fashion-data/01-RAW/jackets_raw" \
        --csv   "E:/fashion-data/csv/labels_jackets.csv" \
        --copy                 # omit --copy for a dry-run (no files moved)
        --min_per_class 30     # skip classes with fewer images (optional)

Output CSV columns
------------------
    name   : unique image stem (no extension), matches the copied filename
    Class  : subcategory label (bomber, blazer, ...)
    source : original relative path inside --src (for traceability)
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Global excludes — if ANY of these appear in the folder name, skip it
# regardless of which rule would otherwise match.
# ---------------------------------------------------------------------------
GLOBAL_EXCLUDES = [
    # Non-jacket garments
    "vest", "cardigan", "hoodie", "sweatshirt", "pullover",
    "tunic", "tee", "shirt", "blouse",
    "dress", "skirt", "pants", "shorts",
    "jegging", "jeans", "jean",          # moto jeans / jeggings
    "tank",                              # moto tank tops
    "sweater", "knit_sweater",           # faux fur sweaters
    # Accessories / footwear
    "shoe", "bag", "scarf", "hat", "glove", "sock", "accessory",
    # Non-jacket tops
    "top",
]

# ---------------------------------------------------------------------------
# Keyword rules  ->  label
#   Each rule is (label, required_keywords, excluded_keywords)
#   A folder matches if its lowercase name contains ALL required keywords
#   and NONE of the excluded keywords.
#   Rules are evaluated in ORDER — first match wins.
#
#   IMPORTANT: parka rules come BEFORE fur jacket rules so that
#   "Faux_Fur_Hood_Parka" and "Faux_Shearling_Hooded_Parka" are
#   correctly classified as parka rather than fur jacket.
# ---------------------------------------------------------------------------
RULES = [
    # baseball jacket — exclude plain jerseys (no jacket silhouette)
    ("baseball",   ["baseball"],        ["jersey"]),

    # biker / moto jacket
    ("biker",      ["biker"],           []),
    ("biker",      ["moto"],            []),

    # blazer
    ("blazer",     ["blazer"],          []),

    # bomber
    ("bomber",     ["bomber"],          []),
    ("bomber",     ["flight"],          ["flight_suit"]),

    # parka — BEFORE fur jacket so fur-trimmed parkas go here
    ("parka",      ["parka"],           []),
    ("parka",      ["anorak"],          []),

    # fur jacket — after parka
    ("fur jacket", ["faux_fur"],        []),
    ("fur jacket", ["fur"],             []),
    ("fur jacket", ["shearling"],       []),
    ("fur jacket", ["teddy"],           ["bear"]),
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def classify_folder(folder_name: str) -> str | None:
    """Return label for a folder name, or None if no rule matches."""
    name_lower = folder_name.lower().replace("-", "_").replace(" ", "_")

    # Global exclusion — bail immediately if any exclude word found
    for excl in GLOBAL_EXCLUDES:
        if excl in name_lower:
            return None

    # Apply rules in order — first match wins
    for label, required, excluded in RULES:
        if all(kw in name_lower for kw in required):
            if not any(kw in name_lower for kw in excluded):
                return label

    return None


def build_csv(src: Path, dst: Path, csv_path: Path,
              do_copy: bool, min_per_class: int) -> None:

    src = src.resolve()
    dst = dst.resolve()
    csv_path = csv_path.resolve()

    print(f"\nScanning : {src}")
    print(f"Output   : {dst}")
    print(f"CSV      : {csv_path}")
    print(f"Copy     : {do_copy}")
    print(f"Min/class: {min_per_class}\n")

    # --- First pass: collect all matched folders ---
    folder_map: dict[str, list[Path]] = defaultdict(list)
    skipped_folders = []
    matched_folders = []

    for folder in sorted(src.iterdir()):
        if not folder.is_dir():
            continue
        label = classify_folder(folder.name)
        if label is None:
            skipped_folders.append(folder.name)
            continue

        images = [
            f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]
        if not images:
            continue

        folder_map[label].extend(images)
        matched_folders.append((folder.name, label, len(images)))

    # --- Report matched folders grouped by label ---
    print("Matched folders:")
    current_label = None
    for fname, lbl, count in sorted(matched_folders, key=lambda x: (x[1], x[0])):
        if lbl != current_label:
            print(f"\n  [{lbl.upper()}]")
            current_label = lbl
        print(f"    {fname}  ({count} images)")

    print(f"\nSkipped folders : {len(skipped_folders)}")

    # --- Filter classes below minimum ---
    rows = []
    class_counts: dict[str, int] = defaultdict(int)

    for label, img_paths in sorted(folder_map.items()):
        if len(img_paths) < min_per_class:
            print(f"  SKIP class '{label}': only {len(img_paths)} images "
                  f"(min={min_per_class})")
            continue
        for img_path in img_paths:
            rows.append((label, img_path))
            class_counts[label] += 1

    print(f"\nClass summary after min filter:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label:12s} : {count:>5} images")
    print(f"  {'TOTAL':12s} : {sum(class_counts.values()):>5} images")

    if not rows:
        print("\nERROR: No images passed all filters. Check --src path and rules.")
        return

    # --- Copy files and write CSV ---
    if do_copy:
        dst.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    global_counter: dict[str, int] = defaultdict(int)

    for label, img_path in rows:
        global_counter[label] += 1
        stem = f"{label.replace(' ', '_')}_{global_counter[label]:05d}"
        new_filename = stem + img_path.suffix.lower()

        if do_copy:
            dst_file = dst / new_filename
            if not dst_file.exists():
                shutil.copy2(img_path, dst_file)

        csv_rows.append({
            "name":   stem,
            "Class":  label,
            "source": str(img_path.relative_to(src)),
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "Class", "source"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'DRY RUN - no files copied' if not do_copy else 'DONE'}:")
    print(f"  CSV rows written  : {len(csv_rows)}")
    if do_copy:
        print(f"  Images copied to  : {dst}")
    print(f"  CSV saved to      : {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Map DeepFashion folders -> jacket subcategory CSV"
    )
    parser.add_argument("--src", required=True,
                        help="Path to DeepFashion img/img folder")
    parser.add_argument("--dst", required=True,
                        help="Destination folder for renamed images")
    parser.add_argument("--csv", required=True,
                        help="Output CSV path")
    parser.add_argument("--copy", action="store_true",
                        help="Actually copy files (default: dry run)")
    parser.add_argument("--min_per_class", type=int, default=30,
                        help="Minimum images per class (default: 30)")
    args = parser.parse_args()

    build_csv(
        src=Path(args.src),
        dst=Path(args.dst),
        csv_path=Path(args.csv),
        do_copy=args.copy,
        min_per_class=args.min_per_class,
    )


if __name__ == "__main__":
    main()
