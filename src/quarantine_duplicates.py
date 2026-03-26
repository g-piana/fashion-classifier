"""
quarantine_duplicates.py
========================
Reads the duplicate report CSV produced by find_duplicate_stems.py and
moves ALL copies of each duplicate stem into a quarantine folder for
manual review and reassignment.

Quarantine folder structure
---------------------------
    quarantine/
        flat-shoes-women/
            C0S6047__loafer.jpg       ← original subcat encoded in filename
            C0S6047__ballet-flat.jpg
        heeled-shoes-women/
            B5W7023__pump.jpg
            B5W7023__slingback.jpg

The double-underscore separator lets you read at a glance which
subcategory each copy came from. After review, move the correct copy
back into its subcategory folder and delete the rest.

A restore CSV is also written so you can track decisions:
    quarantine/restore_log.csv
    columns: stem, category, quarantine_path, original_subcat, original_path, action

Dry-run mode (--dry-run) prints what would happen without moving anything.

Usage
-----
    # Move all duplicates to quarantine
    python src/quarantine_duplicates.py `
        --duplicates-csv "E:/fashion-data/csv/duplicate_stems.csv" `
        --quarantine-dir "E:/fashion-data/01-RAW/nillab_01/_quarantine"

    # Dry run first to preview
    python src/quarantine_duplicates.py `
        --duplicates-csv "E:/fashion-data/csv/duplicate_stems.csv" `
        --quarantine-dir "E:/fashion-data/01-RAW/nillab_01/_quarantine" `
        --dry-run

After manual review
-------------------
    1. Open quarantine/<category>/ in Explorer or your image viewer
    2. For each stem (pair/group of images), decide which subcategory is correct
    3. Move the correct file back to:
           categories/<category>/<correct_subcat>/<stem>.jpg
       renaming it to remove the __subcat suffix first
    4. Delete the remaining copy/copies
    5. Update restore_log.csv with your decision for traceability
"""

import argparse
import csv
import shutil
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_duplicates(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quarantine_duplicates(
    rows: list[dict],
    quarantine_dir: Path,
    dry_run: bool,
) -> list[dict]:
    """
    Move all copies of each duplicate stem into the quarantine folder.
    Returns restore log rows.
    """
    quarantine_dir = quarantine_dir.resolve()
    restore_log: list[dict] = []

    moved_count   = 0
    skipped_count = 0

    for row in rows:
        stem     = row["stem"]
        category = row["category"]
        n        = int(row["occurrences"])

        cat_quarantine = quarantine_dir / category
        if not dry_run:
            cat_quarantine.mkdir(parents=True, exist_ok=True)

        for i in range(1, n + 1):
            subcat    = row.get(f"subcat_{i}", "")
            orig_path = Path(row.get(f"path_{i}", ""))

            if not orig_path.exists():
                print(f"  WARNING: source not found, skipping: {orig_path}")
                skipped_count += 1
                continue

            ext          = orig_path.suffix.lower()
            new_name     = f"{stem}__{subcat}{ext}"
            dest_path    = cat_quarantine / new_name

            restore_log.append({
                "stem":             stem,
                "category":         category,
                "subcat":           subcat,
                "quarantine_path":  str(dest_path),
                "original_path":    str(orig_path),
                "action":           "",   # filled in manually during review
            })

            if dry_run:
                print(f"  [DRY RUN] {orig_path.relative_to(orig_path.parents[2])}  →  {dest_path.relative_to(quarantine_dir.parent) if quarantine_dir.parent in dest_path.parents else dest_path.name}")
            else:
                if dest_path.exists():
                    print(f"  SKIP (already quarantined): {new_name}")
                    skipped_count += 1
                    continue
                shutil.move(str(orig_path), str(dest_path))
                moved_count += 1
                print(f"  MOVED: {orig_path.name}  →  {new_name}")

    if not dry_run:
        print(f"\nMoved   : {moved_count} files")
        print(f"Skipped : {skipped_count} files")
    else:
        print(f"\n[DRY RUN] Would move {len(restore_log)} files")

    return restore_log


def write_restore_log(log: list[dict], quarantine_dir: Path, dry_run: bool) -> None:
    if not log:
        return

    log_path = quarantine_dir / "restore_log.csv"
    fieldnames = ["stem", "category", "subcat",
                  "quarantine_path", "original_path", "action"]

    if dry_run:
        print(f"\n[DRY RUN] Restore log would be written → {log_path}")
        return

    quarantine_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log)

    print(f"\nRestore log written → {log_path}")
    print(
        "\nNext steps:\n"
        "  1. Open the quarantine folder in Explorer / your image viewer\n"
        "  2. For each stem group (pairs share the same stem prefix), pick the correct subcategory\n"
        "  3. Rename the keeper:  C0S6047__loafer.jpg  →  C0S6047.jpg\n"
        "  4. Move it back to:   categories/<category>/<correct_subcat>/\n"
        "  5. Delete the remaining copies\n"
        "  6. Fill in the 'action' column in restore_log.csv for traceability\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Move duplicate stems to a quarantine folder for manual review"
    )
    p.add_argument("--duplicates-csv",  type=Path, required=True,
                   help="CSV produced by find_duplicate_stems.py")
    p.add_argument("--quarantine-dir",  type=Path, required=True,
                   help="Destination folder for quarantined images")
    p.add_argument("--dry-run",         action="store_true",
                   help="Print what would happen without moving any files")
    args = p.parse_args()

    rows = load_duplicates(args.duplicates_csv)
    if not rows:
        print("No duplicates in CSV — nothing to do.")
        return

    total_files = sum(int(r["occurrences"]) for r in rows)
    print(f"\nLoaded {len(rows)} duplicate stem(s), {total_files} file copies to quarantine")
    print(f"Quarantine dir: {args.quarantine_dir.resolve()}\n")

    if args.dry_run:
        print("=== DRY RUN — no files will be moved ===\n")

    restore_log = quarantine_duplicates(rows, args.quarantine_dir, args.dry_run)
    write_restore_log(restore_log, args.quarantine_dir.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
