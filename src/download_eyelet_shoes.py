"""
download_eyelet_shoes.py
========================
Downloads ~50 shoe images with visible metal eyelets for ML fine-tuning.

Targeting the eyelet embellishment definition used in the ml_workflow taxonomy:
  "Visible metal-rimmed holes on the shoe upper, either for lacing or purely
   decorative. The metal rim must be clearly visible."

Strategy
--------
  - 10 search queries across women / men / style categories
  - 5-6 images per query → ~50 total
  - Saved to E:/fashion-data/01-RAW/eyelet_finetuning/ (Windows path)
  - Basic JPEG conversion + min-size filter to discard thumbnails
  - MD5 dedup to avoid near-duplicates

Requirements
------------
    pip install icrawler Pillow

Usage
-----
    python download_eyelet_shoes.py
    python download_eyelet_shoes.py --out-dir "E:/fashion-data/01-RAW/eyelet_finetuning" --per-query 6
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Search queries — 10 queries × per_query = ~50 images
# Balanced: ~5 women, 5 men queries; mixed styles (oxford, boot, sneaker, derby)
# Phrased to make metal eyelets visible and prominent in results
# ─────────────────────────────────────────────────────────────────────────────
QUERIES = [
    # --- Women ---
    "women oxford shoes visible metal eyelets lace up",
    "women lace up ankle boots metal eyelet hardware",
    "women combat boots prominent metal eyelets close up",
    "women derby shoes metal eyelet lacing detail",
    "women sneakers decorative metal eyelets",
    # --- Men ---
    "men oxford shoes metal eyelets lacing",
    "men derby brogues metal eyelet holes",
    "men leather boots metal eyelet lace holes",
    "men military boots eyelet hardware close up",
    "men sneakers metal eyelet detail shoe upper",
]

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OUT_DIR  = "E:/fashion-data/01-RAW/eyelet_finetuning"
MIN_SIDE_PX      = 300   # discard images smaller than this
JPEG_QUALITY     = 90
TARGET_TOTAL     = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download eyelet shoe images for ML fine-tuning")
    p.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Destination folder (default: {DEFAULT_OUT_DIR})"
    )
    p.add_argument(
        "--per-query", type=int, default=6,
        help="Max images to attempt per search query (default: 6)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print queries and exit without downloading"
    )
    return p.parse_args()


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def clean_and_validate(
    raw_dir: Path,
    out_dir: Path,
    min_side: int = MIN_SIDE_PX,
    jpeg_quality: int = JPEG_QUALITY,
) -> list[Path]:
    """
    Convert to JPEG, enforce min size, remove duplicates.
    Returns list of accepted output paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    accepted: list[Path] = []

    for src in sorted(raw_dir.iterdir()):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {src.name} — cannot open: {e}")
            continue

        if min(img.size) < min_side:
            print(f"  [SKIP] {src.name} — too small ({img.size})")
            continue

        dst = out_dir / (src.stem + ".jpg")
        img.save(dst, "JPEG", quality=jpeg_quality)

        h = md5(dst)
        if h in seen_hashes:
            print(f"  [DUP]  {src.name} — duplicate, removed")
            dst.unlink()
            continue

        seen_hashes.add(h)
        accepted.append(dst)

    return accepted


def download_query(query: str, tmp_dir: Path, max_num: int) -> None:
    """Download images for a single query into tmp_dir."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(storage={"root_dir": str(tmp_dir)})
    crawler.crawl(keyword=query, max_num=max_num, file_idx_offset=0)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    per_query = args.per_query

    print("=" * 60)
    print("Eyelet Shoe Image Downloader")
    print(f"Queries      : {len(QUERIES)}")
    print(f"Per query    : {per_query}")
    print(f"Target total : {TARGET_TOTAL}")
    print(f"Output dir   : {out_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Queries that would be executed:")
        for i, q in enumerate(QUERIES, 1):
            print(f"  {i:02d}. {q}")
        sys.exit(0)

    # Use a temp staging area next to the final output
    tmp_root = out_dir.parent / "_eyelet_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_accepted: list[Path] = []

    for i, query in enumerate(QUERIES, 1):
        tmp_dir = tmp_root / f"q{i:02d}"
        print(f"\n[{i:02d}/{len(QUERIES)}] '{query}'")

        try:
            download_query(query, tmp_dir, per_query)
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            continue

        # Validate + move accepted images into out_dir
        accepted = clean_and_validate(tmp_dir, out_dir, min_side=MIN_SIDE_PX)

        # Rename to avoid collisions across queries
        for path in accepted:
            new_name = out_dir / f"eyelet_q{i:02d}_{path.name}"
            path.rename(new_name)
            all_accepted.append(new_name)

        print(f"  → {len(accepted)} image(s) accepted")

        # Early stop if we've hit the target
        if len(all_accepted) >= TARGET_TOTAL:
            print(f"\n[TARGET REACHED] {len(all_accepted)} images collected.")
            break

    # Cleanup temp dir
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print(f"  Accepted     : {len(all_accepted)} images")
    print(f"  Output dir   : {out_dir}")
    print("=" * 60)

    if len(all_accepted) < TARGET_TOTAL:
        print(
            f"\n[WARNING] Only {len(all_accepted)} images collected (target: {TARGET_TOTAL}).\n"
            "  Consider increasing --per-query or adding more queries."
        )
    else:
        print(
            "\nNext step — verify with infer_shoes.py before adding to training:\n"
            f"  python src/infer_shoes.py \\\n"
            f"    --image-dir \"{out_dir}\" \\\n"
            f"    --out-csv   \"E:/fashion-data/csv/eyelet_check.csv\" \\\n"
            f"    --no-toe-heel\n"
        )


if __name__ == "__main__":
    main()
