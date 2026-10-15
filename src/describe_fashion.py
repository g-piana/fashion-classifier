"""
describe_fashion.py
===================
Generates precise, editorial-quality fashion descriptions for garment images.

Descriptions follow the style of specialist fashion publications — precise
terminology, visually grounded, objective. No brand attribution, no designer
or collection inference, no material guesses beyond what is unambiguously
visible.

Garment-agnostic: handles jackets, coats, dresses, tops, trousers, skirts,
knitwear, outerwear and any other apparel category in a single run.

Features
--------
  - Resume support: already-described rows are skipped on re-run
  - Prompt versioning: --prompt-version written to output so mixed runs
    are detectable and selectively re-runnable
  - Crash-safe: output written after every image (O(1) per write)
  - Multi-worker: --workers N for parallel API calls

Output columns
--------------
  name            — image stem (filename without extension)
  description     — editorial description string
  prompt_version  — prompt version label for traceability

Usage
-----
    # Basic run
    python src/describe_fashion.py \\
        --image-dir "E:/fashion-data/01-RAW/garments_production" \\
        --out-csv   "E:/fashion-data/csv/garments_descriptions.csv"

    # Resume an interrupted run
    python src/describe_fashion.py \\
        --image-dir "E:/fashion-data/01-RAW/garments_production" \\
        --out-csv   "E:/fashion-data/csv/garments_descriptions.csv" \\
        --resume

    # Parallel run (recommended for large batches)
    python src/describe_fashion.py \\
        --image-dir "E:/fashion-data/01-RAW/garments_production" \\
        --out-csv   "E:/fashion-data/csv/garments_descriptions.csv" \\
        --workers 5 --resume

    # Dry run — see what would be processed without calling the API
    python src/describe_fashion.py \\
        --image-dir "E:/fashion-data/01-RAW/garments_production" \\
        --out-csv   "E:/fashion-data/csv/garments_descriptions.csv" \\
        --dry-run

    # Increment prompt version when you change the prompt
    python src/describe_fashion.py ... --prompt-version v2

Environment
-----------
    $env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
    export ANTHROPIC_API_KEY="sk-ant-..."   # bash/zsh
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL                  = "claude-sonnet-4-5"
MAX_IMAGE_PX           = 1568
JPEG_QUALITY           = 85
RETRY_ATTEMPTS         = 4
RETRY_DELAY            = 5.0          # seconds; doubled on each retry
SUPPORTED_EXTS         = {".jpg", ".jpeg", ".png", ".webp",
                           ".JPG", ".JPEG", ".PNG", ".WEBP"}
DEFAULT_PROMPT_VERSION = "v1"

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a specialist fashion writer contributing to a high-end fashion "
    "publication. You write precise, editorial product descriptions for garment "
    "images. Your descriptions use accurate fashion terminology and are grounded "
    "strictly in what is visually observable in the image. "
    "You never guess, infer or invent details that are not clearly visible. "
    "You never mention: brand names, designer names, collection season or year, "
    "trend references, styling context, price or value judgements, comfort or "
    "wearability claims, or comparisons to other garments or designers. "
    "You never guess fabric composition unless it is absolutely unambiguous from "
    "the image (e.g. sheer organza is acceptable; 'silk' or 'cashmere' are not). "
    "Write in plain English, third person, present tense. "
    "Always begin directly with the garment — lead with its defining "
    "characteristic: silhouette, cut, or garment type. "
    "Never open with 'This', 'The garment', 'A', 'An', or any article or "
    "demonstrative pronoun. "
    "No marketing language. No superlatives. No opinions. "
    "Respond with a single JSON object containing one key: 'description'."
)

USER_PROMPT = (
    "Write an editorial fashion description for this garment image.\n\n"
    "Structure the description across two to four sentences:\n"
    "1. Open with the garment type, silhouette and dominant colour — "
    "use precise fashion terminology "
    "(e.g. 'double-breasted overcoat', 'slip dress', 'wide-leg tailored trouser', "
    "'asymmetric-hem midi skirt', 'ribbed turtleneck knit').\n"
    "2. Describe the cut, fit and construction details that define the garment "
    "(e.g. notched lapels, raglan sleeves, dropped shoulder, pintuck front, "
    "flared hem, bias cut, patch pockets, belted waist).\n"
    "3. Describe closure, fastening or structural details if clearly visible "
    "(e.g. concealed front zip, double-breasted button fastening, "
    "exposed seam detailing, drawstring hem).\n"
    "4. Describe embellishments or decorative details if present "
    "(e.g. contrast topstitching, embroidered motif, lace trim, sequin "
    "embellishment, pleated ruffle insert) — omit this sentence if the "
    "garment is plain.\n\n"
    "Colour guidance:\n"
    "- Name the colour precisely: 'ivory', 'ecru', 'cobalt', 'burgundy', "
    "'forest green' rather than generic 'white', 'blue', 'red', 'green'.\n"
    "- If the garment is multicolour, name the dominant colour first then "
    "the secondary.\n"
    "- If there is a print or pattern, name it: 'houndstooth', 'Prince of Wales "
    "check', 'abstract floral print', 'vertical pinstripe'.\n\n"
    "Rules:\n"
    "- Only describe what is clearly visible in the image.\n"
    "- Use fashion-specific vocabulary throughout: avoid generic terms like "
    "'nice detail' or 'decorative element' — name the detail precisely.\n"
    "- No brand names, designer names, year, era or trend references.\n"
    "- No material composition claims unless unambiguous from the image.\n"
    "- No styling, occasion or wearability commentary.\n"
    "- Target length: 60–100 words.\n\n"
    'Respond with ONLY: {"description": "your description here"}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_images(image_dir: Path) -> list[Path]:
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(image_dir.rglob(f"*{ext}"))
    return sorted(set(images))


def encode_image(image_path: Path, max_px: int = MAX_IMAGE_PX) -> tuple[str, str]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return b64, "image/jpeg"

# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def _call(client: anthropic.Anthropic, b64: str, media_type: str) -> str | None:
    """Returns description string or None on unrecoverable failure."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        }},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }],
            )
            raw = response.content[0].text.strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            return parsed.get("description", "").strip()

        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            tqdm.write(f"  Rate limit — waiting {wait:.0f}s (attempt {attempt + 1})")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            tqdm.write(f"  JSON parse error: {e}")
            return None
        except Exception as e:
            tqdm.write(f"  API error: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))

    tqdm.write("  Giving up after max retries")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

COLS = ["name", "description", "prompt_version"]


def run_descriptions(
    client: anthropic.Anthropic,
    image_dir: Path,
    out_csv: Path,
    resume: bool,
    dry_run: bool,
    prompt_version: str,
    workers: int = 1,
) -> None:

    images = find_images(image_dir)
    if not images:
        print(f"No images found in {image_dir}")
        return
    print(f"Found {len(images)} images in {image_dir}")

    # Resume: load existing output and skip already-done stems
    done_stems: set[str] = set()
    existing_rows: list[dict] = []
    if resume and out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        done_stems    = set(existing_df["name"].astype(str))
        existing_rows = existing_df.to_dict("records")
        print(f"Resuming — {len(done_stems)} done, "
              f"{len(images) - len(done_stems)} remaining")

    to_process = [p for p in images if p.stem not in done_stems]

    if dry_run:
        print(f"\nDRY RUN — would process {len(to_process)} images")
        for p in to_process[:10]:
            print(f"  {p.name}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        return

    if not to_process:
        print("Nothing to process — all images already described.")
        return

    results      = list(existing_rows)
    failed_count = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating descriptions ({prompt_version}) "
          f"with {workers} worker(s)...\n")

    write_header = not (resume and out_csv.exists())
    write_lock   = threading.Lock()

    def _append_row(record: dict) -> None:
        """Append one row to the CSV. Must be called inside write_lock."""
        nonlocal write_header
        row_df = pd.DataFrame([record])[COLS]
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            row_df.to_csv(f, index=False, header=write_header)
        write_header = False

    def _process_one(img_path: Path) -> dict:
        nonlocal failed_count
        stem = img_path.stem

        try:
            b64, media_type = encode_image(img_path)
        except Exception as e:
            tqdm.write(f"  ENCODE ERROR {img_path.name}: {e}")
            record = {"name": stem, "description": "", "prompt_version": prompt_version}
            with write_lock:
                failed_count += 1
                _append_row(record)
            return record

        description = _call(client, b64, media_type)

        if description is None:
            tqdm.write(f"  FAILED: {stem}")
            description = ""

        record = {"name": stem, "description": description, "prompt_version": prompt_version}

        with write_lock:
            if not description:
                failed_count += 1
            _append_row(record)

        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, p): p for p in to_process}
        with tqdm(total=len(to_process), desc="Describing") as bar:
            for future in as_completed(futures):
                try:
                    record = future.result()
                    results.append(record)
                except Exception as e:
                    img_path = futures[future]
                    tqdm.write(f"  WORKER ERROR {img_path.name}: {e}")
                finally:
                    bar.update(1)

    succeeded = len(results) - failed_count
    print(f"\nDone.")
    print(f"  Total written  : {len(results)}")
    print(f"  Succeeded      : {succeeded}")
    print(f"  Failed         : {failed_count}")
    print(f"  Output         : {out_csv}")

    # Sample output from this run (non-empty descriptions only)
    just_run = [r for r in results if r.get("description")]
    if just_run:
        print("\nSample descriptions:")
        for r in just_run[:3]:
            print(f"\n  [{r['name']}]")
            print(f"  {r['description']}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate editorial fashion descriptions for garment images"
    )
    p.add_argument("--image-dir",  type=Path, required=True,
                   help="Folder containing garment images (searched recursively)")
    p.add_argument("--out-csv",    type=Path, required=True,
                   help="Output CSV path (name, description, prompt_version)")
    p.add_argument("--resume",     action="store_true",
                   help="Skip images already present in out-csv")
    p.add_argument("--dry-run",    action="store_true",
                   help="Show what would be processed without calling the API")
    p.add_argument("--prompt-version", type=str, default=DEFAULT_PROMPT_VERSION,
                   help=f"Label for this prompt version, written to output CSV "
                        f"(default: {DEFAULT_PROMPT_VERSION}). "
                        f"Increment (v2, v3...) when you change the prompt so "
                        f"mixed runs are detectable.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers (default: 1, recommended: 4-5)")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "PowerShell : $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "bash/zsh   : export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    client = anthropic.Anthropic(api_key=api_key or "dry-run")

    run_descriptions(
        client=client,
        image_dir=args.image_dir,
        out_csv=args.out_csv,
        resume=args.resume,
        dry_run=args.dry_run,
        prompt_version=args.prompt_version,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
