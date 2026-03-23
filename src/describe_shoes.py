"""
describe_shoes.py
=================
Generates a concise, factual fashion description for each shoe image.

Reads images from a folder and optionally a tagged CSV (from infer_shoes.py)
to ground descriptions in already-verified attributes rather than letting
the model free-associate.

Descriptions are objective and visual — no brand attribution, no year/era
references, no material guesses beyond what is clearly visible.

Features
--------
  - Resume support: already-described rows are skipped on re-run
  - Prompt versioning: --prompt-version written to output so mixed runs
    are detectable and selectively re-runnable
  - Tag-grounded mode: when tagged CSV is provided, verified tags are
    injected into the prompt to prevent contradiction
  - Crash-safe: output written after every image

Usage
-----
    # Basic — images only, no tags
    python src/describe_shoes.py \
        --image-dir "E:/fashion-data/01-RAW/shoes_production" \
        --out-csv   "E:/fashion-data/csv/shoes_descriptions.csv"

    # Tag-grounded — recommended for production
    python src/describe_shoes.py \
        --image-dir  "E:/fashion-data/01-RAW/shoes_production" \
        --tagged-csv "E:/fashion-data/csv/shoes_tagged.csv" \
        --out-csv    "E:/fashion-data/csv/shoes_descriptions.csv"

    # Resume an interrupted run
    python src/describe_shoes.py \
        --image-dir  "..." \
        --tagged-csv "..." \
        --out-csv    "E:/fashion-data/csv/shoes_descriptions.csv" \
        --resume

    # Dry run
    python src/describe_shoes.py --image-dir "..." --out-csv "..." --dry-run

    # Override prompt version label (increment when you change prompts)
    python src/describe_shoes.py ... --prompt-version v2

Environment
-----------
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL          = "claude-sonnet-4-5"
MAX_IMAGE_PX   = 1568
JPEG_QUALITY   = 85
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5.0
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}

DEFAULT_PROMPT_VERSION = "v1"

# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy reference (for tag formatting in prompt)
# ─────────────────────────────────────────────────────────────────────────────

CONSTRUCTION_ATTRS = [
    "strappy", "ankle-strap", "ankle-wrap", "sling-back",
    "t-bar", "cross-over", "platform", "lace-up", "zip-up", "slouch",
]

EMBELLISHMENT_ATTRS = [
    "bead", "bow", "buckle", "chain", "crystal", "embroidery",
    "eyelet", "feather", "flower", "fringe", "fur", "hardware", "lace",
    "logo", "mesh-insert", "patch", "pearl", "pom-pom", "ribbon", "sequin",
    "stripe", "stud", "tassel", "zipper",
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a footwear product description writer for a fashion retail platform. "
    "Your descriptions are concise, factual and based strictly on what is visually "
    "observable in the image. "
    "You never guess, infer or invent details that are not clearly visible. "
    "You never mention: brand names, designers, year, era, trend references, "
    "material composition (e.g. 'genuine leather', 'suede') unless it is absolutely "
    "unambiguous from the image, comfort claims, quality claims, or comparisons to "
    "other shoes or styles. "
    "Write in plain English, third person, present tense. "
    "Always begin the description directly with the adjective or noun — "
    "for example 'Pointed-toe pump...' or 'Open-toe sandal with ankle strap...'. "
    "Never begin with an article (A, An, The), 'This is', 'This shoe', "
    "or any similar phrasing. "
    "No marketing language. No superlatives. No opinions. "
    "Respond with a single JSON object containing one key: 'description'."
)


def build_prompt_ungrounded() -> str:
    """Prompt when no tagged CSV is available."""
    return (
        "Write a concise product description for this shoe image.\n\n"
        "Structure your description in up to three sentences:\n"
        "1. Heel type and toe shape and overall shoe silhouette "
        "(e.g. pump, sandal, boot, bootie, sneaker, loafer, mule)\n"
        "2. Construction or closure details if clearly visible "
        "(e.g. ankle strap, lace-up, zip closure, platform sole)\n"
        "3. Embellishments or decorative details if present "
        "(e.g. crystal embellishment, bow detail, fringe trim) — "
        "omit this sentence if the shoe is plain\n\n"
        "Rules:\n"
        "- Only describe what is clearly visible\n"
        "- No brand names, designer names, year or era references\n"
        "- No material claims unless unambiguous (e.g. clear plastic sole is fine, "
        "'genuine leather upper' is not)\n"
        "- No comfort, quality or style opinions\n"
        "- Maximum 60 words\n\n"
        'Respond with ONLY: {"description": "your description here"}'
    )


def build_prompt_grounded(tags: dict) -> str:
    """
    Prompt when verified tags are available from infer_shoes.py output.
    Tags are injected to ground the description and prevent contradiction.
    """
    toe    = tags.get("toe_shape", "")
    heel   = tags.get("heel_type", "")

    construction = [
        a for a in CONSTRUCTION_ATTRS
        if str(tags.get(a, "")).lower() == "true"
    ]
    embellishments = [
        a for a in EMBELLISHMENT_ATTRS
        if str(tags.get(a, "")).lower() == "true"
    ]

    tag_block_parts = []
    if toe:
        tag_block_parts.append(f"Toe shape: {toe}")
    if heel:
        tag_block_parts.append(f"Heel type: {heel}")
    if construction:
        tag_block_parts.append(f"Construction: {', '.join(construction)}")
    if embellishments:
        tag_block_parts.append(f"Embellishments: {', '.join(embellishments)}")

    tag_block = "\n".join(tag_block_parts) if tag_block_parts else "(no tags available)"

    return (
        "Write a concise product description for this shoe image.\n\n"
        "The following attributes have already been verified for this shoe — "
        "your description MUST be consistent with them and should incorporate them naturally:\n\n"
        f"{tag_block}\n\n"
        "Structure your description in up to three sentences:\n"
        "1. Heel type and toe shape and overall shoe silhouette "
        "(e.g. pump, sandal, boot, bootie, sneaker, loafer, mule)\n"
        "2. Construction or closure details if present — use the verified tags above\n"
        "3. Embellishments or decorative details if present — use the verified tags above; "
        "omit this sentence if no embellishments are listed\n\n"
        "Rules:\n"
        "- Stay consistent with the verified tags — do not contradict them\n"
        "- You may add visible details not captured by the tags "
        "(e.g. colour, silhouette shape, shaft height for boots)\n"
        "- No brand names, designer names, year or era references\n"
        "- No material claims unless unambiguous from the image\n"
        "- No comfort, quality or style opinions\n"
        "- Maximum 60 words\n\n"
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


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in SUPPORTED_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for sub in image_dir.iterdir():
        if not sub.is_dir():
            continue
        for ext in SUPPORTED_EXTS:
            p = sub / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def encode_image(image_path: Path, max_px: int = MAX_IMAGE_PX) -> tuple[str, str]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def _call(
    client: anthropic.Anthropic,
    b64: str,
    media_type: str,
    user: str,
) -> str | None:
    """Returns description string or None on failure."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        }},
                        {"type": "text", "text": user},
                    ],
                }],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            return parsed.get("description", "").strip()

        except anthropic.RateLimitError:
            if attempt < RETRY_ATTEMPTS - 1:
                tqdm.write(f"  Rate limit — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                tqdm.write("  Rate limit — giving up")
                return None
        except json.JSONDecodeError as e:
            tqdm.write(f"  JSON parse error: {e}")
            return None
        except Exception as e:
            tqdm.write(f"  API error: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_descriptions(
    client: anthropic.Anthropic,
    image_dir: Path,
    tagged_csv: Path | None,
    out_csv: Path,
    resume: bool,
    dry_run: bool,
    prompt_version: str,
    workers: int = 1,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Load tagged CSV if provided
    tags_by_stem: dict[str, dict] = {}
    if tagged_csv and tagged_csv.exists():
        tags_df = pd.read_csv(tagged_csv)
        tags_by_stem = {
            str(row["name"]): row.to_dict()
            for _, row in tags_df.iterrows()
        }
        print(f"Loaded tags for {len(tags_by_stem)} images from {tagged_csv}")
    else:
        if tagged_csv:
            print(f"WARNING: tagged CSV not found at {tagged_csv} — running ungrounded")
        else:
            print("No tagged CSV provided — running ungrounded")

    # Collect images
    images = find_images(image_dir)
    if not images:
        print(f"No images found in {image_dir}")
        return
    print(f"Found {len(images)} images")

    # Resume: load existing output
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
        grounded = sum(1 for p in to_process if p.stem in tags_by_stem)
        print(f"\nDRY RUN — would process {len(to_process)} images")
        print(f"  Tag-grounded: {grounded}  |  Ungrounded: {len(to_process) - grounded}")
        for p in to_process[:10]:
            mode = "grounded" if p.stem in tags_by_stem else "ungrounded"
            print(f"  [{mode}] {p.name}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        return

    if not to_process:
        print("Nothing to process — all images already described.")
        return

    results = list(existing_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Thread-safe counters and CSV writer
    write_header = not (resume and out_csv.exists())
    write_lock   = threading.Lock()
    counters     = {"grounded": 0, "ungrounded": 0, "failed": 0}

    print(f"\nGenerating descriptions ({prompt_version}) "
          f"with {workers} worker(s)...\n")

    def _process_one(img_path: Path) -> dict:
        stem = img_path.stem

        try:
            b64, media_type = encode_image(img_path)
        except Exception as e:
            tqdm.write(f"  ENCODE ERROR {img_path.name}: {e}")
            record = {
                "name": stem, "description": "",
                "grounded": False, "prompt_version": prompt_version,
            }
            with write_lock:
                counters["failed"] += 1
                _append_row(record)
            return record

        if stem in tags_by_stem:
            prompt   = build_prompt_grounded(tags_by_stem[stem])
            is_grounded = True
        else:
            prompt   = build_prompt_ungrounded()
            is_grounded = False

        description = _call(client, b64, media_type, prompt)

        if description is None:
            tqdm.write(f"  FAILED: {stem}")
            description = ""

        record = {
            "name":           stem,
            "description":    description,
            "grounded":       is_grounded,
            "prompt_version": prompt_version,
        }

        with write_lock:
            if is_grounded:
                counters["grounded"] += 1
            else:
                counters["ungrounded"] += 1
            if not description:
                counters["failed"] += 1
            _append_row(record)

        return record

    def _append_row(record: dict) -> None:
        """Must be called inside write_lock."""
        nonlocal write_header
        row_df = pd.DataFrame([record])
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            row_df.to_csv(f, index=False, header=write_header)
        write_header = False

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

    print(f"\nDone.")
    print(f"  Total described : {len(results)}")
    print(f"  Tag-grounded    : {counters['grounded']}")
    print(f"  Ungrounded      : {counters['ungrounded']}")
    print(f"  Failed          : {counters['failed']}")
    print(f"  Output          : {out_csv}")

    # Sample output
    df = pd.read_csv(out_csv)
    non_empty = df[df["description"].str.len() > 0]
    if len(non_empty) > 0:
        print(f"\nSample descriptions:")
        for _, row in non_empty.head(3).iterrows():
            print(f"\n  [{row['name']}]")
            print(f"  {row['description']}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate factual fashion descriptions for shoe images"
    )
    p.add_argument("--image-dir",   type=Path, required=True,
                   help="Folder containing shoe images (searched recursively)")
    p.add_argument("--tagged-csv",  type=Path, default=None,
                   help="Tagged CSV from infer_shoes.py (optional but recommended)")
    p.add_argument("--out-csv",     type=Path, required=True,
                   help="Output CSV path (name, description, grounded, prompt_version)")
    p.add_argument("--resume",      action="store_true",
                   help="Skip images already present in out-csv")
    p.add_argument("--dry-run",     action="store_true",
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
            "Run: $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)"
        )
    client = anthropic.Anthropic(api_key=api_key or "dry-run")

    run_descriptions(
        client=client,
        image_dir=args.image_dir,
        tagged_csv=args.tagged_csv,
        out_csv=args.out_csv,
        resume=args.resume,
        dry_run=args.dry_run,
        prompt_version=args.prompt_version,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
