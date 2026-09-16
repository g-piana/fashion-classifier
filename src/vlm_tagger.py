"""
vlm_tagger.py
=============
Domain-agnostic VLM attribute tagging engine.

Reads attribute definitions from taxonomies/vlm/<domain>.yaml and
calls the Claude API to tag images. Works for any domain — shoes,
jackets, bags — as long as a VLM config file exists.

This module is called by pipeline.py when --vlm is passed. It can
also be run standalone for batch attribute tagging.

Architecture
------------
Tier 1 (ResNet)  — category + subcategory  →  pipeline.py / run_cascade()
Tier 2 (VLM)     — fine-grained attributes →  vlm_tagger.py / tag_images()

The two tiers are independent: VLM receives raw image paths, ResNet
receives preprocessed npy files. They share only the output CSV.

Usage (standalone)
------------------
    python src/vlm_tagger.py `
        --domain    shoes `
        --image-dir "E:/fashion-data/01-RAW/nillab_01/photo" `
        --out-csv   "E:/fashion-data/csv/vlm_shoes.csv" `
        --resume

    # Dry run — shows what would be called without hitting the API
    python src/vlm_tagger.py --domain shoes --image-dir "..." --out-csv "..." --dry-run

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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}
MAX_IMAGE_PX      = 1568
JPEG_QUALITY      = 85
DEFAULT_VLM_ROOT  = Path(__file__).parent.parent / "taxonomies" / "vlm"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_vlm_config(domain: str, vlm_root: Path) -> dict:
    """Load taxonomies/vlm/<domain>.yaml."""
    path = Path(vlm_root) / f"{domain}.yaml"
    if not path.exists():
        available = [p.stem for p in Path(vlm_root).glob("*.yaml")]
        raise FileNotFoundError(
            f"No VLM config found for domain '{domain}' at {path}\n"
            f"Available: {available}\n"
            f"Create taxonomies/vlm/{domain}.yaml to enable VLM tagging."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image(image_path: Path) -> tuple[str, str]:
    """
    Load, resize if needed, and base64-encode an image for the Claude API.
    Returns (base64_string, media_type).
    """
    img = Image.open(image_path).convert("RGB")

    # Resize if larger than MAX_IMAGE_PX on either axis
    if max(img.size) > MAX_IMAGE_PX:
        ratio = MAX_IMAGE_PX / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


# ---------------------------------------------------------------------------
# Prompt builders — driven by VLM config
# ---------------------------------------------------------------------------

def _build_categorical_prompt(group_name: str, group_cfg: dict) -> str:
    """Build prompt for a categorical attribute group (e.g. toe_shape + heel_type)."""
    lines = []
    json_fields = []

    for attr_name, attr_cfg in group_cfg["attributes"].items():
        classes     = attr_cfg["classes"]
        descriptions = attr_cfg.get("descriptions", {})

        class_block = "\n".join(
            f'  - "{c}": {descriptions.get(c, c)}' for c in classes
        )
        class_list = ", ".join(f'"{c}"' for c in classes)

        lines.append(f"{attr_name.upper().replace('_', ' ')} options:\n{class_block}")
        lines.append(f"- Choose EXACTLY ONE {attr_name} from: [{class_list}]")
        json_fields.append(f'  "{attr_name}": "<value>"')

    json_template = "{\n" + ",\n".join(json_fields) + "\n}"
    return (
        f"Examine this image carefully.\n\n"
        + "\n\n".join(lines)
        + f"\n\nRespond with ONLY this JSON:\n{json_template}"
    )


def _build_multi_label_prompt(group_name: str, group_cfg: dict) -> str:
    """Build prompt for a multi-label attribute group (true/false per attribute)."""
    attr_lines = []
    json_fields = []

    for attr_name, attr_cfg in group_cfg["attributes"].items():
        desc = attr_cfg.get("description", attr_name.replace("-", " "))
        attr_lines.append(f'  "{attr_name}": {desc}')
        json_fields.append(f'  "{attr_name}": true_or_false')

    attrs_block   = "\n".join(attr_lines)
    json_template = "{\n" + "\n".join(json_fields) + "\n}"

    return (
        f"Examine this image and detect which attributes are present.\n\n"
        f"Attributes:\n{attrs_block}\n\n"
        f"Rules:\n"
        f"- true = attribute is clearly visible\n"
        f"- false = absent, not visible, or uncertain\n"
        f"- Multiple attributes can be true simultaneously\n"
        f"- No explanations, no markdown, just the JSON\n\n"
        f"Respond with ONLY:\n{json_template}"
    )


def build_prompt(group_name: str, group_cfg: dict) -> str:
    label_type = group_cfg["label_type"]
    if label_type == "categorical":
        return _build_categorical_prompt(group_name, group_cfg)
    elif label_type == "multi_label":
        return _build_multi_label_prompt(group_name, group_cfg)
    else:
        raise ValueError(f"Unknown label_type '{label_type}' in group '{group_name}'")


# ---------------------------------------------------------------------------
# Single API call with retry
# ---------------------------------------------------------------------------

def _api_call(
    client:        anthropic.Anthropic,
    model:         str,
    b64:           str,
    media_type:    str,
    system_prompt: str,
    user_prompt:   str,
    retry_attempts: int,
    retry_delay:   float,
) -> dict | None:
    """Make one Claude API call. Returns parsed JSON dict or None on failure."""
    for attempt in range(retry_attempts):
        try:
            image_block = {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": media_type,
                    "data":       b64,
                },
            }
            text_block = {"type": "text", "text": user_prompt}
            response = client.messages.create(
                model       = model,
                max_tokens  = 512,
                temperature = 0,
                system      = system_prompt,
                messages    = [{"role": "user", "content": [image_block, text_block]}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            return json.loads(raw)

        except anthropic.RateLimitError:
            if attempt < retry_attempts - 1:
                print(f"\n  [VLM] Rate limit — waiting {retry_delay}s...")
                time.sleep(retry_delay)
        except json.JSONDecodeError as e:
            # Log the raw response so we can diagnose prompt issues
            try:
                raw_text = response.content[0].text[:300]
            except Exception:
                raw_text = "<unavailable>"
            print(f"\n  [VLM] JSON parse error: {e}")
            print(f"  [VLM] Raw response   : {raw_text!r}")
            return None
        except Exception as e:
            print(f"\n  [VLM] API error ({type(e).__name__}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
    return None


# ---------------------------------------------------------------------------
# Tag a single image
# ---------------------------------------------------------------------------

def tag_image(
    image_path:    Path,
    vlm_cfg:       dict,
    client:        anthropic.Anthropic,
) -> dict[str, Any]:
    """
    Run all attribute groups for one image.
    Returns a flat dict: {attr_name: value, ...} with image stem.
    Empty string means the API call failed for that group.
    """
    record: dict[str, Any] = {"name": image_path.stem}

    # Initialise all output columns to empty string
    for group_cfg in vlm_cfg["attribute_groups"].values():
        for attr_name in group_cfg["attributes"]:
            record[attr_name] = ""

    # Encode image once, reuse across all group calls
    try:
        b64, media_type = encode_image(image_path)
    except Exception:
        return record   # all columns stay empty

    model           = vlm_cfg.get("model", "claude-sonnet-4-6")
    retry_attempts  = vlm_cfg.get("retry_attempts", 3)
    retry_delay     = vlm_cfg.get("retry_delay_s", 5.0)

    for group_name, group_cfg in vlm_cfg["attribute_groups"].items():
        system_prompt = group_cfg["system_prompt"].strip()
        label_type    = group_cfg["label_type"]

        # Auto-chunk large multi_label groups to keep prompts manageable.
        # Threshold: >24 attributes -> split into chunks of 24.
        # Categorical groups are never chunked (always small).
        MAX_ATTRS  = vlm_cfg.get("max_attrs_per_call", 24)
        attrs      = group_cfg["attributes"]

        if label_type == "multi_label" and len(attrs) > MAX_ATTRS:
            attr_items = list(attrs.items())
            chunks = [
                dict(attr_items[i:i + MAX_ATTRS])
                for i in range(0, len(attr_items), MAX_ATTRS)
            ]
        else:
            chunks = [attrs]

        for chunk_idx, chunk_attrs in enumerate(chunks):
            chunk_cfg   = {**group_cfg, "attributes": chunk_attrs}
            user_prompt = build_prompt(group_name, chunk_cfg)
            suffix      = f" (part {chunk_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""

            result = _api_call(
                client         = client,
                model          = model,
                b64            = b64,
                media_type     = media_type,
                system_prompt  = system_prompt,
                user_prompt    = user_prompt,
                retry_attempts = retry_attempts,
                retry_delay    = retry_delay,
            )

            if result is None:
                print(f"  [VLM] {group_name}{suffix} returned None — skipping chunk")
                continue

            if label_type == "categorical":
                for attr_name in chunk_attrs:
                    value         = result.get(attr_name, "")
                    valid_classes = chunk_attrs[attr_name].get("classes", [])
                    if value in valid_classes:
                        record[attr_name] = value
                    else:
                        record[attr_name] = f"_invalid:{value}" if value else ""

            elif label_type == "multi_label":
                for attr_name in chunk_attrs:
                    record[attr_name] = bool(result.get(attr_name, False))

    return record


# ---------------------------------------------------------------------------
# Batch tagging with concurrency
# ---------------------------------------------------------------------------

def tag_images(
    image_paths:  list[Path],
    vlm_cfg:      dict,
    client:       anthropic.Anthropic,
    resume_stems: set[str] | None = None,
) -> list[dict]:
    """
    Tag a list of images using the VLM config.
    Skips stems already in resume_stems.
    Returns list of result dicts (one per image).
    """
    to_process = [
        p for p in image_paths
        if (resume_stems is None or p.stem not in resume_stems)
    ]

    if not to_process:
        return []

    max_workers = vlm_cfg.get("max_workers", 4)
    results     = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(tag_image, img_path, vlm_cfg, client): img_path
            for img_path in to_process
        }

        with tqdm(total=len(futures), desc="VLM tagging", unit="img") as pbar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    img_path = futures[future]
                    results.append({"name": img_path.stem})  # empty row
                pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Main — standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Domain-agnostic VLM attribute tagger"
    )
    p.add_argument("--domain",     required=True,
                   help="Domain name, e.g. shoes, jackets")
    p.add_argument("--image-dir",  type=Path, required=True,
                   help="Directory of raw images")
    p.add_argument("--out-csv",    type=Path, required=True,
                   help="Output CSV path")
    p.add_argument("--resume",     action="store_true",
                   help="Skip images already in out-csv")
    p.add_argument("--dry-run",    action="store_true",
                   help="Show what would be processed without calling API")
    p.add_argument("--vlm-root",   type=Path, default=DEFAULT_VLM_ROOT,
                   help="Path to taxonomies/vlm/ directory")
    p.add_argument("--test-one",   type=Path, default=None,
                   help="Tag a single image and print result — for diagnosis")
    args = p.parse_args()

    vlm_cfg = load_vlm_config(args.domain, args.vlm_root)

    image_paths = sorted(
        p for p in args.image_dir.rglob("*")
        if p.suffix in SUPPORTED_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    print(f"\nDomain     : {args.domain}")
    print(f"Images     : {len(image_paths)}")
    print(f"Groups     : {list(vlm_cfg['attribute_groups'].keys())}")
    print(f"Workers    : {vlm_cfg.get('max_workers', 4)}")
    print(f"Output     : {args.out_csv}")

    # Resume
    resume_stems: set[str] = set()
    existing_rows: list[dict] = []
    if args.resume and args.out_csv.exists():
        existing_df  = pd.read_csv(args.out_csv)
        resume_stems = set(existing_df["name"].astype(str))
        existing_rows = existing_df.to_dict("records")
        remaining = len(image_paths) - len(resume_stems)
        print(f"Resuming   : {len(resume_stems)} done, {remaining} remaining")

    if args.test_one:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=api_key)
        print(f"\nTesting single image: {args.test_one}")
        result = tag_image(args.test_one, vlm_cfg, client)
        print(f"\nResult:")
        for k, v in result.items():
            print(f"  {k:<25} {v}")
        return

    if args.dry_run:
        to_process = [p for p in image_paths if p.stem not in resume_stems]
        print(f"\n[DRY RUN] Would process {len(to_process)} images")
        for img in to_process[:5]:
            print(f"  {img.name}")
        if len(to_process) > 5:
            print(f"  ... and {len(to_process) - 5} more")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)"
        )
    client = anthropic.Anthropic(api_key=api_key)

    new_rows = tag_images(image_paths, vlm_cfg, client, resume_stems)

    # Merge with existing and save
    all_rows = existing_rows + new_rows
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)

    print(f"\n✓  {len(new_rows)} images tagged → {args.out_csv}")
    print(f"   Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
