"""
eval_vlm_attributes.py
======================
Quick experiment: evaluate Claude's zero-shot attribute recognition
against your existing ground-truth labels.

Reads your labels CSV to get ground truth, loads the corresponding
raw images (JPG/PNG from jackets_img/), calls Claude claude-sonnet-4-20250514
via API, and computes F1/precision/recall vs your labels.

Requirements
------------
    pip install anthropic pillow scikit-learn pandas tqdm

Setup
-----
    1. Get API key from console.anthropic.com
    2. Set environment variable:
         $env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
         set ANTHROPIC_API_KEY=sk-ant-...         # CMD

Usage
-----
    # Basic run — epaulette and belted, val split only
    python src/eval_vlm_attributes.py \
        --labels-csv  "E:/fashion-data/csv/labels_model_details.csv" \
        --image-dir   "E:/fashion-data/01-RAW/jackets_img" \
        --attributes  epaulette belted \
        --max-images  50

    # All model_details attributes, full dataset
    python src/eval_vlm_attributes.py \
        --labels-csv  "E:/fashion-data/csv/labels_model_details.csv" \
        --image-dir   "E:/fashion-data/01-RAW/jackets_img" \
        --attributes  epaulette belted cropped asymmetric \
        --max-images  100

    # Use a specific seed split to match your val split in train.py
    python src/eval_vlm_attributes.py \
        --labels-csv  "E:/fashion-data/csv/labels_model_details.csv" \
        --image-dir   "E:/fashion-data/01-RAW/jackets_img" \
        --attributes  epaulette belted \
        --val-only    # uses same 80/20 seed=42 split as train.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL            = "claude-sonnet-4-5"
MAX_IMAGE_PX     = 1568          # Claude's recommended max dimension
JPEG_QUALITY     = 85
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5.0           # seconds between retries on rate-limit
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_image(image_dir: Path, stem: str) -> Path | None:
    """Find image by stem, checking flat dir and one level of subfolders."""
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
    """
    Load, optionally resize, and base64-encode an image.
    Returns (base64_string, media_type).
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

ATTRIBUTE_DESCRIPTIONS = {
    "epaulette": (
        "epaulette or epaulettes — structured fabric strap(s) on top of the shoulder(s), "
        "often with a button. Common on military-style jackets and trench coats."
    ),
    "belted": (
        "belt, sash, or waist tie — a separate strap cinching or wrapping around the waist "
        "or hips. Can be fabric, leather, or cord. Includes detachable and sewn-in belts."
    ),
    "cropped": (
        "cropped length — the garment hem ends above the natural hip, at the waist, "
        "or even shorter (bolero). Noticeably shorter than a standard jacket."
    ),
    "asymmetric": (
        "asymmetric design — the garment has a deliberately uneven silhouette: e.g. an "
        "asymmetric hem, diagonal zip closure, one-sided lapel, or offset front panel."
    ),
    "hooded": (
        "hood — a fabric hood attached to the garment, whether down or up."
    ),
    "drawstring": (
        "drawstring — a cord or tie that can be pulled to cinch the hood, hem, or waist."
    ),
    "double_breasted": (
        "double-breasted — two parallel vertical rows of buttons on the front, "
        "with a wide overlapping front panel."
    ),
    "chest_pocket": (
        "chest pocket — a visible pocket on the chest/upper-front area of the garment."
    ),
}

SYSTEM_PROMPT = (
    "You are a fashion attribute recognition system. "
    "Your job is to detect specific garment attributes in product images. "
    "Be precise: only mark an attribute as present if it is clearly visible. "
    "Do not infer from context — if you cannot see it, mark it absent. "
    "Always respond with valid JSON only, no other text."
)


def build_user_prompt(attributes: list[str]) -> str:
    attr_lines = []
    for attr in attributes:
        desc = ATTRIBUTE_DESCRIPTIONS.get(attr, attr.replace("_", " "))
        attr_lines.append(f'  "{attr}": {desc}')

    attrs_block = "\n".join(attr_lines)

    return (
        f"Examine this garment image and determine whether each attribute is present.\n\n"
        f"Attributes to check:\n{attrs_block}\n\n"
        f"Respond with ONLY a JSON object with this exact structure:\n"
        f"{{\n"
        + "\n".join(f'  "{a}": true_or_false,  // true if clearly visible' for a in attributes)
        + f"\n}}\n\n"
        f"Rules:\n"
        f"- true = attribute is clearly visible in the image\n"
        f"- false = attribute is absent, not visible, or you are uncertain\n"
        f"- No explanations, no markdown, just the JSON object"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Claude API call
# ─────────────────────────────────────────────────────────────────────────────

def call_claude(
    client: anthropic.Anthropic,
    b64_image: str,
    media_type: str,
    attributes: list[str],
) -> dict[str, bool] | None:
    """
    Call Claude with the image and return a dict of {attribute: bool}.
    Returns None on unrecoverable failure.
    """
    user_prompt = build_user_prompt(attributes)

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    }
                ],
            )

            raw = response.content[0].text.strip()

            # Strip markdown code fences if Claude adds them despite instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            return {attr: bool(result.get(attr, False)) for attr in attributes}

        except anthropic.RateLimitError:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"\n  Rate limit — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print("\n  Rate limit — giving up on this image")
                return None

        except (json.JSONDecodeError, KeyError) as e:
            print(f"\n  Parse error: {e}  raw={raw[:100]!r}")
            return None

        except Exception as e:
            print(f"\n  API error: {e}")
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(
    labels_csv: Path,
    attributes: list[str],
    val_only: bool,
    max_images: int | None,
    seed: int = 42,
    val_split: float = 0.2,
) -> pd.DataFrame:
    """
    Load labels CSV and build per-attribute binary ground truth columns.
    The CSV has an 'attributes' column with underscore-joined labels
    (e.g. "belted_epaulette" or "" for no attributes).
    """
    df = pd.read_csv(labels_csv)
    df["attributes"] = df["attributes"].fillna("")

    # Build binary columns per attribute
    for attr in attributes:
        df[f"gt_{attr}"] = df["attributes"].apply(
            lambda s: int(attr in s.split("_"))
        )

    if val_only:
        _, df = train_test_split(df, test_size=val_split, random_state=seed)
        df = df.reset_index(drop=True)
        print(f"Using val split: {len(df)} images")
    else:
        print(f"Using full dataset: {len(df)} images")

    if max_images and len(df) > max_images:
        # Stratify sample to keep attribute balance
        df = df.sample(n=max_images, random_state=seed).reset_index(drop=True)
        print(f"Capped to {max_images} images (stratified sample)")

    # Report attribute distribution in selected set
    print("\nGround truth distribution in evaluation set:")
    for attr in attributes:
        n_pos = df[f"gt_{attr}"].sum()
        print(f"  {attr:<20} {n_pos:>3} positive / {len(df)} total  ({100*n_pos/len(df):.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    labels_csv: Path,
    image_dir: Path,
    attributes: list[str],
    max_images: int | None,
    val_only: bool,
    out_csv: Path | None,
) -> None:

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Load ground truth
    df = load_ground_truth(labels_csv, attributes, val_only, max_images)

    # Track predictions
    results = []
    missing = 0
    failed  = 0

    print(f"\nRunning Claude ({MODEL}) on {len(df)} images...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        stem = row["name"]
        img_path = find_image(image_dir, stem)

        if img_path is None:
            missing += 1
            tqdm.write(f"  MISSING: {stem}")
            continue

        try:
            b64, media_type = encode_image(img_path)
        except Exception as e:
            tqdm.write(f"  ENCODE ERROR {stem}: {e}")
            failed += 1
            continue

        preds = call_claude(client, b64, media_type, attributes)
        if preds is None:
            failed += 1
            continue

        record = {"name": stem}
        for attr in attributes:
            record[f"gt_{attr}"]   = int(row[f"gt_{attr}"])
            record[f"pred_{attr}"] = int(preds[attr])
        results.append(record)

    print(f"\n{'='*55}")
    print(f"Evaluated : {len(results)} images")
    print(f"Missing   : {missing}")
    print(f"Failed    : {failed}")
    print(f"{'='*55}\n")

    if not results:
        print("No results to report.")
        return

    results_df = pd.DataFrame(results)

    # ── Per-attribute metrics ──────────────────────────────────────────────
    print(f"{'Attribute':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8} {'Pred+':>6}  Signal")
    print("-" * 60)

    f1_scores = []
    for attr in attributes:
        gt   = results_df[f"gt_{attr}"].values
        pred = results_df[f"pred_{attr}"].values

        prec    = precision_score(gt, pred, zero_division=0)
        rec     = recall_score(gt, pred, zero_division=0)
        f1      = f1_score(gt, pred, zero_division=0)
        support = int(gt.sum())
        pred_pos = int(pred.sum())

        f1_scores.append(f1)

        if f1 >= 0.80:
            signal = "✓ ships"
        elif f1 >= 0.65:
            signal = "~ fine-tune"
        else:
            signal = "✗ needs data"

        print(f"  {attr:<18} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {support:>8} {pred_pos:>6}  {signal}")

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n  {'Macro F1':<18} {macro_f1:>6.3f}")

    # ── Detailed sklearn report ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Detailed classification report:")
    print(f"{'='*55}")
    for attr in attributes:
        gt   = results_df[f"gt_{attr}"].values
        pred = results_df[f"pred_{attr}"].values
        print(f"\n  [{attr}]")
        print(classification_report(gt, pred, target_names=["absent", "present"], digits=3))

    # ── Error analysis ─────────────────────────────────────────────────────
    print(f"{'='*55}")
    print("Error breakdown:")
    for attr in attributes:
        gt   = results_df[f"gt_{attr}"].values
        pred = results_df[f"pred_{attr}"].values
        fp = int(((pred == 1) & (gt == 0)).sum())   # false positives (overcalling)
        fn = int(((pred == 0) & (gt == 1)).sum())   # false negatives (missing)
        tp = int(((pred == 1) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        print(f"  {attr:<20}  TP={tp}  TN={tn}  FP={fp} (overcall)  FN={fn} (miss)")

    # ── Save results ───────────────────────────────────────────────────────
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Evaluate Claude VLM on fashion attribute recognition"
    )
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--image-dir",   type=Path, required=True)
    p.add_argument("--attributes",  nargs="+", default=["epaulette", "belted"],
                   help="Attribute names to evaluate (must match labels CSV)")
    p.add_argument("--max-images",  type=int, default=50,
                   help="Max images to evaluate (default: 50 for quick test)")
    p.add_argument("--val-only",    action="store_true",
                   help="Use same 80/20 val split as train.py (seed=42)")
    p.add_argument("--out-csv",     type=Path, default=None,
                   help="Optional: save per-image predictions to CSV")
    args = p.parse_args()

    run_evaluation(
        labels_csv  = args.labels_csv,
        image_dir   = args.image_dir,
        attributes  = args.attributes,
        max_images  = args.max_images,
        val_only    = args.val_only,
        out_csv     = args.out_csv,
    )


if __name__ == "__main__":
    main()
