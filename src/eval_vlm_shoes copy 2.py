"""
eval_vlm_shoes.py
=================
Evaluates Claude VLM on shoe attribute recognition against ground truth labels.

Supports two distinct evaluation modes, selectable via --mode:

  toe-heel      Categorical mode — Claude returns exactly ONE toe shape and
                ONE heel type per image. Ground truth CSV must have separate
                columns: toe_shape and heel_type (see Label Studio config below).
                Metrics: per-class accuracy + macro F1 for each group.

  construction  Multi-label mode — Claude independently true/false each
                construction attribute (strappy, ankle-strap, platform, etc.).
                Ground truth CSV has an 'attributes' column with pipe-separated
                values (e.g. "ankle-strap|platform"). Metrics: per-attribute F1.

CSV formats
-----------
  toe-heel mode:
      name, toe_shape, heel_type
      shoe_001, pointed-toe, stiletto-heel
      shoe_002, round-toe, flat-heel

  construction mode:
      name, attributes
      shoe_001, ankle-strap|platform
      shoe_002, strappy
      shoe_003,                        ← empty = no construction attributes

  Note: pipe separator (|) is used instead of underscore (_) to avoid
  conflicts with hyphenated attribute names like "ankle-strap".

Requirements
------------
    pip install anthropic pillow scikit-learn pandas tqdm

Setup
-----
    $env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell

Usage
-----
    # Toe + heel evaluation
    python src/eval_vlm_shoes.py \\
        --mode toe-heel \\
        --labels-csv "E:/fashion-data/csv/shoes_toe_heel.csv" \\
        --image-dir  "E:/fashion-data/01-RAW/shoes" \\
        --max-images 150

    # Construction attributes evaluation
    python src/eval_vlm_shoes.py \\
        --mode construction \\
        --labels-csv "E:/fashion-data/csv/shoes_construction.csv" \\
        --image-dir  "E:/fashion-data/01-RAW/shoes" \\
        --attributes strappy ankle-strap sling-back platform lace-up zip-up \\
        --max-images 150

    # Save per-image predictions to CSV
    python src/eval_vlm_shoes.py \\
        --mode toe-heel \\
        --labels-csv "E:/fashion-data/csv/shoes_toe_heel.csv" \\
        --image-dir  "E:/fashion-data/01-RAW/shoes" \\
        --out-csv    "E:/fashion-data/csv/shoes_toeheel_results.csv"
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL          = "claude-sonnet-4-5"
MAX_IMAGE_PX   = 1568
JPEG_QUALITY   = 85
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5.0
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# Toe + heel definitions
# ─────────────────────────────────────────────────────────────────────────────

TOE_SHAPES = [
    "apron-toe",
    "bicycle-toe",
    "cap-toe",
    "peep-toe",
    "plain-toe",
    "pointed-toe",
    "round-toe",
    "split-toe",
    "square-toe",
    "wing-tip",
]

HEEL_TYPES = [
    "boulevard-heel",
    "cone-heel",
    "flat-heel",
    "french-heel",
    "kitten-heel",
    "low-heel",
    "mid-heel",
    "high-heel",
    "square-heel",
    "stiletto-heel",
    "wedge-heel",
]

TOE_DESCRIPTIONS = {
    "apron-toe":    "A smooth, rounded toe with a curved seam stitched around the front, creating an apron-like panel. Common on loafers and moccasins.",
    "bicycle-toe":  "An elongated, slightly squared toe with a subtle seam running around the tip. Often seen on dress shoes and oxfords.",
    "cap-toe":      "A distinct horizontal seam or overlay across the toe box, creating a separate 'cap'. Classic on oxfords and brogues.",
    "open-toe": "The front of the shoe is fully open with no toe box, leaving all toes completely exposed. Common in sandals, mules and slides.",
    "peep-toe":     "An opening cut into the front of the shoe revealing the toes, typically the big toe and one or two others.",
    "plain-toe":    "A completely smooth, undecorated toe box with no seams, perforations, or overlays. Clean and minimal.",
    "pointed-toe":  "A toe that tapers to a sharp, narrow point extending well beyond the foot. The tip is noticeably elongated.",
    "round-toe":    "A gently curved, rounded toe box that follows the natural shape of the foot. Neither pointed nor squared.",
    "split-toe":    "The toe area is divided by a vertical seam or cut running down the centre, creating two distinct sections.",
    "square-toe":   "The toe box ends in a flat, straight horizontal edge creating a geometric square or rectangular front.",
    "wing-tip":     "A W-shaped or M-shaped decorative cap that extends from the centre of the toe along both sides, often with broguing.",
}

HEEL_DESCRIPTIONS = {
    "boulevard-heel": "A broad, slightly curved heel that is wider at the base than a standard heel. Elegant and stable, wider than a stiletto but narrower than a block heel.",
    "cone-heel":      "A heel that is wide at the top where it meets the shoe and tapers to a narrow circular point at the base, like an inverted cone.",
    "flared-heel":    "A heel that is wide where it meets the shoe's sole, tapers inward as it rises, then widens again where it attaches to the upper. The silhouette forms a distinctive hourglass or concave curve when viewed from the side. May be solid or have an open cutout. Distinct from a wedge (which is triangular and solid) and a cone heel (which tapers uniformly without flaring).",
    "flat-heel":      "No heel elevation — the sole is level from heel to toe, or nearly so (under 1cm). Completely flat.",
    "french-heel":    "A slender, curved heel that flares slightly outward at the base. Elegant, curving inward then out. Similar to a louis heel.",
    "kitten-heel":    "A very short, slim heel (1–4cm) that is delicate and tapered. Shorter than a high heel, taller than flat.",
    "low-heel":       "A modest heel height (2–4cm), noticeably elevated but not high. Includes low block heels and low stacked heels.",
    "mid-heel":       "A medium heel height (4–7cm). Clearly elevated but not dramatic. Includes mid block heels and mid stacked heels.",
    "high-heel":      "A tall heel (7cm+). Significantly elevates the heel. Includes high block heels and high stacked heels.",
    "square-heel":    "A heel with a flat, square cross-section — four flat sides forming a rectangular or square block shape.",
    "stiletto-heel":  "An extremely thin, spike-like heel (under 1cm wide) regardless of height. The heel is a narrow pin or rod.",
    "wedge-heel":     "A solid, continuous wedge of material forming both the heel and sole. No gap between heel and sole. Heel and sole are one piece.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Construction attribute definitions
# ─────────────────────────────────────────────────────────────────────────────

CONSTRUCTION_DESCRIPTIONS = {
    "strappy": (
        "THREE OR MORE thin straps crossing the foot or ankle. "
        "Cage-like or multi-band construction. NOT strappy if only 1 or 2 straps."
    ),
    "ankle-strap": (
        "A strap forming a COMPLETE LOOP around the ankle circumference, "
        "typically with a buckle or elastic. Full encirclement is required — "
        "a strap that merely reaches ankle height without wrapping around does NOT qualify."
    ),
    "ankle-wrap": (
        "Laces, ribbons, or ties that wrap multiple times around the ankle and lower leg, "
        "tied in a knot or bow. Distinct from a single buckled ankle-strap."
    ),
    "sling-back": (
        "A single strap crossing ONLY the back of the heel. "
        "The front of the foot is enclosed by the shoe body, not by straps. "
        "No strap crosses the instep or toes."
    ),
    "t-bar": (
        "A T-shaped strap construction: one vertical strap running from the toe toward "
        "the ankle meets one horizontal strap crossing the instep, forming a T or Y shape."
    ),
    "cross-over": (
        "Two or more straps that cross over each other diagonally across the foot or ankle, "
        "creating an X or criss-cross pattern."
    ),
    "platform": (
        "A thick, raised sole under the toe and ball of the foot (typically 2cm+), "
        "creating visible elevation at the front of the shoe as well as the heel."
    ),
    "lace-up": (
        "Multiple eyelets, hooks, or loops with a lace or cord threaded through them "
        "to close the shoe. Includes oxford lacing, derby lacing, and boot lacing."
    ),
    "zip-up": (
        "A visible zipper used as the primary or secondary closure. "
        "Can be on the side, back, or front of the shoe or boot."
    ),
    "slouch": (
        "A boot with intentionally excess, bunched-up or collapsed shaft material "
        "that falls loosely around the ankle or lower leg rather than standing upright."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TOE_HEEL = (
    "You are a footwear attribute recognition system. "
    "You classify shoe images by toe shape and heel type. "
    "Each shoe has exactly one toe shape and exactly one heel type — "
    "choose the single best match from the provided lists. "
    "Be precise. Only select what is clearly visible. "
    "Always respond with valid JSON only, no other text."
)

SYSTEM_PROMPT_CONSTRUCTION = (
    "You are a footwear attribute recognition system. "
    "You detect construction and closure attributes in shoe images. "
    "Multiple attributes can be present simultaneously. "
    "Only mark an attribute as present if it is clearly visible. "
    "Do not infer — if you cannot see it, mark it false. "
    "Always respond with valid JSON only, no other text."
)


def build_toe_heel_prompt() -> str:
    toe_block = "\n".join(
        f'    "{t}": {TOE_DESCRIPTIONS[t]}' for t in TOE_SHAPES
    )
    heel_block = "\n".join(
        f'    "{h}": {HEEL_DESCRIPTIONS[h]}' for h in HEEL_TYPES
    )
    toe_list  = ", ".join(f'"{t}"' for t in TOE_SHAPES)
    heel_list = ", ".join(f'"{h}"' for h in HEEL_TYPES)

    return (
        f"Examine this shoe image and classify it.\n\n"
        f"TOE SHAPE options:\n{toe_block}\n\n"
        f"HEEL TYPE options:\n{heel_block}\n\n"
        f"Rules:\n"
        f"- Choose EXACTLY ONE toe shape from: [{toe_list}]\n"
        f"- Choose EXACTLY ONE heel type from: [{heel_list}]\n"
        f"- If the toe is partially hidden, choose the closest visible match\n"
        f"- If unsure between two heel heights, choose the shorter one\n\n"
        f"Respond with ONLY this JSON structure:\n"
        f'{{\n  "toe_shape": "<value>",\n  "heel_type": "<value>"\n}}'
    )


def build_construction_prompt(attributes: list[str]) -> str:
    attr_lines = "\n".join(
        f'  "{a}": {CONSTRUCTION_DESCRIPTIONS.get(a, a.replace("-", " "))}'
        for a in attributes
    )
    json_template = "\n".join(
        f'  "{a}": true_or_false' for a in attributes
    )
    return (
        f"Examine this shoe image and detect which construction attributes are present.\n\n"
        f"Attributes:\n{attr_lines}\n\n"
        f"Rules:\n"
        f"- true = attribute is clearly visible\n"
        f"- false = absent, not visible, or uncertain\n"
        f"- Multiple attributes can be true simultaneously\n"
        f"- No explanations, no markdown, just the JSON\n\n"
        f"Respond with ONLY:\n{{\n{json_template}\n}}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Claude API calls
# ─────────────────────────────────────────────────────────────────────────────

def _call(
    client: anthropic.Anthropic,
    b64: str,
    media_type: str,
    system: str,
    user: str,
) -> dict | None:
    """Single Claude API call with retry. Returns parsed dict or None."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
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
            return json.loads(raw.strip())

        except anthropic.RateLimitError:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"\n  Rate limit — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print("\n  Rate limit — giving up")
                return None
        except json.JSONDecodeError as e:
            print(f"\n  Parse error: {e}")
            return None
        except Exception as e:
            print(f"\n  API error: {e}")
            return None
    return None


def call_toe_heel(
    client: anthropic.Anthropic, b64: str, media_type: str
) -> dict[str, str] | None:
    result = _call(client, b64, media_type, SYSTEM_PROMPT_TOE_HEEL, build_toe_heel_prompt())
    if result is None:
        return None
    toe  = result.get("toe_shape", "").strip()
    heel = result.get("heel_type", "").strip()
    # Validate against known values
    if toe not in TOE_SHAPES:
        print(f"\n  Invalid toe_shape: {toe!r}")
        toe = "plain-toe"   # safe fallback
    if heel not in HEEL_TYPES:
        print(f"\n  Invalid heel_type: {heel!r}")
        heel = "flat-heel"  # safe fallback
    return {"toe_shape": toe, "heel_type": heel}


def call_construction(
    client: anthropic.Anthropic,
    b64: str,
    media_type: str,
    attributes: list[str],
) -> dict[str, bool] | None:
    result = _call(
        client, b64, media_type,
        SYSTEM_PROMPT_CONSTRUCTION,
        build_construction_prompt(attributes),
    )
    if result is None:
        return None
    return {attr: bool(result.get(attr, False)) for attr in attributes}


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_toe_heel(
    labels_csv: Path,
    val_only: bool,
    max_images: int | None,
    seed: int = 42,
    val_split: float = 0.2,
) -> pd.DataFrame:
    """
    Expects CSV columns: name, toe_shape, heel_type
    """
    df = pd.read_csv(labels_csv)
    required = {"name", "toe_shape", "heel_type"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}\n"
            f"toe-heel mode expects: name, toe_shape, heel_type"
        )

    if val_only:
        _, df = train_test_split(df, test_size=val_split, random_state=seed)
        df = df.reset_index(drop=True)
        print(f"Using val split: {len(df)} images")
    else:
        print(f"Using full dataset: {len(df)} images")

    if max_images and len(df) > max_images:
        df = df.sample(n=max_images, random_state=seed).reset_index(drop=True)
        print(f"Capped to {max_images} images")

    print("\nGround truth distribution:")
    print("  Toe shapes:")
    for v, c in df["toe_shape"].value_counts().items():
        print(f"    {v:<20} {c:>4}")
    print("  Heel types:")
    for v, c in df["heel_type"].value_counts().items():
        print(f"    {v:<20} {c:>4}")

    return df


def load_gt_construction(
    labels_csv: Path,
    attributes: list[str],
    val_only: bool,
    max_images: int | None,
    seed: int = 42,
    val_split: float = 0.2,
) -> pd.DataFrame:
    """
    Expects CSV columns: name, attributes (pipe-separated, e.g. "ankle-strap|platform")
    Pipe separator avoids collision with hyphens in attribute names.
    """
    df = pd.read_csv(labels_csv)
    df["attributes"] = df["attributes"].fillna("")

    # Binary column per attribute — split on pipe, not underscore
    for attr in attributes:
        df[f"gt_{attr}"] = df["attributes"].apply(
            lambda s: int(attr in [x.strip() for x in s.split("|")])
        )

    if val_only:
        _, df = train_test_split(df, test_size=val_split, random_state=seed)
        df = df.reset_index(drop=True)
        print(f"Using val split: {len(df)} images")
    else:
        print(f"Using full dataset: {len(df)} images")

    if max_images and len(df) > max_images:
        df = df.sample(n=max_images, random_state=seed).reset_index(drop=True)
        print(f"Capped to {max_images} images")

    print("\nGround truth distribution in evaluation set:")
    for attr in attributes:
        n_pos = df[f"gt_{attr}"].sum()
        print(f"  {attr:<22} {n_pos:>3} positive / {len(df)} total  ({100*n_pos/len(df):.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metrics reporting
# ─────────────────────────────────────────────────────────────────────────────

def report_toe_heel(results_df: pd.DataFrame) -> None:
    for group, col in [("TOE SHAPE", "toe_shape"), ("HEEL TYPE", "heel_type")]:
        gt   = results_df[f"gt_{col}"].values
        pred = results_df[f"pred_{col}"].values
        acc  = accuracy_score(gt, pred)
        mf1  = f1_score(gt, pred, average="macro", zero_division=0)

        print(f"\n{'='*55}")
        print(f"{group}  —  accuracy={acc:.3f}  macro_F1={mf1:.3f}")
        print(f"{'='*55}")

        labels = sorted(set(gt) | set(pred))
        print(classification_report(gt, pred, labels=labels, digits=3, zero_division=0))

        # Confusion summary: most common errors
        errors = results_df[results_df[f"gt_{col}"] != results_df[f"pred_{col}"]]
        if not errors.empty:
            print(f"  Top confusions (true → predicted):")
            conf = (
                errors.groupby([f"gt_{col}", f"pred_{col}"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(8)
            )
            for _, r in conf.iterrows():
                print(f"    {r[f'gt_{col}']:<22} → {r[f'pred_{col}']:<22} ({r['count']}x)")


def report_construction(results_df: pd.DataFrame, attributes: list[str]) -> None:
    print(f"\n{'='*55}")
    print("CONSTRUCTION ATTRIBUTES")
    print(f"{'='*55}")
    print(f"{'Attribute':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8} {'Pred+':>6}  Signal")
    print("-" * 65)

    f1_scores = []
    for attr in attributes:
        gt   = results_df[f"gt_{attr}"].values
        pred = results_df[f"pred_{attr}"].values
        prec = precision_score(gt, pred, zero_division=0)
        rec  = recall_score(gt, pred, zero_division=0)
        f1   = f1_score(gt, pred, zero_division=0)
        support  = int(gt.sum())
        pred_pos = int(pred.sum())
        f1_scores.append(f1)

        signal = "✓ ships" if f1 >= 0.80 else ("~ fine-tune" if f1 >= 0.65 else "✗ needs work")
        print(f"  {attr:<20} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {support:>8} {pred_pos:>6}  {signal}")

    print(f"\n  {'Macro F1':<20} {sum(f1_scores)/len(f1_scores):>6.3f}")

    print(f"\n{'='*55}")
    print("Error breakdown:")
    for attr in attributes:
        gt   = results_df[f"gt_{attr}"].values
        pred = results_df[f"pred_{attr}"].values
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tp = int(((pred == 1) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        print(f"  {attr:<22}  TP={tp}  TN={tn}  FP={fp} (overcall)  FN={fn} (miss)")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loops
# ─────────────────────────────────────────────────────────────────────────────

def run_toe_heel(
    client: anthropic.Anthropic,
    labels_csv: Path,
    image_dir: Path,
    max_images: int | None,
    val_only: bool,
    out_csv: Path | None,
) -> None:
    df = load_gt_toe_heel(labels_csv, val_only, max_images)

    results = []
    missing = failed = 0

    print(f"\nRunning Claude ({MODEL}) — toe-heel mode on {len(df)} images...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        stem     = row["name"]
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

        preds = call_toe_heel(client, b64, media_type)
        if preds is None:
            failed += 1
            continue

        results.append({
            "name":          stem,
            "gt_toe_shape":  row["toe_shape"],
            "pred_toe_shape": preds["toe_shape"],
            "gt_heel_type":  row["heel_type"],
            "pred_heel_type": preds["heel_type"],
        })

    print(f"\nEvaluated: {len(results)}  Missing: {missing}  Failed: {failed}")

    if not results:
        print("No results.")
        return

    results_df = pd.DataFrame(results)
    report_toe_heel(results_df)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")


def run_construction(
    client: anthropic.Anthropic,
    labels_csv: Path,
    image_dir: Path,
    attributes: list[str],
    max_images: int | None,
    val_only: bool,
    out_csv: Path | None,
) -> None:
    df = load_gt_construction(labels_csv, attributes, val_only, max_images)

    results = []
    missing = failed = 0

    print(f"\nRunning Claude ({MODEL}) — construction mode on {len(df)} images...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        stem     = row["name"]
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

        preds = call_construction(client, b64, media_type, attributes)
        if preds is None:
            failed += 1
            continue

        record = {"name": stem}
        for attr in attributes:
            record[f"gt_{attr}"]   = int(row[f"gt_{attr}"])
            record[f"pred_{attr}"] = int(preds[attr])
        results.append(record)

    print(f"\nEvaluated: {len(results)}  Missing: {missing}  Failed: {failed}")

    if not results:
        print("No results.")
        return

    results_df = pd.DataFrame(results)
    report_construction(results_df, attributes)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONSTRUCTION_ATTRS = [
    "strappy", "ankle-strap", "ankle-wrap", "sling-back",
    "t-bar", "cross-over", "platform", "lace-up", "zip-up", "slouch",
]


def main():
    p = argparse.ArgumentParser(
        description="Evaluate Claude VLM on shoe attribute recognition"
    )
    p.add_argument("--mode", choices=["toe-heel", "construction"],
                   required=True,
                   help="toe-heel: categorical toe+heel classification; "
                        "construction: multi-label strap/closure attributes")
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--image-dir",   type=Path, required=True)
    p.add_argument("--attributes",  nargs="+",
                   default=DEFAULT_CONSTRUCTION_ATTRS,
                   help="Construction attributes to evaluate (construction mode only)")
    p.add_argument("--max-images",  type=int, default=None)
    p.add_argument("--val-only",    action="store_true")
    p.add_argument("--out-csv",     type=Path, default=None)
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)"
        )
    client = anthropic.Anthropic(api_key=api_key)

    if args.mode == "toe-heel":
        run_toe_heel(client, args.labels_csv, args.image_dir,
                     args.max_images, args.val_only, args.out_csv)
    else:
        run_construction(client, args.labels_csv, args.image_dir,
                         args.attributes, args.max_images, args.val_only, args.out_csv)


if __name__ == "__main__":
    main()
