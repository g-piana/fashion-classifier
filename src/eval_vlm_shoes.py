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
    "open-toe",
    "pointed-toe",
    "round-toe",
    "square-toe",
]

HEEL_TYPES = [
    "comma-heel",
    "cone-heel",
    "flat-heel",
    "french-heel",
    "kitten-heel",
    "low-heel",
    "mid-heel",
    "high-heel",
    "stiletto-heel",
    "wedge-heel",
]
#    "apron-toe":    "A smooth, rounded toe with a curved seam stitched around the front, creating an apron-like panel. Common on loafers and moccasins.",
#    "cap-toe":      "A distinct horizontal seam or overlay across the toe box, creating a separate 'cap'. Classic on oxfords and brogues.",
# "peep-toe": (
#     "A closed toe box with a small circular or oval cutout or opening at the very front, "
#     "allowing just the tips of one or two toes to peek through. "
#     "The shoe otherwise has a complete closed upper. "
#     "NOT open-toe: open-toe has NO toe box at all — the entire front is open. "
#     "The key distinction: peep-toe has a hole IN a closed box; open-toe has no box."
# ),
# "bicycle-toe":  "An elongated, slightly squared toe with a subtle seam running around the tip. Often seen on dress shoes and oxfords.",
# "plain-toe":    "A completely smooth, undecorated toe box with no seams, perforations, or overlays. Clean and minimal.",
# "wing-tip":     "A W-shaped or M-shaped decorative cap that extends from the centre of the toe along both sides, often with broguing.",
# "split-toe":    "The toe area is divided by a vertical seam or cut running down the centre, creating two distinct sections.",


TOE_DESCRIPTIONS = {
    "open-toe": (
        "The shoe has NO closed toe box — the front of the shoe is completely open "
        "and all toes are fully exposed. This is a construction feature, not a toe shape. "
        "Sandals, slides, and mules are typically open-toe. "
        "NOT round/square/pointed: those describe the shape of a CLOSED toe box. "
        "If there is no toe box at all, it is open-toe regardless of the foot shape visible."
    ),

        # "NOT a peep-toe: peep-toe has a small hole or cutout in an otherwise closed toe box — "
        # "only the tip of one or two toes peeks through. "

    "pointed-toe": (
        "The toe box narrows to a visibly tapered or pointed tip — the sides converge "
        "toward a single point or very narrow end. The degree of point can vary from "
        "aggressively sharp to mildly tapered, but there is always a clear convergence. "
        "NOT round-toe: round toes end in a smooth curve with no convergence. "
        "If the tip is narrower than the widest part of the toe box, it is pointed-toe."
    ),
    "round-toe": (
        "The toe box has a smoothly curved, semicircular tip with no angles or points. "
        "The curvature is symmetric and gentle — like the end of an oval. "
        "NOT pointed-toe: if the tip narrows to any discernible point or taper, it is pointed-toe. "
        "NOT square-toe: if the tip has a flat horizontal edge, it is square-toe. "
        "When in doubt between round and pointed, look at the very tip — "
        "a round toe ends in a curve, a pointed toe ends in a convergence."
    ),
    "square-toe":   "The toe box ends in a flat, straight horizontal edge creating a geometric square or rectangular front.",

}

HEEL_DESCRIPTIONS = {

    "cone-heel": (
        "A heel shaped like a truncated cone: narrow at the top where it meets the shoe "
        "upper, and wider at the base where it contacts the ground. The shaft tapers "
        "outward from top to bottom — the opposite of a stiletto. "
        "Viewed from the side, the outline of the heel is trapezoidal: narrow at top, "
        "flaring out toward the sole, ending in a flat base. "
        "The taper is smooth and continuous with no concave curves. "
        "NOT a stiletto: stilettos are thin throughout and taper to a point, not a wide flat base. "
        "NOT a mid-heel: mid-heels have a cylindrical shaft of uniform width, no outward flare. "
        "NOT a kitten-heel: kitten heels are short and straight with no flare. "
        "The defining feature is the outward flare from top to base."
    ),
    "comma-heel": (
        "A heel that is wide and substantial at the top where it attaches under the shoe body, "
        "with the back face curving outward as it descends, "
        "then tapering to a fine narrow point at the ground contact — similar to a stiletto tip. "
        "Viewed from the side, the back face of the heel forms a visible convex curve outward, "
        "giving the overall silhouette the shape of a comma or teardrop. "
        "The defining feature is the wide flared top that curves and narrows toward the ground. "
        "Height is typically mid to high (6-10cm). "
        "CRITICAL DISTINCTION from stiletto: a stiletto runs straight and thin throughout "
        "from top to ground — the comma-heel is distinctly wider at the top with a curved back face. "
        "CRITICAL DISTINCTION from kitten heel: a kitten heel is short (under 5cm) and straight "
        "with no outward curve on the back face."
    ),
    "flat-heel":      "No heel elevation — the sole is level from heel to toe, or nearly so (under 1cm). Completely flat.",
    "french-heel":    "A slender, curved heel that flares slightly outward at the base. it has have a more pronounced S-curve with respect to boulevard-heel. Elegant, curving inward then out. Similar to a louis heel.",
    "kitten-heel": (
        "A short, slender heel typically under 5cm (2 inches) in height. "
        "The heel shaft runs straight or with only the most minimal taper — "
        "there is NO visible inward curve or concave profile on the front face. "
        "The side silhouette is essentially a thin straight column. "
        "CRITICAL DISTINCTION: if the heel curves inward at any point along its shaft "
        "— even subtly — it is NOT a kitten heel. Kitten heels are defined by their "
        "shortness AND their straight profile together. A curved heel of the same "
        "height is a comma or spindle heel, not a kitten heel."
    ),
    "low-heel": (
        "A heel with minimal elevation, roughly 1-4cm high. The shaft is too short "
        "to have a recognizable shape — it appears as a small block or lift. "
        "NOT a flat-heel: there is a distinct, separate heel unit visible. "
        "NOT a kitten-heel: kitten heels have a visibly thin, slender shaft. "
        "When in doubt between low-heel and flat-heel, check if the heel unit is "
        "clearly distinct from the sole — if yes, it is low-heel."
    ),
    "mid-heel": (
    "A heel of medium elevation, roughly 4-7cm high, with a broad, block-like "
    "or chunky cylindrical shaft that has no distinctive shape — no taper, "
    "no curve, no flare. The shaft is simply a sturdy column of moderate height. "
    "NOT a cone-heel: cone heels flare outward from top to base — narrow top, wide base. "
    "NOT a comma-heel: comma heels have a visible concave curve on the front face. "
    "NOT a square-heel: square heels have a distinctly squared cross-section. "
    "When the heel is medium height and the shaft is a plain block with no "
    "recognizable geometry, it is mid-heel."
    ),        
    "high-heel": (
        "A heel that is elevated (roughly 7cm or more) with a shaft that is "
        "visibly broader than a stiletto — block-shaped, chunky, or tapered but thick. "
        "NOT a stiletto: stiletto heels have an extremely thin needle-like shaft. "
        "NOT a mid-heel: mid-heels are the same shape but shorter."
    ),
    "stiletto-heel": (
        "A heel with an extremely thin, needle-like shaft — the cross-section "
        "at mid-shaft is visibly narrow, typically under 1cm diameter. "
        "Height is usually high but the defining feature is shaft thinness, not height. "
        "NOT a high-heel: if the shaft is broad or block-shaped, even at the same height."
    ),    
    "wedge-heel":     "A solid, continuous wedge of material forming both the heel and sole. No gap between heel and sole. Heel and sole are one piece.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Construction attribute definitions
# ─────────────────────────────────────────────────────────────────────────────

CONSTRUCTION_DESCRIPTIONS = {
    "strappy": (
        "THREE OR MORE thin straps across the foot or ankle. "
        "The straps are the primary structural element of the upper — "
        "the shoe is mostly straps rather than a solid upper. "
        "NOT ankle-strap: a single strap around the ankle does not make a shoe strappy. "
        "NOT cross-over: two straps crossing is not strappy. "
        "Count the straps — if fewer than 3, it is not strappy."
        "NOT lace-up: shoes with laces or a cage/cutout upper are not strappy. " 
        "NOT ankle-wrap: if the straps wrap around the ankle as a wrapping mechanism, label ankle-wrap instead of strappy — unless there are also 3+ straps on the vamp itself."
    ),
    "ankle-strap": (
        "A single dedicated strap that forms a COMPLETE LOOP around the ankle, "
        "fastened with a buckle or clasp visible on the side or back of the ankle. "
        "The strap must be a separate, distinct element from the main upper. "
        "NOT strappy: if the shoe has 3+ straps, the topmost strap near the ankle "
        "does not additionally qualify as ankle-strap unless it has its own dedicated buckle "
        "forming a complete independent loop. "
        "NOT a sandal with straps near the ankle: proximity to the ankle is not sufficient — "
        "the strap must encircle the ankle completely with a fastening."
    ),
    "ankle-wrap": (
        "Laces, ribbons, or ties that wrap multiple times around the ankle and lower leg, "
        "tied in a knot or bow. Distinct from a single buckled ankle-strap."
    ),
    "sling-back": (
        "A single strap that goes across the back of the heel only, "
        "holding the shoe on from behind. There is NO strap over the instep or ankle. "
        "The front of the shoe has a closed or open toe box with no ankle coverage. "
        "NOT ankle-strap: ankle-strap wraps the full ankle circumference. "
        "The sling-back strap sits at the very back of the heel, not around the ankle."
    ),
    "t-bar": (
        "A T-shaped strap construction: one vertical strap running from the toe toward "
        "the ankle meets one horizontal strap crossing the instep, forming a T or Y shape."
    ),
    "cross-over": (
        "Two straps that cross each other diagonally over the instep, "
        "forming a visible X or V shape. The crossing must be clearly visible. "
        "Can co-occur with ankle-strap if there is also a buckled ankle band. "
        "NOT strappy: if there are 3 or more straps total, classify as strappy."
    ),
    "platform": (
        "A visibly raised, thick sole section under the forefoot — "
        "at least 2cm of sole material elevating the toe area off the ground. "
        "Platform is independent of heel type: a wedge can have a platform, "
        "a stiletto can have a platform, a block heel can have a platform. "
        "A wedge with a thick forefoot sole (2cm+) IS a platform even if "
        "the wedge and platform form a continuous sole unit. "
        "NOT a platform: a wedge where the forefoot sole tapers to thin at the toe. "
        "NOT a platform: sneakers, trainers, or flat casual shoes where the "
        "thick rubber sole is athletic/structural rather than a fashion platform."
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
        f"- open-toe means NO closed toe box at all — sandals, slides, mules\n"
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
        toe = "_invalid"   # visible fallback — do not mask bad outputs
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
    image_dir: Path | None = None,
    seed: int = 42,
    val_split: float = 0.2,
) -> pd.DataFrame:
    """
    Expects CSV columns: name, toe_shape, heel_type.
    If image_dir is provided, rows are pre-filtered to stems that
    actually exist there — enables targeted eval on a subfolder.
    """
    df = pd.read_csv(labels_csv)
    required = {"name", "toe_shape", "heel_type"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}\n"
            f"toe-heel mode expects: name, toe_shape, heel_type"
        )

    if image_dir is not None:
        mask = df["name"].apply(lambda stem: find_image(image_dir, stem) is not None)
        n_before = len(df)
        df = df[mask].reset_index(drop=True)
        print(f"Filtered to images present in {image_dir.name}/: {len(df)} / {n_before}")
        if len(df) == 0:
            raise ValueError(
                f"No CSV rows matched any image in {image_dir}.\n"
                f"Check that filenames match the 'name' column in the CSV."
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
    # Support both "attributes" and "construction" as the column name
    col = "attributes" if "attributes" in df.columns else "construction"
    df[col] = df[col].fillna("")

    # Binary column per attribute — split on pipe, not underscore
    for attr in attributes:
        df[f"gt_{attr}"] = df[col].apply(
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
    df = load_gt_toe_heel(labels_csv, val_only, max_images, image_dir=image_dir)

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


# ─────────────────────────────────────────────────────────────────────────────
# Embellishment mode
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_EMBELLISHMENT = (
    "You are a footwear embellishment recognition system. "
    "You detect decorative and ornamental details applied to shoes. "
    "Multiple embellishments can be present simultaneously. "
    "Only mark an embellishment as present if it is clearly visible as a "
    "deliberate decorative element. "
    "Do not infer — if you cannot see it clearly, mark it false. "
    "Always respond with valid JSON only, no other text."
)

EMBELLISHMENT_DESCRIPTIONS: dict[str, str] = {
    "bead": (
        "Small decorative beads applied to the upper as ornamental detail. "
        "Must be clearly individual beaded elements, not rhinestones or crystals."
        "NOT crystals, NOT pearls, NOT rhinestones — beads are opaque, typically ceramic, glass or plastic, strung or individually glued"
    ),
    "bow": (
        "A bow or bow-tie shaped decorative element on the shoe. "
        "Can be fabric, ribbon, leather or synthetic. "
        "Must be a recognisable bow shape — two loops with a knot or centre."
         "Bow includes structured geometric toe ornaments in the bow-plaque family, not just classic ribbon bows"
    ),
    "buckle": (
        "A buckle present purely as a decorative design statement — "
        "oversized, ornamental, or placed where it serves no functional closure purpose. "
        "Examples: a large statement buckle on the vamp of a loafer, "
        "a buckle on a non-opening position, or a buckle that is clearly "
        "the focal design element of the shoe. "
        "NOT any buckle that fastens an ankle strap, sling-back or any other closure — "
        "these are functional regardless of size. "
        "If the buckle's primary purpose is to close or adjust the shoe, it is NOT this embellishment. "
        "When in doubt, do not mark buckle."
    ),
    "chain": (
        "Metal or decorative chain used as a trim, strap or ornamental detail. "
        "Includes chain-link trim along edges, chain ankle straps, or chain tassels."
    ),
    "crystal": (
        "Faceted decorative stones including rhinestones, crystals, diamantés or gemstones "
        "applied as embellishment. Includes single accent stones and full crystal-covered uppers. "
        "Any sparkling faceted stone counts regardless of whether it is genuine or synthetic."
    ),
    "embroidery": (
        "Decorative stitching forming patterns, motifs or designs on the upper. "
        "Must be visible needlework — thread patterns stitched into the material. "
        "NOT a woven or printed pattern."
    ),
    "eyelet": (
        "Visible metal-rimmed holes on the shoe upper, either for lacing or purely decorative. "
        "The metal rim must be clearly visible — a distinct circular metal ring around the hole. "
        "Common on boots and sneakers as lace holes when the metal eyelet ring is prominent. "
        "Also includes decorative punched eyelets used as surface ornamentation. "
        "NOT sequins: sequins are solid flat discs with no hole through the shoe material. "
        "NOT perforations: small punched holes in leather with no metal rim are not eyelets. "
        "NOT fur texture or fabric weave patterns. "
        "The key feature is a METAL RIM around a HOLE — if you cannot see a distinct "
        "metal ring, it is not an eyelet."
    ),
    "feather": (
        "Real or synthetic feathers used as trim or decoration. "
        "Typically found at the toe, ankle or heel area. "
        "Must be clearly feathers — fluffy, quill-like or plume-shaped elements."
    ),
    "flower": (
        "Floral appliqué, fabric flowers, 3D flower-shaped embellishments, "
        "or embroidered floral motifs where flowers are a clearly recognisable subject. "
        "Flower and embroidery should both be marked when the embroidery depicts flowers. "
        "Physical raised or appliqué element — but also includes flat embroidered flowers "
        "when the floral motif is the dominant design. "
        "NOT a floral print or woven pattern on the fabric itself."
    ),
    "fringe": (
        "Strips of hanging material — leather, suede, fabric or synthetic — "
        "creating a fringe effect. Must have clearly visible individual hanging strips. "
        "NOT tassel: fringe runs along an edge as multiple strips; a tassel is a single bunch."
    ),
    "fur": (
        "Real or faux fur used as trim or upper material. "
        "Includes fur trim along edges, fur collar at the ankle opening, "
        "fur-covered uppers, and fur pompoms. "
        "A fur pompom should be marked as BOTH fur AND pom-pom simultaneously. "
        "The key is the fur TEXTURE — soft, dense, fluffy fibres."
    ),
    "hardware": (
        "A rigid metal decorative element applied to the upper that is not a buckle, "
        "not a chain and not a stud. "
        "Includes metal bars, plates, rings, interlocking links, logo plaques and "
        "architectural metal ornaments used as a design focal point. "
        "Typically sits across the vamp of loafers or on the strap of sandals "
        "as a statement piece. "
        "NOT a buckle: hardware does not open, close or adjust the shoe. "
        "NOT a chain: chain is a series of linked loops — hardware is a single rigid piece "
        "or a small cluster of rigid metal elements. "
        "NOT a stud: studs are small individual points applied across a surface — "
        "hardware is a larger single ornamental metal element."
    ),
    "lace": (
        "Lace fabric used as an upper material or applied as decorative overlay. "
        "Must be clearly lace fabric with its characteristic openwork pattern. "
        "NOT laces (shoelaces) — this refers to lace fabric as a material."
    ),
    "mesh-insert": (
        "A section of mesh, net or open-weave fabric inserted into or overlaid on the upper. "
        "Must be clearly a mesh material — a visible grid or net texture. "
        "Includes athletic mesh, fishnet overlays and decorative net panels."
    ),
    "patch": (
        "A distinct piece of fabric, leather or synthetic material applied onto the upper "
        "as a decorative element, clearly separate from the main upper material. "
        "The patch has visible edges — often with stitching, a border or a colour contrast "
        "that outlines its boundaries. "
        "May contain a printed, woven, knitted or embroidered motif within it. "
        "NOT embroidery: embroidery is stitching directly onto the upper with no separate "
        "fabric piece — a patch is an entire applied panel with distinct edges. "
        "NOT a mesh-insert: a patch is opaque and decorative, not a functional open-weave panel."
    ),

    "pearl": (
        "Pearl or pearl-like spherical decorative elements applied to the shoe. "
        "Includes genuine pearls, faux pearls and pearl-finish beads. "
        "Distinguished from crystal by their round, non-faceted, lustrous appearance."
    ),
    "pom-pom": (
        "A rounded or starburst-shaped decorative ball or rosette attached to the shoe "
        "as a distinct ornamental element. "
        "Can be fluffy (fur, yarn, fabric fibres) OR structured (beads, crystals, "
        "fabric petals radiating from a centre). "
        "The defining feature is a DISTINCT ROUNDED OR RADIAL DECORATIVE ELEMENT "
        "attached at the toe, vamp or ankle — clearly separate from the shoe upper itself. "
        "Includes: fur pompoms, yarn pompoms, beaded starburst rosettes, "
        "crystal sunburst decorations, and fabric flower-ball hybrids. "
        "A fur pompom should be marked as BOTH pom-pom AND fur simultaneously. "
        "NOT a bow: a bow has two distinct loops. "
        "NOT a flower appliqué: flat fabric flowers without a ball/radial structure."
    ),
    "ribbon": (
        "Ribbon used as decorative trim, bow or tie detail. "
        "Includes satin ribbons, grosgrain ribbons and ribbon bows. "
        "NOT functional laces — only count ribbon used as a decorative element."
    ),
    "sequin": (
        "Small shiny disc-shaped decorative elements covering part or all of the upper. "
        "Creates a glittery, light-reflecting surface. "
        "Distinguished from crystal by being flat discs rather than faceted stones."
    ),
    "stripe": (
        "One or more vertical contrasting bands or panels running along the upper of the shoe or boot, "
        "clearly distinct in colour or material from the main upper. "
        "The stripe is flush with the upper surface — integrated into the design rather than "
        "applied on top as trim or ribbon. "
        "Includes a single contrasting stripe (e.g. white panel on a black boot) "
        "and multiple parallel stripes (e.g. two or three bands of different colours). "
        "Common on boots and ankle boots as a vertical panel from shaft to toe. "
        "NOT a ribbon: ribbon sits on top of the upper as a separate element. "
        "NOT a seam or stitching line: the stripe must be a visibly distinct colour or material band."
    ),
    "stud": (
        "Metal studs, spikes or rivet-like decorative elements applied to the upper. "
        "Includes pyramid studs, dome studs, spikes and decorative rivets. "
        "Common on biker-style boots, sandal straps and edgy designs."
        "Metal studs, snaps, disc hardware, spikes or rivet-like elements applied to the upper."
        "Includes dome studs, pyramid studs, decorative press-studs and large metal disc details."
    ),
    "tassel": (
        "A hanging bunch of threads, cords or leather strips tied together at the top. "
        "Typically appears on loafers, mules or ankle ties. "
        "Distinguished from fringe by being a single gathered bunch rather than edge strips."
    ),

}

    # "logo": (
    #     "A visible brand logo, monogram or lettering used as a deliberate design element. "
    #     "Includes embossed logos, printed brand marks, metal logo plates and repeat logo patterns."
    # ),
    #"zipper": (
    #     "A zipper used as a visible decorative design element rather than purely as a closure. "
    #     "Includes exposed contrast zippers, decorative zippers on non-opening positions, "
    #     "or zippers used as surface ornamentation. "
    #     "Do NOT mark a plain functional back-zip or side-zip boot closure as this embellishment "
    #     "unless it is clearly intended as a decorative feature."
    #     "a plain boot side-zip or back-zip is NOT decorative even if visible"
    # ),
DEFAULT_EMBELLISHMENT_ATTRS = list(EMBELLISHMENT_DESCRIPTIONS.keys())


def build_embellishment_prompt(attributes: list[str]) -> str:
    attr_lines = "\n".join(
        f'  "{a}": {EMBELLISHMENT_DESCRIPTIONS.get(a, a.replace("-", " "))}'
        for a in attributes
    )
    json_template = "\n".join(
        f'  "{a}": true_or_false' for a in attributes
    )
    return (
        f"Examine this shoe image and detect which embellishments are present.\n\n"
        f"Embellishments:\n{attr_lines}\n\n"
        f"Rules:\n"
        f"- true = embellishment is clearly visible as a deliberate decorative element\n"
        f"- false = absent, not visible, purely functional, or uncertain\n"
        f"- Multiple embellishments can be true simultaneously\n"
        f"- No explanations, no markdown, just the JSON\n\n"
        f"Respond with ONLY:\n{{\n{json_template}\n}}"
    )


def call_embellishment(
    client: anthropic.Anthropic,
    b64: str,
    media_type: str,
    attributes: list[str],
) -> dict[str, bool] | None:
    result = _call(
        client, b64, media_type,
        SYSTEM_PROMPT_EMBELLISHMENT,
        build_embellishment_prompt(attributes),
    )
    if result is None:
        return None
    return {attr: bool(result.get(attr, False)) for attr in attributes}


def load_gt_embellishment(
    labels_csv: Path,
    attributes: list[str],
    val_only: bool,
    max_images: int | None,
    seed: int = 42,
    val_split: float = 0.2,
) -> pd.DataFrame:
    """
    Expects CSV columns: name, embellishment (pipe-separated values).
    Falls back to column named 'embellishments' if present.
    Images with no embellishment should have an empty string or NaN.
    """
    df = pd.read_csv(labels_csv)

    # Flexible column name detection
    col = None
    for candidate in ("embellishment", "embellishments"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(
            f"CSV must have an 'embellishment' or 'embellishments' column.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    df[col] = df[col].fillna("")

    for attr in attributes:
        df[f"gt_{attr}"] = df[col].apply(
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
        if n_pos > 0:
            print(f"  {attr:<22} {n_pos:>3} positive / {len(df)} total  ({100*n_pos/len(df):.1f}%)")
    zero_attrs = [a for a in attributes if df[f"gt_{a}"].sum() == 0]
    if zero_attrs:
        print(f"  (zero positives in this split: {', '.join(zero_attrs)})")

    return df


def run_embellishment(
    client: anthropic.Anthropic,
    labels_csv: Path,
    image_dir: Path,
    attributes: list[str],
    max_images: int | None,
    val_only: bool,
    out_csv: Path | None,
) -> None:
    df = load_gt_embellishment(labels_csv, attributes, val_only, max_images)

    results = []
    missing = failed = 0

    print(f"\nRunning Claude ({MODEL}) — embellishment mode on {len(df)} images...\n")

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

        preds = call_embellishment(client, b64, media_type, attributes)
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
    # Reuse construction reporter — identical multi-label format
    report_construction(results_df, attributes)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Probe mode — free-text VLM vocabulary extraction
# ─────────────────────────────────────────────────────────────────────────────

PROBE_SYSTEM = (
    "You are a professional footwear expert with deep knowledge of shoe construction "
    "and design terminology. Describe shoes using precise industry vocabulary."
)

PROBE_HEEL_PROMPT = (
    "Look at this shoe image. Focus exclusively on the heel.\n\n"
    "Describe the heel using the exact technical terms a footwear designer or buyer "
    "would use. Be specific about:\n"
    "  1. The heel NAME (e.g. the specific industry term for this heel type)\n"
    "  2. The heel SHAPE in one sentence\n"
    "  3. Any alternative names this heel is known by\n\n"
    "Respond with ONLY a JSON object:\n"
    "{\n"
    '  "heel_name": "the primary industry term",\n'
    '  "heel_shape": "one sentence description of the shape profile",\n'
    '  "alternative_names": ["other term", "another term"]\n'
    "}"
)

PROBE_TOE_PROMPT = (
    "Look at this shoe image. Focus exclusively on the toe box.\n\n"
    "Describe the toe shape using the exact technical terms a footwear designer "
    "or buyer would use. Be specific about:\n"
    "  1. The toe shape NAME (the specific industry term)\n"
    "  2. The toe shape profile in one sentence\n"
    "  3. Any alternative names\n\n"
    "Respond with ONLY a JSON object:\n"
    "{\n"
    '  "toe_name": "the primary industry term",\n'
    '  "toe_shape": "one sentence description",\n'
    '  "alternative_names": ["other term"]\n'
    "}"
)

PROBE_BOTH_PROMPT = (
    "Look at this shoe image. Describe both the heel and toe box.\n\n"
    "Use the exact technical terms a footwear designer or buyer would use.\n\n"
    "Respond with ONLY a JSON object:\n"
    "{\n"
    '  "heel_name": "primary industry term for the heel type",\n'
    '  "heel_shape": "one sentence description of the heel profile",\n'
    '  "heel_alternative_names": ["other term", "another term"],\n'
    '  "toe_name": "primary industry term for the toe shape",\n'
    '  "toe_shape": "one sentence description",\n'
    '  "toe_alternative_names": ["other term"]\n'
    "}"
)


def call_probe(
    client: anthropic.Anthropic,
    b64_image: str,
    media_type: str,
    focus: str,          # "heel", "toe", or "both"
) -> dict | None:
    prompt = {
        "heel": PROBE_HEEL_PROMPT,
        "toe":  PROBE_TOE_PROMPT,
        "both": PROBE_BOTH_PROMPT,
    }[focus]

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                temperature=0,
                system=PROBE_SYSTEM,
                messages=[{
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
                        {"type": "text", "text": prompt},
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
        except (json.JSONDecodeError, KeyError) as e:
            print(f"\n  Parse error: {e}  raw={raw[:120]!r}")
            return None
        except Exception as e:
            print(f"\n  API error: {e}")
            return None
    return None


def run_probe(
    client: anthropic.Anthropic,
    image_dir: Path,
    focus: str,
    out_csv: Path | None,
) -> None:
    """
    Run free-text heel/toe vocabulary probe on all images in image_dir.
    No labels CSV needed — just point at a folder of images.
    Prints a frequency table of terms used and saves raw results to CSV.
    """
    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Probing {len(image_paths)} images (focus={focus})...\n")

    rows = []
    for img_path in tqdm(image_paths, desc="Probing"):
        try:
            b64, media_type = encode_image(img_path)
        except Exception as e:
            tqdm.write(f"  ENCODE ERROR {img_path.name}: {e}")
            continue

        result = call_probe(client, b64, media_type, focus)
        if result is None:
            continue

        row = {"name": img_path.stem}
        row.update(result)
        rows.append(row)
        tqdm.write(
            f"  {img_path.stem:<35} "
            f"heel={result.get('heel_name', '-'):<25} "
            f"toe={result.get('toe_name', '-')}"
        )

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)

    # ── Frequency tables ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"VLM VOCABULARY PROBE  —  {len(df)} images")
    print(f"{'='*55}\n")

    if "heel_name" in df.columns:
        print("Heel terms used (primary name):")
        for term, count in df["heel_name"].value_counts().items():
            print(f"  {count:>3}x  {term}")

        # Collect all alternative names too
        if "heel_alternative_names" in df.columns:
            all_alts = []
            for alts in df["heel_alternative_names"].dropna():
                if isinstance(alts, list):
                    all_alts.extend(alts)
                elif isinstance(alts, str):
                    try:
                        all_alts.extend(json.loads(alts))
                    except Exception:
                        pass
            if all_alts:
                from collections import Counter
                print("\nHeel alternative names mentioned:")
                for term, count in Counter(all_alts).most_common():
                    print(f"  {count:>3}x  {term}")

    if "toe_name" in df.columns:
        print("\nToe terms used (primary name):")
        for term, count in df["toe_name"].value_counts().items():
            print(f"  {count:>3}x  {term}")

    # ── Save ──────────────────────────────────────────────────────────────
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nRaw probe results → {out_csv}")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate Claude VLM on shoe attribute recognition"
    )
    p.add_argument("--mode", choices=["toe-heel", "construction", "embellishment", "probe"],
                   required=True,
                   help="toe-heel: categorical toe+heel classification; "
                        "construction: multi-label strap/closure attributes; "
                        "embellishment: multi-label decorative detail detection; "
                        "probe: free-text vocabulary extraction (no labels needed)")
    p.add_argument("--labels-csv",  type=Path, required=False, default=None,
                   help="Labels CSV (required for toe-heel and construction modes)")
    p.add_argument("--image-dir",   type=Path, required=True)
    p.add_argument("--attributes",  nargs="+",
                   default=DEFAULT_CONSTRUCTION_ATTRS,
                   help="Construction attributes to evaluate (construction mode only)")
    p.add_argument("--embellishments", nargs="+",
                   default=DEFAULT_EMBELLISHMENT_ATTRS,
                   help="Embellishment attributes to evaluate (embellishment mode only)")
    p.add_argument("--probe-focus", choices=["heel", "toe", "both"], default="both",
                   help="What to probe in probe mode (default: both)")
    p.add_argument("--val-only",    action="store_true")
    p.add_argument("--out-csv",     type=Path, default=None)
    p.add_argument("--max-images",     type=int, default=10)
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
    elif args.mode == "construction":
        run_construction(client, args.labels_csv, args.image_dir,
                         args.attributes, args.max_images, args.val_only, args.out_csv)
    elif args.mode == "embellishment":
        run_embellishment(client, args.labels_csv, args.image_dir,
                          args.embellishments, args.max_images, args.val_only, args.out_csv)
    else:  # probe
        run_probe(client, args.image_dir, args.probe_focus, args.out_csv)


if __name__ == "__main__":
    main()
