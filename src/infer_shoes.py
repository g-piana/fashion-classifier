"""
infer_shoes.py
==============
Production inference pipeline for shoe attribute tagging.

For each image, makes TWO Claude API calls:
  Call 1 — toe shape + heel type  (categorical, one value each)
  Call 2 — construction + embellishment attributes  (multi-label)

Output is a single CSV with one row per image and all attributes as columns.

Features
--------
  - Resume support: already-processed rows are skipped on re-run
  - Progress bar with ETA
  - Graceful handling of API errors (row saved with empty values, not lost)
  - Configurable concurrency for faster throughput

Usage
-----
    # Basic — process all images in a folder
    python src/infer_shoes.py \
        --image-dir "E:/fashion-data/01-RAW/shoes_production" \
        --out-csv   "E:/fashion-data/csv/shoes_tagged.csv"

    # Resume an interrupted run (skips already-processed stems)
    python src/infer_shoes.py \
        --image-dir "E:/fashion-data/01-RAW/shoes_production" \
        --out-csv   "E:/fashion-data/csv/shoes_tagged.csv" \
        --resume

    # Process a single category subfolder
    python src/infer_shoes.py \
        --image-dir "E:/fashion-data/01-RAW/shoes_production/heels" \
        --out-csv   "E:/fashion-data/csv/shoes_tagged.csv" \
        --resume

    # Skip toe/heel or skip construction/embellishment
    python src/infer_shoes.py \
        --image-dir "..." --out-csv "..." \
        --no-toe-heel
        --no-attributes

    # Dry run — show what would be processed without calling API
    python src/infer_shoes.py \
        --image-dir "..." --out-csv "..." \
        --dry-run

Environment
-----------
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
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

# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy
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

CONSTRUCTION_ATTRS = [
    "strappy", "ankle-strap", "ankle-wrap", "sling-back",
    "t-bar", "cross-over", "platform", "lace-up", "zip-up", "slouch",
]

EMBELLISHMENT_ATTRS = [
    "bead", "bow", "buckle", "chain", "crystal", "embroidery",
    "eyelet", "feather", "flower", "fringe", "fur", "hardware", "lace",
     "mesh-insert", "patch", "pearl", "pom-pom", "ribbon", "sequin",
    "stripe", "stud", "tassel",
]

ALL_ATTRIBUTE_COLS = CONSTRUCTION_ATTRS + EMBELLISHMENT_ATTRS

# ─────────────────────────────────────────────────────────────────────────────
# Toe + heel descriptions
# ─────────────────────────────────────────────────────────────────────────────

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
    "NOT a comma-heel: comma heels are wide at the top with a back face that curves outward as it descends. "
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
# Construction descriptions
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
        "A strap that goes across the back of the heel, holding the shoe on from behind. "
        "The defining feature is the strap position: at the back of the heel, "
        "not wrapping around the full ankle circumference. "
        "Sling-back CAN co-occur with ankle-strap: some shoes have both a heel sling-back "
        "strap AND a separate ankle-strap with a buckle — mark both when present. "
        "NOT ankle-strap alone: an ankle-strap that wraps the full ankle with a side buckle "
        "is ankle-strap, not sling-back. "
        "The sling-back strap runs horizontally across the back of the heel only."
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
# Embellishment descriptions
# ─────────────────────────────────────────────────────────────────────────────

EMBELLISHMENT_DESCRIPTIONS = {
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

# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TOE_HEEL = (
    "You are a footwear attribute recognition system. "
    "You classify shoe images by toe shape and heel type. "
    "Each shoe has exactly one toe shape and exactly one heel type — "
    "choose the single best match from the provided lists. "
    "Be precise. Only select what is clearly visible. "
    "Always respond with valid JSON only, no other text."
)

SYSTEM_PROMPT_ATTRIBUTES = (
    "You are a footwear attribute recognition system. "
    "You detect construction, closure and embellishment attributes in shoe images. "
    "Multiple attributes can be present simultaneously. "
    "Only mark an attribute as present if it is clearly visible. "
    "Do not infer — if you cannot see it, mark it false. "
    "Always respond with valid JSON only, no other text."
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

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
        f"Respond with ONLY this JSON:\n"
        f'{{\n  "toe_shape": "<value>",\n  "heel_type": "<value>"\n}}'
    )


def build_attributes_prompt(
    construction: list[str],
    embellishments: list[str],
) -> str:
    c_lines = "\n".join(
        f'  "{a}": {CONSTRUCTION_DESCRIPTIONS.get(a, a)}'
        for a in construction
    )
    e_lines = "\n".join(
        f'  "{a}": {EMBELLISHMENT_DESCRIPTIONS.get(a, a)}'
        for a in embellishments
    )
    all_attrs = construction + embellishments
    json_template = "\n".join(f'  "{a}": true_or_false' for a in all_attrs)
    return (
        f"Examine this shoe image and detect which attributes are present.\n\n"
        f"CONSTRUCTION & CLOSURE:\n{c_lines}\n\n"
        f"EMBELLISHMENTS:\n{e_lines}\n\n"
        f"Rules:\n"
        f"- true = attribute is clearly visible\n"
        f"- false = absent, not visible, or uncertain\n"
        f"- Multiple attributes can be true simultaneously\n"
        f"- No explanations, no markdown, just the JSON\n\n"
        f"Respond with ONLY:\n{{\n{json_template}\n}}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_images(image_dir: Path) -> list[Path]:
    """Recursively find all supported images in image_dir."""
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(image_dir.rglob(f"*{ext}"))
    return sorted(set(images))


def encode_image(image_path: Path, max_px: int = MAX_IMAGE_PX) -> tuple[str, str]:
    import base64, io
    from PIL import Image

    with Image.open(image_path) as img:          # ← explicit close
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)

    b64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()                                   # ← explicit close
    return b64, "image/jpeg"

# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def _call(
    client: anthropic.Anthropic,
    b64: str,
    media_type: str,
    system: str,
    user: str,
    max_tokens: int = 512,
) -> dict | None:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
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
                tqdm.write(f"  Rate limit — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                tqdm.write("  Rate limit — giving up on this image")
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
# Per-image inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_image(
    client: anthropic.Anthropic,
    image_path: Path,
    run_toe_heel: bool,
    run_attributes: bool,
    construction: list[str],
    embellishments: list[str],
) -> dict:
    """
    Run inference on a single image.
    Returns a dict with all output columns — empty strings on failure.
    """
    record: dict = {"name": image_path.stem}

    # Initialise all columns to empty string (distinguishable from False)
    if run_toe_heel:
        record["toe_shape"] = ""
        record["heel_type"] = ""
    if run_attributes:
        for a in construction + embellishments:
            record[a] = ""

    try:
        b64, media_type = encode_image(image_path)
    except Exception as e:
        tqdm.write(f"  ENCODE ERROR {image_path.name}: {e}")
        return record

    # Call 1 — toe shape + heel type
    if run_toe_heel:
        result = _call(
            client, b64, media_type,
            SYSTEM_PROMPT_TOE_HEEL,
            build_toe_heel_prompt(),
            max_tokens=64,
        )
        if result:
            toe = result.get("toe_shape", "")
            heel = result.get("heel_type", "")
            record["toe_shape"] = toe if toe in TOE_SHAPES else f"_invalid:{toe}"
            record["heel_type"] = heel if heel in HEEL_TYPES else f"_invalid:{heel}"

    # Call 2 — construction + embellishment (combined)
    if run_attributes:
        result = _call(
            client, b64, media_type,
            SYSTEM_PROMPT_ATTRIBUTES,
            build_attributes_prompt(construction, embellishments),
            max_tokens=512,
        )
        if result:
            for a in construction + embellishments:
                record[a] = bool(result.get(a, False))

    return record

# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    client: anthropic.Anthropic,
    image_dir: Path,
    out_csv: Path,
    run_toe_heel: bool,
    run_attributes: bool,
    construction: list[str],
    embellishments: list[str],
    resume: bool,
    dry_run: bool,
    workers: int = 1,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
        done_stems  = set(existing_df["name"].astype(str))
        existing_rows = existing_df.to_dict("records")
        print(f"Resuming — {len(done_stems)} already processed, "
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
        print("Nothing to process — all images already done.")
        return

    # Column order for output CSV
    cols = ["name"]
    if run_toe_heel:
        cols += ["toe_shape", "heel_type"]
    if run_attributes:
        cols += construction + embellishments

    results = list(existing_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(to_process)} images "
          f"({'toe+heel + ' if run_toe_heel else ''}attributes) "
          f"with {workers} worker(s)...\n")

    write_header = not (resume and out_csv.exists())
    write_lock   = threading.Lock()

    def _process_and_write(img_path: Path) -> dict:
        record = infer_image(
            client, img_path,
            run_toe_heel, run_attributes,
            construction, embellishments,
        )
        with write_lock:
            nonlocal write_header
            row_df = pd.DataFrame([record])[cols]
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                row_df.to_csv(f, index=False, header=write_header)
            write_header = False
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_and_write, p): p for p in to_process}
        with tqdm(total=len(to_process), desc="Tagging") as bar:
            for future in as_completed(futures):
                try:
                    record = future.result()
                    results.append(record)
                except Exception as e:
                    img_path = futures[future]
                    tqdm.write(f"  WORKER ERROR {img_path.name}: {e}")
                finally:
                    bar.update(1)

    print(f"\nDone. {len(results)} images tagged → {out_csv}")

    # Summary
    df = pd.DataFrame(results)
    if run_toe_heel and "toe_shape" in df.columns:
        print("\nToe shape distribution:")
        print(df["toe_shape"].value_counts().to_string())
        print("\nHeel type distribution:")
        print(df["heel_type"].value_counts().to_string())
    if run_attributes:
        print("\nAttribute prevalence:")
        attr_cols = [c for c in (construction + embellishments) if c in df.columns]
        for col in attr_cols:
            n = df[col].apply(lambda x: x is True or x == "True" or x == 1).sum()
            pct = 100 * n / len(df)
            if n > 0:
                print(f"  {col:<20} {n:>4}  ({pct:.1f}%)")



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Production shoe attribute tagging pipeline"
    )
    p.add_argument("--image-dir",  type=Path, required=True,
                   help="Folder containing shoe images (searched recursively)")
    p.add_argument("--out-csv",    type=Path, required=True,
                   help="Output CSV path")
    p.add_argument("--resume",     action="store_true",
                   help="Skip images already present in out-csv")
    p.add_argument("--dry-run",    action="store_true",
                   help="Show what would be processed without calling the API")
    p.add_argument("--no-toe-heel",    dest="toe_heel",    action="store_false",
                   help="Skip toe shape and heel type classification")
    p.add_argument("--no-attributes",  dest="attributes",  action="store_false",
                   help="Skip construction and embellishment detection")
    p.add_argument("--construction",   nargs="+", default=CONSTRUCTION_ATTRS,
                   help="Override construction attribute list")
    p.add_argument("--embellishments", nargs="+", default=EMBELLISHMENT_ATTRS,
                   help="Override embellishment attribute list")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers (default: 1, recommended: 4-5)")
    args = p.parse_args()

    if not args.toe_heel and not args.attributes:
        p.error("--no-toe-heel and --no-attributes cannot both be set")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)"
        )
    client = anthropic.Anthropic(api_key=api_key or "dry-run")

    run_inference(
        client=client,
        image_dir=args.image_dir,
        out_csv=args.out_csv,
        run_toe_heel=args.toe_heel,
        run_attributes=args.attributes,
        construction=args.construction,
        embellishments=args.embellishments,
        resume=args.resume,
        dry_run=args.dry_run,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()