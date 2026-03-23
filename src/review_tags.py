"""
review_tags.py
==============
Generates a standalone HTML page showing shoe images with their
predicted tags for visual validation.

Reads the tagged CSV and the image folder, embeds images as base64
so the HTML file is fully self-contained and portable.

Usage
-----
    python src/review_tags.py \
        --tagged-csv "E:/fashion-data/csv/shoes_tagged.csv" \
        --image-dir  "E:/fashion-data/01-RAW/shoes_production" \
        --out-html   "E:/fashion-data/review/shoes_review.html" \
        --page-size  50

    # Review only images with a specific attribute
    python src/review_tags.py ... --filter-attr crystal

    # Review only images where toe_shape is a specific value
    python src/review_tags.py ... --filter-toe open-toe

    # Limit to first N images (fast preview)
    python src/review_tags.py ... --max-images 30
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"}
THUMB_SIZE      = (280, 280)   # thumbnail size in the HTML grid
THUMB_QUALITY   = 75

CATEGORICAL_COLS = ["toe_shape", "heel_type"]

CONSTRUCTION_ATTRS = [
    "strappy", "ankle-strap", "ankle-wrap", "sling-back",
    "t-bar", "cross-over", "platform", "lace-up", "zip-up", "slouch",
]

EMBELLISHMENT_ATTRS = [
    "bead", "bow", "buckle", "chain", "crystal", "embroidery",
    "eyelet", "feather", "flower", "fringe", "fur", "hardware", "lace", 
    "mesh-insert", "patch", "pearl", "pom-pom", "ribbon", "sequin", "stripe", "stud",
    "tassel",
]

# Tag badge colours
COLOR_TOE       = "#2563eb"   # blue
COLOR_HEEL      = "#7c3aed"   # purple
COLOR_CONSTRUCT = "#059669"   # green
COLOR_EMBELLISH = "#d97706"   # amber
COLOR_MISSING   = "#6b7280"   # grey (image not found)
COLOR_INVALID   = "#dc2626"   # red (invalid prediction)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def encode_thumbnail(image_path: Path) -> str:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=THUMB_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def is_true(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    try:
        return bool(val)
    except Exception:
        return False


def badge(text: str, color: str) -> str:
    return (
        f'<span class="badge" style="background:{color}">'
        f'{text}</span>'
    )


def make_card(row: pd.Series, image_dir: Path, attr_cols: list[str]) -> str:
    stem      = str(row["name"])
    img_path  = find_image(image_dir, stem)

    # Image
    if img_path:
        b64 = encode_thumbnail(img_path)
        img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="{stem}">'
    else:
        img_tag = f'<div class="no-img">Image not found</div>'

    # Categorical badges
    cat_badges = []
    for col in CATEGORICAL_COLS:
        if col not in row.index:
            continue
        val = str(row[col]) if pd.notna(row[col]) else ""
        if not val:
            continue
        color = COLOR_TOE if col == "toe_shape" else COLOR_HEEL
        if val.startswith("_invalid"):
            color = COLOR_INVALID
        label = val.replace("_invalid:", "⚠ ")
        cat_badges.append(badge(label, color))

    # Attribute badges — only show True ones
    construct_badges = []
    embellish_badges = []

    for col in attr_cols:
        if col in CATEGORICAL_COLS or col == "name":
            continue
        val = row.get(col, False)
        if not is_true(val):
            continue
        if col in CONSTRUCTION_ATTRS:
            construct_badges.append(badge(col, COLOR_CONSTRUCT))
        elif col in EMBELLISHMENT_ATTRS:
            embellish_badges.append(badge(col, COLOR_EMBELLISH))

    # Assemble tag sections
    tag_html = ""
    if cat_badges:
        tag_html += f'<div class="tag-row">{"".join(cat_badges)}</div>'
    if construct_badges:
        tag_html += f'<div class="tag-row">{"".join(construct_badges)}</div>'
    if embellish_badges:
        tag_html += f'<div class="tag-row">{"".join(embellish_badges)}</div>'
    if not cat_badges and not construct_badges and not embellish_badges:
        tag_html = '<div class="tag-row"><span class="no-tags">no tags</span></div>'

    missing_note = "" if img_path else '<div class="missing-note">⚠ image not found</div>'

    return f"""
    <div class="card">
      <div class="img-wrap">{img_tag}</div>
      <div class="info">
        <div class="stem">{stem}</div>
        {missing_note}
        <div class="tags">{tag_html}</div>
      </div>
    </div>"""


def render_html(
    rows: list[pd.Series],
    image_dir: Path,
    attr_cols: list[str],
    title: str,
    total_count: int,
    page_size: int,
) -> str:
    # Legend
    legend_items = [
        (COLOR_TOE,       "Toe shape"),
        (COLOR_HEEL,      "Heel type"),
        (COLOR_CONSTRUCT, "Construction"),
        (COLOR_EMBELLISH, "Embellishment"),
        (COLOR_INVALID,   "Invalid prediction"),
    ]
    legend_html = "".join(
        f'<span class="legend-item">'
        f'<span class="legend-dot" style="background:{c}"></span>{l}</span>'
        for c, l in legend_items
    )

    # Stats
    df_rows = pd.DataFrame(rows)
    stats_parts = [f"Showing {len(rows)} of {total_count} images"]
    if "toe_shape" in df_rows.columns:
        top_toe = df_rows["toe_shape"].value_counts().index[0] if len(df_rows) else "-"
        stats_parts.append(f"Most common toe: {top_toe}")
    if "heel_type" in df_rows.columns:
        top_heel = df_rows["heel_type"].value_counts().index[0] if len(df_rows) else "-"
        stats_parts.append(f"Most common heel: {top_heel}")
    stats_html = " &nbsp;|&nbsp; ".join(stats_parts)

    # Cards
    print(f"Generating {len(rows)} image cards...")
    cards_html = ""
    for i, row in enumerate(rows):
        cards_html += make_card(row, image_dir, attr_cols)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)} cards rendered")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #111827; color: #f3f4f6; padding: 24px;
  }}
  h1 {{ color: #f9fafb; font-size: 1.4em; margin-bottom: 4px; }}
  .meta {{ color: #9ca3af; font-size: 0.85em; margin-bottom: 16px; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 12px;
    margin-bottom: 20px; padding: 12px;
    background: #1f2937; border-radius: 8px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8em; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: #1f2937; border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    transition: transform 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); }}
  .img-wrap {{
    width: 100%; aspect-ratio: 1;
    background: #374151; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
  }}
  .img-wrap img {{ width: 100%; height: 100%; object-fit: cover; }}
  .no-img {{
    color: #6b7280; font-size: 0.85em; text-align: center; padding: 20px;
  }}
  .info {{ padding: 10px 12px 12px; }}
  .stem {{ font-size: 0.75em; color: #9ca3af; margin-bottom: 8px; word-break: break-all; }}
  .missing-note {{ font-size: 0.75em; color: #f87171; margin-bottom: 6px; }}
  .tags {{ display: flex; flex-direction: column; gap: 5px; }}
  .tag-row {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  .badge {{
    display: inline-block; padding: 2px 8px;
    border-radius: 12px; font-size: 0.72em; font-weight: 600;
    color: white; white-space: nowrap;
  }}
  .no-tags {{ font-size: 0.75em; color: #6b7280; font-style: italic; }}

  .filter-bar {{
    margin-bottom: 20px; padding: 12px 16px;
    background: #1f2937; border-radius: 8px;
    font-size: 0.85em; color: #d1d5db;
  }}
  .filter-bar input {{
    margin-left: 8px; padding: 4px 10px;
    background: #374151; border: 1px solid #4b5563;
    color: #f3f4f6; border-radius: 6px; font-size: 0.9em;
  }}
  .filter-bar button {{
    margin-left: 8px; padding: 4px 12px;
    background: #3b82f6; border: none; color: white;
    border-radius: 6px; cursor: pointer; font-size: 0.9em;
  }}
  .count {{ color: #60a5fa; font-weight: 600; }}
</style>
</head>
<body>
<h1>🥿 Shoe Tag Review</h1>
<div class="meta">{stats_html}</div>
<div class="legend">{legend_html}</div>

<div class="filter-bar">
  Filter by stem:
  <input type="text" id="filterInput" placeholder="type to filter..."
         oninput="filterCards()">
  <button onclick="document.getElementById('filterInput').value=''; filterCards()">
    Clear
  </button>
  &nbsp;&nbsp;
  Visible: <span class="count" id="visibleCount">{len(rows)}</span>
</div>

<div class="grid" id="cardGrid">
{cards_html}
</div>

<script>
function filterCards() {{
  const q = document.getElementById('filterInput').value.toLowerCase();
  const cards = document.querySelectorAll('.card');
  let visible = 0;
  cards.forEach(card => {{
    const stem = card.querySelector('.stem').textContent.toLowerCase();
    const tags = card.querySelector('.tags').textContent.toLowerCase();
    const show = !q || stem.includes(q) || tags.includes(q);
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.getElementById('visibleCount').textContent = visible;
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate standalone HTML review page for shoe tags"
    )
    p.add_argument("--tagged-csv",  type=Path, required=True)
    p.add_argument("--image-dir",   type=Path, required=True)
    p.add_argument("--out-html",    type=Path, required=True)
    p.add_argument("--max-images",  type=int, default=None,
                   help="Limit to first N images")
    p.add_argument("--filter-attr", type=str, default=None,
                   help="Only show images where this attribute is True")
    p.add_argument("--filter-toe",  type=str, default=None,
                   help="Only show images with this toe_shape value")
    p.add_argument("--filter-heel", type=str, default=None,
                   help="Only show images with this heel_type value")
    args = p.parse_args()

    df = pd.read_csv(args.tagged_csv)
    total = len(df)
    print(f"Loaded {total} rows from {args.tagged_csv}")

    # Filters
    if args.filter_attr:
        col = args.filter_attr
        if col not in df.columns:
            print(f"WARNING: column '{col}' not found — available: {df.columns.tolist()}")
        else:
            df = df[df[col].apply(is_true)].reset_index(drop=True)
            print(f"Filtered to {len(df)} images with {col}=True")

    if args.filter_toe:
        df = df[df["toe_shape"] == args.filter_toe].reset_index(drop=True)
        print(f"Filtered to {len(df)} images with toe_shape={args.filter_toe}")

    if args.filter_heel:
        df = df[df["heel_type"] == args.filter_heel].reset_index(drop=True)
        print(f"Filtered to {len(df)} images with heel_type={args.filter_heel}")

    if args.max_images:
        df = df.head(args.max_images)
        print(f"Limited to first {len(df)} images")

    attr_cols = [c for c in df.columns if c != "name"]
    rows = [df.iloc[i] for i in range(len(df))]

    title = f"Shoe Tag Review — {args.tagged_csv.stem}"
    html = render_html(rows, args.image_dir, attr_cols, title, total, page_size=50)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"\nReview page saved → {args.out_html}")
    print("Open in any browser — no server needed.")

    import webbrowser
    webbrowser.open(args.out_html.as_uri())


if __name__ == "__main__":
    main()
