"""
review_cascade.py
=================
Generates a standalone HTML page showing shoe images with their
predicted category and subcategory for visual validation.

Reads the cascade predictions CSV (output of infer_cascade.py) and
an image folder, embeds thumbnails as base64 so the HTML file is
fully self-contained and portable — open it in any browser, no server needed.

Expected CSV columns (from infer_cascade.py):
    name, category, category_conf, subcategory, subcategory_conf

Usage
-----
    python src/review_cascade.py `
        --pred-csv  "E:/fashion-data/csv/predictions_shoes_cascade.csv" `
        --image-dir "E:/fashion-data/01-RAW/nillab_01/photo" `
        --out-html  "E:/fashion-data/review/cascade_review.html"

    # Filter to a specific category
    python src/review_cascade.py ... --filter-category heeled-shoes-women

    # Filter to a specific subcategory
    python src/review_cascade.py ... --filter-subcategory pump

    # Only show low-confidence predictions (spot-check errors)
    python src/review_cascade.py ... --max-conf 0.70

    # Only show high-confidence predictions
    python src/review_cascade.py ... --min-conf 0.90

    # Limit images for a quick preview
    python src/review_cascade.py ... --max-images 100
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

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}
THUMB_SIZE     = (300, 300)
THUMB_QUALITY  = 75

# One accent colour per category — cycles if more than 8 categories
CATEGORY_PALETTE = [
    "#3b82f6",  # blue       — boots-and-booties-women
    "#8b5cf6",  # violet     — flat-shoes-women
    "#ec4899",  # pink       — heeled-shoes-women
    "#f59e0b",  # amber      — sandals-women
    "#10b981",  # emerald    — sneakers-women
    "#ef4444",  # red
    "#06b6d4",  # cyan
    "#f97316",  # orange
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_image(image_dir: Path, stem: str) -> Path | None:
    """Flat first, then one level of subfolders."""
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


def conf_bar(conf: float, color: str) -> str:
    """Thin confidence bar under the label."""
    pct  = round(conf * 100)
    warn = " conf-warn" if conf < 0.70 else ""
    return (
        f'<div class="conf-bar-wrap{warn}">'
        f'<div class="conf-bar" style="width:{pct}%;background:{color}"></div>'
        f'<span class="conf-label">{pct}%</span>'
        f'</div>'
    )


def make_card(row: pd.Series, image_dir: Path, cat_color: dict[str, str]) -> str:
    stem     = str(row["name"])
    category = str(row.get("category", ""))
    cat_conf = float(row.get("category_conf", 0.0))
    subcat   = str(row.get("subcategory", ""))
    sub_conf = float(row.get("subcategory_conf", 0.0))

    color = cat_color.get(category, "#6b7280")

    img_path = find_image(image_dir, stem)
    if img_path:
        b64     = encode_thumbnail(img_path)
        img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="{stem}" loading="lazy">'
    else:
        img_tag = '<div class="no-img">image not found</div>'

    low_cat = ' class="low"' if cat_conf < 0.70 else ''
    low_sub = ' class="low"' if sub_conf < 0.70 else ''

    return f"""
    <div class="card" data-category="{category}" data-subcategory="{subcat}">
      <div class="img-wrap">{img_tag}</div>
      <div class="info">
        <div class="stem" title="{stem}">{stem}</div>
        <div class="label-block">
          <div class="label-row">
            <span class="pill cat" style="background:{color}">{category or "—"}</span>
          </div>
          {conf_bar(cat_conf, color)}
          <div class="label-row" style="margin-top:6px">
            <span class="pill sub"{low_cat}>{subcat or "—"}</span>
          </div>
          {conf_bar(sub_conf, "#94a3b8")}
        </div>
      </div>
    </div>"""


def render_html(
    df: pd.DataFrame,
    image_dir: Path,
    cat_color: dict[str, str],
    title: str,
    total_count: int,
) -> str:

    # Summary stats
    cat_counts = df["category"].value_counts()
    avg_cat_conf = df["category_conf"].mean()
    avg_sub_conf = df["subcategory_conf"].mean()
    low_conf_n   = (df["category_conf"] < 0.70).sum()

    stats_parts = [
        f"Showing <b>{len(df)}</b> of {total_count} images",
        f"Avg category conf: <b>{avg_cat_conf:.1%}</b>",
        f"Avg subcategory conf: <b>{avg_sub_conf:.1%}</b>",
        f"Low confidence (&lt;70%): <b>{low_conf_n}</b>",
    ]
    stats_html = " &nbsp;·&nbsp; ".join(stats_parts)

    # Category filter buttons
    cat_buttons = '<button class="cat-btn active" onclick="filterCat(\'all\', this)">All</button>\n'
    for cat, count in cat_counts.items():
        color = cat_color.get(cat, "#6b7280")
        cat_buttons += (
            f'<button class="cat-btn" onclick="filterCat(\'{cat}\', this)" '
            f'style="--accent:{color}">{cat} <span class="cnt">{count}</span></button>\n'
        )

    # Cards
    print(f"Generating {len(df)} image cards …")
    cards_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        cards_html += make_card(row, image_dir, cat_color)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

  :root {{
    --bg:       #0a0a0f;
    --surface:  #13131a;
    --surface2: #1c1c28;
    --border:   #2a2a3a;
    --text:     #e8e8f0;
    --muted:    #6b6b80;
    --low:      #f87171;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 32px 24px;
  }}

  /* ── Header ── */
  .header {{
    display: flex; align-items: flex-end;
    justify-content: space-between; flex-wrap: wrap;
    gap: 12px; margin-bottom: 28px;
    border-bottom: 1px solid var(--border); padding-bottom: 20px;
  }}
  .header h1 {{
    font-size: 1.6rem; font-weight: 800;
    letter-spacing: -0.02em;
  }}
  .header h1 span {{ color: var(--muted); font-weight: 400; }}
  .stats {{
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; color: var(--muted);
    line-height: 1.7;
    text-align: right;
  }}
  .stats b {{ color: var(--text); }}

  /* ── Controls ── */
  .controls {{
    display: flex; flex-wrap: wrap; gap: 10px;
    align-items: center; margin-bottom: 24px;
  }}
  .search-wrap {{
    position: relative; flex: 1; min-width: 220px; max-width: 340px;
  }}
  .search-wrap input {{
    width: 100%; padding: 8px 12px 8px 36px;
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px;
    font-family: 'DM Mono', monospace; font-size: 0.82rem;
    outline: none; transition: border-color 0.2s;
  }}
  .search-wrap input:focus {{ border-color: #3b82f6; }}
  .search-wrap::before {{
    content: '⌕'; position: absolute;
    left: 10px; top: 50%; transform: translateY(-50%);
    color: var(--muted); font-size: 1.1rem; pointer-events: none;
  }}

  .conf-filter {{
    display: flex; align-items: center; gap: 6px;
    font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--muted);
  }}
  .conf-filter input[type=range] {{
    accent-color: #3b82f6; width: 100px; cursor: pointer;
  }}
  .conf-filter span {{ color: var(--text); min-width: 36px; }}

  /* ── Category buttons ── */
  .cat-bar {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px;
  }}
  .cat-btn {{
    padding: 5px 14px; border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--surface2); color: var(--muted);
    font-family: 'Syne', sans-serif; font-size: 0.78rem; font-weight: 600;
    cursor: pointer; transition: all 0.15s;
  }}
  .cat-btn:hover {{
    border-color: var(--accent, #3b82f6);
    color: var(--text);
  }}
  .cat-btn.active {{
    background: var(--accent, #3b82f6);
    border-color: var(--accent, #3b82f6);
    color: #fff;
  }}
  .cat-btn .cnt {{
    opacity: 0.7; font-size: 0.85em; margin-left: 3px;
  }}

  /* ── Counter ── */
  .counter {{
    font-family: 'DM Mono', monospace; font-size: 0.78rem;
    color: var(--muted); margin-bottom: 16px;
  }}
  .counter b {{ color: #3b82f6; }}

  /* ── Grid ── */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
  }}

  /* ── Card ── */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
    transition: transform 0.15s, border-color 0.15s;
  }}
  .card:hover {{
    transform: translateY(-3px);
    border-color: #3b82f680;
  }}
  .img-wrap {{
    width: 100%; aspect-ratio: 1;
    background: var(--surface2);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }}
  .img-wrap img {{
    width: 100%; height: 100%; object-fit: cover;
    transition: transform 0.3s;
  }}
  .card:hover .img-wrap img {{ transform: scale(1.03); }}
  .no-img {{
    color: var(--muted); font-size: 0.8rem;
    font-family: 'DM Mono', monospace;
  }}

  .info {{ padding: 10px 12px 14px; }}
  .stem {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem; color: var(--muted);
    margin-bottom: 10px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}

  .label-block {{ display: flex; flex-direction: column; gap: 2px; }}
  .label-row   {{ display: flex; align-items: center; }}

  .pill {{
    display: inline-block; padding: 3px 10px;
    border-radius: 6px; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.01em; color: #fff;
    max-width: 100%; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .pill.sub {{
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text); font-weight: 600;
  }}
  .pill.sub.low {{ border-color: var(--low); color: var(--low); }}

  /* ── Confidence bar ── */
  .conf-bar-wrap {{
    display: flex; align-items: center; gap: 6px;
    margin: 4px 0 2px;
  }}
  .conf-bar-wrap.conf-warn .conf-label {{ color: var(--low); }}
  .conf-bar {{
    height: 3px; border-radius: 2px;
    background: #3b82f6; transition: width 0.3s;
    min-width: 2px;
  }}
  .conf-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; color: var(--muted);
    flex-shrink: 0;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Cascade Review <span>/ shoes</span></h1>
  <div class="stats">{stats_html}</div>
</div>

<div class="controls">
  <div class="search-wrap">
    <input type="text" id="searchInput" placeholder="search by stem or subcategory…"
           oninput="applyFilters()">
  </div>
  <div class="conf-filter">
    min conf
    <input type="range" id="confSlider" min="0" max="100" value="0"
           oninput="updateConf(this.value)">
    <span id="confVal">0%</span>
  </div>
</div>

<div class="cat-bar">
{cat_buttons}
</div>

<div class="counter">Showing <b id="visCount">{len(df)}</b> images</div>

<div class="grid" id="cardGrid">
{cards_html}
</div>

<script>
let activeCat = 'all';

function filterCat(cat, btn) {{
  activeCat = cat;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}}

function updateConf(val) {{
  document.getElementById('confVal').textContent = val + '%';
  applyFilters();
}}

function applyFilters() {{
  const q       = document.getElementById('searchInput').value.toLowerCase();
  const minConf = parseInt(document.getElementById('confSlider').value) / 100;
  const cards   = document.querySelectorAll('.card');
  let vis = 0;

  cards.forEach(card => {{
    const cat    = card.dataset.category;
    const sub    = card.dataset.subcategory.toLowerCase();
    const stem   = card.querySelector('.stem').textContent.toLowerCase();
    const catBar = card.querySelectorAll('.conf-bar')[0];
    const barW   = catBar ? parseFloat(catBar.style.width) / 100 : 1;

    const catOk  = activeCat === 'all' || cat === activeCat;
    const textOk = !q || stem.includes(q) || sub.includes(q) || cat.toLowerCase().includes(q);
    const confOk = barW >= minConf;

    const show = catOk && textOk && confOk;
    card.style.display = show ? '' : 'none';
    if (show) vis++;
  }});
  document.getElementById('visCount').textContent = vis;
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate standalone HTML review page for cascade predictions"
    )
    p.add_argument("--pred-csv",          type=Path, required=True,
                   help="CSV from infer_cascade.py "
                        "(columns: name, category, category_conf, subcategory, subcategory_conf)")
    p.add_argument("--image-dir",         type=Path, required=True,
                   help="Root folder containing shoe images (flat or one level of subfolders)")
    p.add_argument("--out-html",          type=Path, required=True,
                   help="Output HTML path")
    p.add_argument("--max-images",        type=int, default=None,
                   help="Limit to first N rows after filters (default: all)")
    p.add_argument("--filter-category",   type=str, default=None,
                   help="Only show images with this category value")
    p.add_argument("--filter-subcategory",type=str, default=None,
                   help="Only show images with this subcategory value")
    p.add_argument("--min-conf",          type=float, default=None,
                   help="Only show images where category_conf >= value (0–1)")
    p.add_argument("--max-conf",          type=float, default=None,
                   help="Only show images where category_conf <= value (useful for spotting errors)")
    p.add_argument("--no-browser",        action="store_true",
                   help="Don't open the HTML file in the browser after generation")
    args = p.parse_args()

    df = pd.read_csv(args.pred_csv)
    total = len(df)
    print(f"Loaded {total} rows from {args.pred_csv.name}")

    required_cols = {"name", "category", "category_conf", "subcategory", "subcategory_conf"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV is missing columns: {missing_cols}\n"
            f"Expected output from infer_cascade.py. Got: {list(df.columns)}"
        )

    # ── Filters ────────────────────────────────────────────────────────────
    if args.filter_category:
        df = df[df["category"] == args.filter_category].reset_index(drop=True)
        print(f"  filter category='{args.filter_category}' → {len(df)} rows")

    if args.filter_subcategory:
        df = df[df["subcategory"] == args.filter_subcategory].reset_index(drop=True)
        print(f"  filter subcategory='{args.filter_subcategory}' → {len(df)} rows")

    if args.min_conf is not None:
        df = df[df["category_conf"] >= args.min_conf].reset_index(drop=True)
        print(f"  filter category_conf >= {args.min_conf} → {len(df)} rows")

    if args.max_conf is not None:
        df = df[df["category_conf"] <= args.max_conf].reset_index(drop=True)
        print(f"  filter category_conf <= {args.max_conf} → {len(df)} rows")

    if args.max_images:
        df = df.head(args.max_images)
        print(f"  capped to first {len(df)} rows")

    if len(df) == 0:
        print("WARNING: no rows remain after filters — HTML will be empty.")

    # ── Build colour map ────────────────────────────────────────────────────
    categories = sorted(df["category"].dropna().unique())
    cat_color  = {cat: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
                  for i, cat in enumerate(categories)}

    # ── Render ──────────────────────────────────────────────────────────────
    title = f"Cascade Review — {args.pred_csv.stem}"
    html  = render_html(df, args.image_dir, cat_color, title, total)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"\nReview page saved → {args.out_html}")
    print("Open in any browser — fully self-contained, no server needed.")

    if not args.no_browser:
        import webbrowser
        webbrowser.open(args.out_html.as_uri())


if __name__ == "__main__":
    main()
