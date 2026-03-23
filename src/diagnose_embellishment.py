"""
diagnose_construction.py
========================
Opens an HTML image grid for false-positive or false-negative cases
of any construction attribute, so you can visually spot patterns
without manually digging through CSV.

Usage
-----
    # Show all ankle-strap false positives
    python src/diagnose_construction.py \
        --eval-csv  "E:/fashion-data/csv/construction_eval_01.csv" \
        --labels-csv "E:/fashion-data/csv/shoes_labels.csv" \
        --image-dir "E:/fashion-data/01-RAW/shoes" \
        --attr ankle-strap \
        --error fp

    # Show cross-over false negatives (misses)
    python src/diagnose_construction.py ... --attr cross-over --error fn

    # Show ALL error types for all attributes
    python src/diagnose_construction.py ... --attr all --error both

    # Show only images where Claude predicted ankle-strap (TP + FP together)
    python src/diagnose_construction.py ... --attr ankle-strap --error pred
"""

from __future__ import annotations
import argparse
import base64
import webbrowser
from pathlib import Path
import pandas as pd

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]
ALL_ATTRS = [
    "bead", "bow", "buckle", "chain", "crystal", "embroidery",
    "eyelet", "feather", "flower", "fringe", "fur", "lace", "logo",
    "mesh-insert", "pearl", "pom-pom", "ribbon", "sequin", "stud",
    "tassel", "zipper",
]


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # one level of subfolders
    for sub in image_dir.iterdir():
        if not sub.is_dir():
            continue
        for ext in IMAGE_EXTENSIONS:
            p = sub / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        data = f.read()
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def load_eval_csv(eval_csv: Path, attrs: list[str]) -> pd.DataFrame:
    """
    Eval CSV can have columns like pred_ankle-strap / gt_ankle-strap,
    OR it may have columns named differently. We detect the format.
    """
    df = pd.read_csv(eval_csv)
    cols = df.columns.tolist()
    print(f"Eval CSV columns: {cols[:10]}{'...' if len(cols)>10 else ''}")
    return df


def build_cases(
    eval_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    attr: str,
    error_type: str,  # "fp", "fn", "tp", "both", "pred"
    image_dir: Path,
) -> list[dict]:
    """
    Merge eval predictions with GT labels and return matching error cases.
    Handles two possible eval CSV formats:
      Format A: columns  pred_<attr>  and  gt_<attr>  (one col per attribute)
      Format B: columns  predicted  and  construction  (raw strings)
    """
    pred_col = f"pred_{attr}"
    gt_col   = f"gt_{attr}"

    # --- Detect format ---
    if pred_col in eval_df.columns and gt_col in eval_df.columns:
        # Format A — already binary per attribute
        merged = eval_df[["name", pred_col, gt_col]].copy()
        merged.rename(columns={pred_col: "pred", gt_col: "gt"}, inplace=True)
    elif "predicted" in eval_df.columns:
        # Format B — raw predicted string, merge with GT CSV for gt
        gt_df = gt_df.copy()
        gt_col_name = next((c for c in ("embellishment", "embellishments") if c in gt_df.columns), None)
        if gt_col_name is None:
            raise ValueError("GT CSV must have an 'embellishment' or 'embellishments' column.")
        gt_df["gt"] = gt_df[gt_col_name].fillna("").apply(
            lambda x: int(attr in [a.strip() for a in x.split("|")])
        )
        eval_df["pred"] = eval_df["predicted"].fillna("").apply(
            lambda x: int(attr in [a.strip() for a in x.replace(",", "|").split("|")])
        )
        merged = eval_df[["name", "pred"]].merge(gt_df[["name", "gt"]], on="name", how="inner")
    else:
        raise ValueError(
            f"Cannot find columns for attribute '{attr}' in eval CSV.\n"
            f"Available: {eval_df.columns.tolist()}"
        )

    # Also attach full GT embellishment string for display
    gt_col_name = next((c for c in ("embellishment", "embellishments") if c in gt_df.columns), "embellishment")
    gt_full = gt_df.set_index("name")[gt_col_name].fillna("").to_dict()

    # --- Filter by error type ---
    if error_type == "fp":
        subset = merged[(merged["pred"] == 1) & (merged["gt"] == 0)]
    elif error_type == "fn":
        subset = merged[(merged["pred"] == 0) & (merged["gt"] == 1)]
    elif error_type == "tp":
        subset = merged[(merged["pred"] == 1) & (merged["gt"] == 1)]
    elif error_type == "tn":
        subset = merged[(merged["pred"] == 0) & (merged["gt"] == 0)]
    elif error_type == "pred":
        subset = merged[merged["pred"] == 1]
    else:  # "both" — all errors
        subset = merged[((merged["pred"] == 1) & (merged["gt"] == 0)) |
                        ((merged["pred"] == 0) & (merged["gt"] == 1))]

    cases = []
    for _, row in subset.iterrows():
        stem = row["name"]
        img_path = find_image(image_dir, stem)
        gt_str = gt_full.get(stem, "")
        pred_label = "✓ predicted" if row["pred"] == 1 else "✗ not predicted"
        gt_label   = f"GT: {gt_str if gt_str else '(none)'}"

        tp = row["pred"] == 1 and row["gt"] == 1
        fp = row["pred"] == 1 and row["gt"] == 0
        fn = row["pred"] == 0 and row["gt"] == 1

        if tp:
            badge = "TP"
            color = "#2a9d2a"
        elif fp:
            badge = "FP"
            color = "#c0392b"
        else:
            badge = "FN"
            color = "#e67e22"

        cases.append({
            "stem":     stem,
            "img_path": img_path,
            "gt_str":   gt_str,
            "pred":     int(row["pred"]),
            "gt":       int(row["gt"]),
            "badge":    badge,
            "color":    color,
            "label":    f"{badge} — GT: [{gt_str or 'none'}]",
        })

    return cases


def render_html(cases: list[dict], attr: str, error_type: str, out_path: Path) -> None:
    missing = [c for c in cases if c["img_path"] is None]
    found   = [c for c in cases if c["img_path"] is not None]

    cards = []
    for c in found:
        b64 = img_to_b64(c["img_path"])
        cards.append(f"""
        <div class="card">
          <div class="badge" style="background:{c['color']}">{c['badge']}</div>
          <img src="{b64}" alt="{c['stem']}">
          <div class="info">
            <div class="stem">{c['stem']}</div>
            <div class="gt">GT: {c['gt_str'] or '<em>none</em>'}</div>
          </div>
        </div>""")

    missing_list = ", ".join(m["stem"] for m in missing) if missing else "none"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{attr} — {error_type.upper()} cases</title>
<style>
  body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1   {{ color: #e0c97f; }}
  .summary {{ margin-bottom: 20px; color: #aaa; font-size: 0.9em; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .card {{
    background: #16213e; border-radius: 8px; overflow: hidden;
    width: 200px; position: relative;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }}
  .badge {{
    position: absolute; top: 6px; left: 6px;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.75em; font-weight: bold; color: white;
  }}
  .card img {{ width: 200px; height: 200px; object-fit: cover; display: block; }}
  .info {{ padding: 8px; }}
  .stem {{ font-size: 0.75em; color: #aaa; word-break: break-all; }}
  .gt  {{ font-size: 0.8em; color: #e0c97f; margin-top: 4px; }}
  .missing {{ color: #888; font-size: 0.85em; margin-top: 10px; }}
</style>
</head>
<body>
<h1>Attribute: <code>{attr}</code> — {error_type.upper()} ({len(cases)} cases)</h1>
<div class="summary">
  Found images: {len(found)} &nbsp;|&nbsp; Missing on disk: {len(missing)}
  {'<br>Missing stems: ' + missing_list if missing else ''}
</div>
<div class="grid">{''.join(cards)}</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"HTML saved → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Visual diagnostic grid for construction eval errors")
    p.add_argument("--eval-csv",   required=True, type=Path)
    p.add_argument("--labels-csv", required=True, type=Path)
    p.add_argument("--image-dir",  required=True, type=Path)
    p.add_argument("--attr",  default="ankle-strap",
                   help=f"Attribute name, or 'all'. One of: {ALL_ATTRS}")
    p.add_argument("--error", default="fp",
                   choices=["fp", "fn", "tp", "tn", "both", "pred"],
                   help="fp=false positives, fn=false negatives, pred=all Claude predicted positive")
    p.add_argument("--out-html", type=Path, default=None,
                   help="Output HTML path (default: eval-csv dir/diagnose_<attr>_<error>.html)")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    eval_df   = pd.read_csv(args.eval_csv)
    labels_df = pd.read_csv(args.labels_csv)

    attrs = ALL_ATTRS if args.attr == "all" else [args.attr]

    for attr in attrs:
        print(f"\n--- {attr} ({args.error.upper()}) ---")
        try:
            cases = build_cases(eval_df, labels_df, attr, args.error, args.image_dir)
        except ValueError as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Cases: {len(cases)}")
        for c in cases:
            status = "IMG OK" if c["img_path"] else "IMG MISSING"
            print(f"    {c['badge']}  {c['stem']:30s}  GT=[{c['gt_str'] or 'none':40s}]  [{status}]")

        if not cases:
            print("  No cases found.")
            continue

        out_html = args.out_html or (
            args.eval_csv.parent / f"diagnose_{attr.replace('-','_')}_{args.error}.html"
        )
        render_html(cases, attr, args.error, out_html)

        if not args.no_browser:
            webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
