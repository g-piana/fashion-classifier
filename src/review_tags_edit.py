"""
review_tags_edit.py
===================
Generates a standalone editable HTML page for reviewing and correcting
VLM-tagged attributes. One image at a time — no DOM bloat.

The reviewer sees one large image with all attributes on the right.
Navigate with arrow keys or prev/next buttons.
Export downloads a corrected CSV ready for train.py.

No server required. Opens automatically in the default browser.

Usage
-----
    python src/review_tags_edit.py `
        --tagged-csv  "E:/fashion-data/csv/vlm_shoes_boot.csv" `
        --image-dir   "E:/fashion-data/01-RAW/shoes_emb_md_300" `
        --out-html    "E:/fashion-data/review/shoes_emb_md_300_edit.html" `
        --vlm-config  "taxonomies/vlm/shoes.yaml"

    # Only images that have _invalid: values
    python src/review_tags_edit.py ... --only-invalid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in SUPPORTED_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in image_dir.rglob(f"{stem}.*"):
        if p.suffix in SUPPORTED_EXTS:
            return p
    return None


def load_vlm_config(path: Path | None) -> dict:
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def infer_schema(df: pd.DataFrame, vlm_cfg: dict) -> dict:
    schema = {}
    cfg_attrs: dict[str, dict] = {}
    for group_cfg in (vlm_cfg.get("attribute_groups") or {}).values():
        label_type = group_cfg.get("label_type", "multi_label")
        for attr_name, attr_cfg in group_cfg.get("attributes", {}).items():
            if label_type == "categorical":
                cfg_attrs[attr_name] = {
                    "type": "categorical",
                    "classes": attr_cfg.get("classes", []),
                }
            else:
                cfg_attrs[attr_name] = {"type": "boolean", "classes": None}

    for col in df.columns:
        if col == "name":
            continue
        if col in cfg_attrs:
            schema[col] = cfg_attrs[col]
            continue
        sample = df[col].dropna()
        if sample.empty:
            schema[col] = {"type": "boolean", "classes": None}
            continue
        bool_vals = {"true", "false", "1", "0"}
        if all(str(v).lower() in bool_vals for v in sample):
            schema[col] = {"type": "boolean", "classes": None}
        else:
            classes = sorted(str(v) for v in sample.unique()
                             if not str(v).startswith("_invalid:"))
            schema[col] = {"type": "categorical", "classes": classes}
    return schema


def build_html(rows: list[dict], schema: dict, csv_name: str) -> str:
    initial_data = json.dumps(rows, ensure_ascii=False)
    schema_json  = json.dumps(schema, ensure_ascii=False)
    cat_cols  = [c for c, s in schema.items() if s["type"] == "categorical"]
    bool_cols = [c for c, s in schema.items() if s["type"] == "boolean"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tag Editor — {csv_name}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0f1117; color: #e2e8f0;
  height: 100vh; overflow: hidden;
  display: flex; flex-direction: column;
}}

/* ── Header ── */
.header {{
  flex-shrink: 0;
  background: #0f1117; border-bottom: 1px solid #1e2a3a;
  padding: 10px 20px;
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
}}
.header-title {{ font-size: 0.85em; color: #94a3b8; flex: 1; }}
.header-title strong {{ color: #e2e8f0; }}
.btn {{
  padding: 6px 14px; border-radius: 6px; border: none;
  font-size: 0.8em; font-weight: 600; cursor: pointer;
}}
.btn-nav  {{ background: #1e293b; color: #94a3b8; min-width: 80px; }}
.btn-nav:hover {{ background: #334155; color: #e2e8f0; }}
.btn-nav:disabled {{ opacity: .3; cursor: default; }}
.btn-export {{ background: #2563eb; color: white; }}
.btn-export:hover {{ opacity: .85; }}
.btn-reset  {{ background: #374151; color: #d1d5db; }}
.btn-reset:hover {{ background: #4b5563; }}
.counter {{
  font-size: 0.8em; color: #475569; white-space: nowrap;
}}
.counter .idx {{ color: #e2e8f0; font-weight: 700; }}
.counter .edited {{ color: #34d399; }}

/* ── Jump input ── */
.jump-wrap {{ display: flex; align-items: center; gap: 6px; }}
.jump-wrap label {{ font-size: 0.75em; color: #475569; }}
.jump-input {{
  width: 58px; padding: 4px 8px; border-radius: 5px;
  background: #1e293b; border: 1px solid #334155;
  color: #e2e8f0; font-size: 0.8em; text-align: center;
}}

/* ── Filter bar ── */
.filter-bar {{
  flex-shrink: 0;
  background: #111827; border-bottom: 1px solid #1e2a3a;
  padding: 7px 20px;
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
}}
.filter-bar label {{ font-size: 0.75em; color: #64748b; }}
.filter-bar select, .filter-bar input {{
  padding: 4px 8px; border-radius: 5px;
  background: #1e293b; border: 1px solid #334155;
  color: #e2e8f0; font-size: 0.78em;
}}

/* ── Main layout ── */
.main {{
  flex: 1; display: flex; overflow: hidden;
}}

/* ── Image pane ── */
.img-pane {{
  flex: 1; min-width: 0;
  display: flex; align-items: center; justify-content: center;
  background: #080b10; position: relative; overflow: hidden;
}}
.img-pane img {{
  max-width: 100%; max-height: 100%;
  object-fit: contain;
}}
.no-img {{
  color: #374151; font-size: 0.9em; text-align: center;
}}
.img-stem {{
  position: absolute; bottom: 10px; left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,.6); color: #94a3b8;
  font-size: 0.7em; padding: 3px 10px; border-radius: 4px;
  white-space: nowrap;
}}
.edited-badge {{
  position: absolute; top: 12px; right: 12px;
  background: #059669; color: white;
  font-size: 0.7em; font-weight: 700;
  padding: 3px 10px; border-radius: 10px;
  display: none;
}}
.edited-badge.show {{ display: block; }}

/* ── Attributes pane ── */
.attr-pane {{
  width: 360px; flex-shrink: 0;
  background: #111827; border-left: 1px solid #1e2a3a;
  overflow-y: auto; padding: 14px 16px;
  display: flex; flex-direction: column; gap: 16px;
}}

.section-label {{
  font-size: 0.65em; font-weight: 700; letter-spacing: .06em;
  color: #475569; text-transform: uppercase;
  padding-bottom: 6px; border-bottom: 1px solid #1e2a3a;
}}

/* Categorical */
.cat-row {{
  display: flex; align-items: center; gap: 10px; margin-top: 8px;
}}
.cat-label {{
  font-size: 0.78em; color: #64748b; min-width: 85px; flex-shrink: 0;
}}
.cat-select {{
  flex: 1; padding: 6px 10px; border-radius: 6px;
  background: #1e293b; border: 1px solid #334155;
  color: #e2e8f0; font-size: 0.82em; cursor: pointer;
}}
.cat-select.changed {{ border-color: #34d399; color: #34d399; }}
.cat-select.invalid {{ border-color: #f59e0b; color: #f59e0b; }}

/* Boolean grid */
.bool-grid {{
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}}
.toggle {{
  padding: 5px 11px; border-radius: 10px;
  font-size: 0.75em; font-weight: 600;
  cursor: pointer; border: 1px solid transparent;
  user-select: none; transition: all .1s;
}}
.toggle.off {{
  background: #1e293b; color: #94a3b8; border-color: #334155;
}}
.toggle.on {{
  background: #1e3a2f; color: #34d399; border-color: #059669;
}}
.toggle.on.changed  {{ border-color: #4ade80; }}
.toggle.off.changed {{ background: #2d1b1b; color: #f87171; border-color: #dc2626; }}

/* Note */
.note-input {{
  width: 100%; padding: 6px 8px; border-radius: 5px;
  background: #1e293b; border: 1px solid #334155;
  color: #94a3b8; font-size: 0.78em; resize: none;
  font-family: inherit; margin-top: 6px;
}}
.note-input:focus {{ outline: none; border-color: #2563eb; }}

/* ── Keyboard hint ── */
.kb-hint {{
  flex-shrink: 0;
  background: #0a0d14; border-top: 1px solid #1e2a3a;
  padding: 5px 20px;
  font-size: 0.7em; color: #334155;
  display: flex; gap: 20px;
}}
.kb-hint span {{ color: #475569; }}
.kb-hint kbd {{
  display: inline-block;
  background: #1e293b; color: #94a3b8;
  border: 1px solid #334155; border-radius: 3px;
  padding: 1px 5px; font-size: 0.9em;
}}

/* ── Toast ── */
.toast {{
  position: fixed; bottom: 36px; left: 50%; transform: translateX(-50%);
  background: #059669; color: white;
  padding: 9px 20px; border-radius: 8px;
  font-size: 0.85em; font-weight: 600;
  opacity: 0; pointer-events: none; transition: opacity .3s; z-index: 200;
}}
.toast.show {{ opacity: 1; }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-title">Tag Editor &nbsp;/&nbsp; <strong>{csv_name}</strong></div>

  <button class="btn btn-nav" id="btnPrev" onclick="navigate(-1)">← Prev</button>
  <button class="btn btn-nav" id="btnNext" onclick="navigate(1)">Next →</button>

  <div class="jump-wrap">
    <label>Go to</label>
    <input class="jump-input" type="number" min="1" id="jumpInput"
           onchange="jumpTo(this.value)" placeholder="n">
  </div>

  <div class="counter">
    <span class="idx" id="cntIdx">1</span>
    / <span id="cntTotal">0</span>
    &nbsp;|&nbsp;
    <span class="edited" id="cntEdited">0</span> edited
  </div>

  <button class="btn btn-reset" onclick="resetCurrent()">Reset this</button>
  <button class="btn btn-reset" onclick="resetAll()">Reset all</button>
  <button class="btn btn-export" onclick="exportCSV()">Export corrected CSV</button>
</div>

<!-- Filter bar -->
<div class="filter-bar">
  <label>Jump to:</label>
  <select id="filterSel" onchange="applyFilter()">
    <option value="all">All images</option>
    <option value="edited">Edited only</option>
    <option value="unedited">Unedited only</option>
    <option value="invalid">Has _invalid: values</option>
  </select>
  <label>Search name</label>
  <input type="text" id="searchInput" placeholder="stem…"
         oninput="applyFilter()" style="width:160px">
  <span id="filterInfo" style="font-size:.75em;color:#475569;"></span>
</div>

<!-- Main -->
<div class="main">
  <!-- Image -->
  <div class="img-pane" id="imgPane">
    <img id="mainImg" src="" alt="">
    <div class="no-img" id="noImg" style="display:none">Image not found</div>
    <div class="img-stem" id="imgStem"></div>
    <div class="edited-badge" id="editedBadge">EDITED</div>
  </div>

  <!-- Attributes -->
  <div class="attr-pane" id="attrPane"></div>
</div>

<!-- Keyboard hint -->
<div class="kb-hint">
  <span><kbd>←</kbd><kbd>→</kbd> navigate</span>
  <span><kbd>Home</kbd> first &nbsp; <kbd>End</kbd> last</span>
  <span><kbd>E</kbd> export</span>
  <span><kbd>R</kbd> reset current</span>
</div>

<div class="toast" id="toast"></div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────
const ALL_DATA   = {initial_data};
const SCHEMA     = {schema_json};
const CSV_NAME   = "{csv_name}";

let data         = JSON.parse(JSON.stringify(ALL_DATA));
const editedRows = new Set();

// Filtered index — array of indices into data[] currently visible
let filteredIdx  = data.map((_, i) => i);
let cursorF      = 0;   // position within filteredIdx

// ── Navigation ────────────────────────────────────────────────────────────
function currentDataIdx() {{ return filteredIdx[cursorF]; }}

function navigate(delta) {{
  const next = cursorF + delta;
  if (next < 0 || next >= filteredIdx.length) return;
  cursorF = next;
  render();
}}

function jumpTo(n) {{
  const i = parseInt(n) - 1;
  if (i >= 0 && i < filteredIdx.length) {{
    cursorF = i;
    render();
  }}
}}

// ── Filter ────────────────────────────────────────────────────────────────
function applyFilter() {{
  const mode   = document.getElementById("filterSel").value;
  const search = document.getElementById("searchInput").value.toLowerCase();

  filteredIdx = data.reduce((acc, row, i) => {{
    const nameMatch   = !search || row.name.toLowerCase().includes(search);
    const hasInvalid  = Object.values(row.attrs).some(v => String(v).startsWith("_invalid:"));
    const modeMatch =
      mode === "all"      ? true :
      mode === "edited"   ? editedRows.has(i) :
      mode === "unedited" ? !editedRows.has(i) :
      mode === "invalid"  ? hasInvalid : true;
    if (nameMatch && modeMatch) acc.push(i);
    return acc;
  }}, []);

  cursorF = 0;
  document.getElementById("filterInfo").textContent =
    filteredIdx.length + " / " + data.length + " shown";
  render();
}}

// ── Render current image ──────────────────────────────────────────────────
function render() {{
  if (filteredIdx.length === 0) {{
    document.getElementById("imgPane").innerHTML =
      "<div class='no-img'>No images match the current filter.</div>";
    document.getElementById("attrPane").innerHTML = "";
    document.getElementById("cntIdx").textContent   = "0";
    document.getElementById("cntTotal").textContent = "0";
    document.getElementById("btnPrev").disabled = true;
    document.getElementById("btnNext").disabled = true;
    return;
  }}

  const idx = currentDataIdx();
  const row = data[idx];
  const orig = ALL_DATA[idx];
  const isEdited = editedRows.has(idx);

  // Image
  const img    = document.getElementById("mainImg");
  const noImg  = document.getElementById("noImg");
  if (row.img_src) {{
    img.src = row.img_src; img.style.display = "";
    noImg.style.display = "none";
  }} else {{
    img.style.display = "none"; noImg.style.display = "";
  }}
  document.getElementById("imgStem").textContent = row.name;
  document.getElementById("editedBadge").className =
    "edited-badge" + (isEdited ? " show" : "");

  // Counters
  document.getElementById("cntIdx").textContent   = cursorF + 1;
  document.getElementById("cntTotal").textContent = filteredIdx.length;
  document.getElementById("cntEdited").textContent = editedRows.size;
  document.getElementById("jumpInput").value = cursorF + 1;

  // Nav buttons
  document.getElementById("btnPrev").disabled = cursorF === 0;
  document.getElementById("btnNext").disabled = cursorF === filteredIdx.length - 1;

  // Attribute pane
  const pane = document.getElementById("attrPane");
  pane.innerHTML = "";

  // ── Categoricals ─────────────────────────────────────────────────────
  const catEntries = Object.entries(SCHEMA).filter(([,s]) => s.type === "categorical");
  if (catEntries.length) {{
    const sec = document.createElement("div");
    sec.innerHTML = `<div class="section-label">Categorical</div>`;

    catEntries.forEach(([col, s]) => {{
      const val     = row.attrs[col] ?? "";
      const origVal = orig.attrs[col] ?? "";
      const isInv   = String(val).startsWith("_invalid:");
      const dispVal = isInv ? "" : val;
      const changed = val !== origVal;

      const options = ["", ...s.classes]
        .map(c => `<option value="${{c}}" ${{c === dispVal ? "selected" : ""}}>${{c || "— clear —"}}</option>`)
        .join("");

      let cls = "cat-select";
      if (isInv)    cls += " invalid";
      else if (changed) cls += " changed";

      const row_el = document.createElement("div");
      row_el.className = "cat-row";
      row_el.innerHTML = `
        <span class="cat-label">${{col.replace(/_/g," ")}}</span>
        <select class="${{cls}}" data-col="${{col}}"
                onchange="onCat(this,'${{col}}',${{idx}})">
          ${{options}}
        </select>`;
      sec.appendChild(row_el);
    }});
    pane.appendChild(sec);
  }}

  // ── Booleans ─────────────────────────────────────────────────────────
  const boolEntries = Object.entries(SCHEMA).filter(([,s]) => s.type === "boolean");
  if (boolEntries.length) {{
    const sec = document.createElement("div");
    sec.innerHTML = `<div class="section-label">Attributes</div>`;
    const grid = document.createElement("div");
    grid.className = "bool-grid";

    boolEntries.forEach(([col]) => {{
      const raw     = row.attrs[col];
      const origRaw = orig.attrs[col];
      const isOn    = raw === true || raw === "True" || raw === "true";
      const wasOn   = origRaw === true || origRaw === "True" || origRaw === "true";
      const changed = isOn !== wasOn;

      const t = document.createElement("span");
      t.className = `toggle ${{isOn ? "on" : "off"}}${{changed ? " changed" : ""}}`;
      t.textContent = col;
      t.dataset.col = col;
      t.onclick = () => onBool(t, col, idx);
      grid.appendChild(t);
    }});

    sec.appendChild(grid);
    pane.appendChild(sec);
  }}

  // ── Note ─────────────────────────────────────────────────────────────
  const noteSec = document.createElement("div");
  const ta = document.createElement("textarea");
  ta.className = "note-input";
  ta.rows = 2;
  ta.placeholder = "Review note (optional)…";
  ta.value = row.note || "";
  ta.oninput = () => {{ data[idx].note = ta.value; }};
  noteSec.appendChild(ta);
  pane.appendChild(noteSec);
}}

// ── Change handlers ───────────────────────────────────────────────────────
function onCat(select, col, idx) {{
  data[idx].attrs[col] = select.value;
  markEdited(idx);
  const origVal = ALL_DATA[idx].attrs[col] ?? "";
  select.className = select.value !== origVal ? "cat-select changed" : "cat-select";
}}

function onBool(el, col, idx) {{
  const isOn = el.classList.contains("on");
  const newOn = !isOn;
  data[idx].attrs[col] = newOn;
  markEdited(idx);

  const origRaw = ALL_DATA[idx].attrs[col];
  const wasOn   = origRaw === true || origRaw === "True" || origRaw === "true";
  const changed = newOn !== wasOn;

  el.className = `toggle ${{newOn ? "on" : "off"}}${{changed ? " changed" : ""}}`;
  document.getElementById("editedBadge").className = "edited-badge show";
  document.getElementById("cntEdited").textContent = editedRows.size;
}}

function markEdited(idx) {{
  editedRows.add(idx);
  document.getElementById("editedBadge").className = "edited-badge show";
  document.getElementById("cntEdited").textContent = editedRows.size;
}}

// ── Reset ─────────────────────────────────────────────────────────────────
function resetCurrent() {{
  const idx = currentDataIdx();
  data[idx] = JSON.parse(JSON.stringify(ALL_DATA[idx]));
  editedRows.delete(idx);
  render();
}}

function resetAll() {{
  if (!confirm("Reset ALL edits to original VLM values?")) return;
  data = JSON.parse(JSON.stringify(ALL_DATA));
  editedRows.clear();
  render();
  showToast("All edits reset");
}}

// ── Export ────────────────────────────────────────────────────────────────
function exportCSV() {{
  const cols = ["name", ...Object.keys(SCHEMA), "note"];
  const header = cols.join(",");
  const lines = data.map(row => {{
    return cols.map(col => {{
      if (col === "name") return csvCell(row.name);
      if (col === "note") return csvCell(row.note || "");
      const v = row.attrs[col];
      if (v === true  || v === "True")  return "True";
      if (v === false || v === "False" || v === undefined || v === null) return "False";
      return csvCell(String(v));
    }}).join(",");
  }});
  const csv  = [header, ...lines].join("\\n");
  const blob = new Blob([csv], {{type:"text/csv;charset=utf-8;"}});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = CSV_NAME.replace(".csv","_corrected.csv");
  a.click(); URL.revokeObjectURL(url);
  showToast("Exported " + data.length + " rows");
}}

function csvCell(v) {{
  if (v.includes(",") || v.includes('"') || v.includes("\\n"))
    return '"' + v.replace(/"/g,'""') + '"';
  return v;
}}

// ── Keyboard ──────────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {{
  // Don't intercept when typing in an input/select/textarea
  if (["INPUT","SELECT","TEXTAREA"].includes(e.target.tagName)) return;
  if (e.key === "ArrowLeft"  || e.key === "ArrowUp")   {{ navigate(-1); e.preventDefault(); }}
  if (e.key === "ArrowRight" || e.key === "ArrowDown")  {{ navigate(1);  e.preventDefault(); }}
  if (e.key === "Home") {{ cursorF = 0; render(); e.preventDefault(); }}
  if (e.key === "End")  {{ cursorF = filteredIdx.length - 1; render(); e.preventDefault(); }}
  if (e.key === "e" || e.key === "E") exportCSV();
  if (e.key === "r" || e.key === "R") resetCurrent();
}});

// ── Toast ─────────────────────────────────────────────────────────────────
function showToast(msg) {{
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2500);
}}

// ── Init ──────────────────────────────────────────────────────────────────
applyFilter();
</script>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate single-image editable HTML tag reviewer from VLM CSV"
    )
    p.add_argument("--tagged-csv",   type=Path, required=True)
    p.add_argument("--image-dir",    type=Path, required=True)
    p.add_argument("--out-html",     type=Path, required=True)
    p.add_argument("--vlm-config",   type=Path, default=None)
    p.add_argument("--only-invalid", action="store_true")
    p.add_argument("--max-images",   type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.tagged_csv, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from {args.tagged_csv}")

    vlm_cfg = load_vlm_config(args.vlm_config)
    schema  = infer_schema(df, vlm_cfg)
    cat_cols  = [c for c, s in schema.items() if s["type"] == "categorical"]
    bool_cols = [c for c, s in schema.items() if s["type"] == "boolean"]
    print(f"Schema: {len(cat_cols)} categorical, {len(bool_cols)} boolean attributes")

    if args.only_invalid:
        mask = df.apply(
            lambda row: any(str(row[c]).startswith("_invalid:") for c in cat_cols),
            axis=1
        )
        df = df[mask].reset_index(drop=True)
        print(f"Filtered to {len(df)} rows with _invalid: values")

    if args.max_images:
        df = df.head(args.max_images)

    rows = []
    missing = 0
    for _, csv_row in df.iterrows():
        stem     = str(csv_row["name"])
        img_path = find_image(args.image_dir, stem)
        img_src  = img_path.as_uri() if img_path else ""
        if not img_path:
            missing += 1
        attrs = {col: csv_row[col] for col in schema if col in csv_row.index}
        rows.append({"name": stem, "img_src": img_src, "attrs": attrs, "note": ""})

    if missing:
        print(f"  {missing} images not found in {args.image_dir}")

    print("Generating HTML...")
    html = build_html(rows, schema, args.tagged_csv.name)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")

    size_kb = args.out_html.stat().st_size / 1024
    print(f"\nReview page → {args.out_html}  ({size_kb:.0f} KB)")
    print(f"Navigate with ← → arrow keys. Export with E key or button.")

    import webbrowser
    webbrowser.open(args.out_html.as_uri())


if __name__ == "__main__":
    main()
