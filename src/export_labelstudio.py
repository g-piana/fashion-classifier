"""
export_labelstudio.py
=====================
Converts a Label Studio JSON export into the training CSV format
expected by train.py (label_type: multi).

Usage
-----
    python src/export_labelstudio.py \
        --ls-json  E:/fashion-data/csv/labelstudio_export.json \
        --out-csv  E:/fashion-data/csv/labels_model_details.csv \
        --split    model_details
"""
from __future__ import annotations
import argparse, json, urllib.parse
from pathlib import Path
import pandas as pd

SPLIT_WIDGETS = {
    "model_details":  ["model_details"],
    "embellishments": ["embellishments"],
    "all":            ["model_details", "embellishments"],
}
# ALL_MODEL_DETAILS  = {"ankle_length","asymmetric","belted","chest_pocket","cropped","drawstring","epaulette"}
# ALL_EMBELLISHMENTS = {"embroidery","eyelet","feather","flower","fringe"}

ALL_MODEL_DETAILS  = {"asymmetric","belted","cropped","epaulette"}
ALL_EMBELLISHMENTS = {"feather","flower","fringe"}

def extract_stem(task: dict) -> str:
    """Extract image filename stem from a Label Studio task.

    Label Studio local file storage uses URLs like:
        /data/local-files/?d=01-RAW/jackets_img/biker/file.jpg
    The actual filename is in the ?d= query parameter, not the path.
    Falls back to uploaded-file URLs (/data/upload/.../file.jpg).
    """
    name = task.get("data", {}).get("name", "")
    if name and name != "local-files":
        return name
    url = task.get("data", {}).get("image", "")
    if not url:
        return ""
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "d" in qs:
        return Path(qs["d"][0]).stem   # local files storage
    return Path(urllib.parse.urlparse(url).path).stem   # uploaded files


def parse_task(task: dict, widget_names: list) -> dict | None:
    name = extract_stem(task)
    annotations = task.get("annotations", [])
    if not annotations:
        return None
    results = annotations[0].get("result", [])
    labels, quality = set(), "ok"
    for r in results:
        fn      = r.get("from_name", "")
        choices = r.get("value", {}).get("choices", [])
        if fn in widget_names:
            labels.update(choices)
        elif fn == "quality" and choices:
            quality = choices[0]
    return {"name": name, "attributes": "_".join(sorted(labels)), "quality": quality}


def export(ls_json: Path, out_csv: Path, split: str) -> None:
    widget_names = SPLIT_WIDGETS[split]
    tasks = json.loads(ls_json.read_text(encoding="utf-8"))
    print(f"Tasks loaded: {len(tasks)}")
    rows, skip_ann, skip_rej = [], 0, 0
    for task in tasks:
        p = parse_task(task, widget_names)
        if p is None:
            skip_ann += 1; continue
        if p["quality"] == "reject":
            skip_rej += 1; continue
        rows.append(p)
    df = pd.DataFrame(rows, columns=["name", "attributes", "quality"])
    print(f"Annotated: {len(df)}  |  no_annotation: {skip_ann}  |  rejected: {skip_rej}")
    all_labels = (ALL_MODEL_DETAILS if split == "model_details"
                  else ALL_EMBELLISHMENTS if split == "embellishments"
                  else ALL_MODEL_DETAILS | ALL_EMBELLISHMENTS)
    print("\nPositive counts:")
    for lbl in sorted(all_labels):
        count = df["attributes"].str.contains(lbl).sum()
        print(f"  {lbl:<18} {count:>5}  ({100*count/max(len(df),1):.1f}%)")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV written -> {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ls-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--split", choices=["model_details","embellishments","all"],
                   default="model_details")
    a = p.parse_args()
    export(a.ls_json, a.out_csv, a.split)

if __name__ == "__main__":
    main()