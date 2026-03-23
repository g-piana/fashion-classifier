import json
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

input_file  = Path("E:/fashion-data/csv/ls_tasks_shoes_emb_03.json")
output_file = Path("E:/fashion-data/csv/shoes_embellishment_labels_03.csv")


def extract_stem(image_url: str) -> str:
    """
    Handle Label Studio local-file URLs of the form:
        /data/local-files/?d=01-RAW%5Cshoes_emb%5Cstemname.JPG
    Returns just the filename stem (no extension, no path).
    """
    parsed = urlparse(image_url)
    qs = parse_qs(parsed.query)
    if "d" in qs:
        # URL-decode and normalise backslashes → forward slashes
        file_path = unquote(qs["d"][0]).replace("\\", "/")
        return Path(file_path).stem
    # Fallback: treat the whole path as a filename
    return Path(unquote(parsed.path)).stem


with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for task in data:
    annotations = task.get("annotations", [])
    if not annotations or not annotations[0].get("result"):
        # Unannotated task — include with empty embellishment
        rows.append({
            "name": extract_stem(task["data"]["image"]),
            "embellishment": "",
        })
        continue

    ann = annotations[0]["result"]
    result = {r["from_name"]: r["value"] for r in ann}

    embellishment = result.get("embellishment", {}).get("choices", [])

    rows.append({
        "name": extract_stem(task["data"]["image"]),
        "embellishment": "|".join(embellishment),
    })

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)
print(f"Saved {len(df)} rows → {output_file}")
print(df.to_string())
