import json
import pandas as pd
from pathlib import Path 
input_file = Path("E:/fashion-data/csv/ls_tasks_shoes_emb.json")
#input_file = Path("E:/fashion-data/csv/ls_tasks_shoes_02.json")
output_file = Path("E:/fashion-data/csv/shoes_labels_emb.csv")
with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)

rows = []
for task in data:
    ann = task["annotations"][0]["result"]
    result = {r["from_name"]: r["value"] for r in ann}
    
    # construction is a list, join with pipe
    construction = result.get("construction", {}).get("choices", [])
    
    rows.append({
        "name": Path(task["data"]["image"]).stem,
        "toe_shape":    result.get("toe_shape", {}).get("choices", [""])[0],
        "heel_type":    result.get("heel_type", {}).get("choices", [""])[0],
        "construction": "|".join(construction),
    })

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)
print(df)