import json
from pathlib import Path

image_dir = Path("E:/fashion-data/01-RAW/shoes")
tasks = [
    {"data": {"image": f"/data/local-files/?d=01-RAW/shoes/{p.name}"}}
    for p in sorted(image_dir.glob("*.jpg"))
]
Path("E:/fashion-data/csv/ls_tasks_shoes.json").write_text(
    json.dumps(tasks, indent=2)
)
print(f"{len(tasks)} tasks written")