from pathlib import Path
import csv
import shutil
source_folder = Path("E:/fashion-data/00-RAW/photo_or_draw_raw/infer")
pred_csv = Path("E:/fashion-data/csv/predictions_photo_or_draw.csv")
target_folder= Path("E:/fashion-data/01-RAW/nillab_01")


with open(pred_csv, "r", encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    for id, item in enumerate(reader):
        #print(id, item)
        source = source_folder / f"{item['name']}.jpg"
        
        if item['prediction'] == 'photo':
            dest_subfolder = target_folder / "photo"
        elif item['prediction'] == 'drawing':
            dest_subfolder = target_folder / "sketch"
        else:
            print("Wrong prediction at id", id)

        destination = dest_subfolder / f"{item['name'].replace("+","-plus")}.jpg"


        if source.exists():
            #print(f"Will copy {source} to {destination}")
            dest = shutil.copy(source, destination)
