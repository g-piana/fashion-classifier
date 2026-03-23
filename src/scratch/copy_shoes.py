import csv
from pathlib import Path
import shutil

num_to_transfer = 60

image_dir = Path("E:/fashion-data/01-RAW/shoes")
full_image_dir = Path("E:/OBID/NILLAB/02-RAW/data_01")
input_csv = Path("E:/OBID/NILLAB/01-CSV/TEST_shoes_var_28.csv")
output_csv = Path("E:/fashion-data/csv/shoes_label_test_01.csv")
mod_det_target = ["strappy", "wedge", "single-strap", "ankle-strap", "ankle-tie-strap", "sling-back"]
image_list = image_dir.glob("*.jpg")
prod_dict = {}
for img_name in image_list:
    prod_dict[img_name.stem] = ""

print(f"Already in destination: {len(prod_dict.keys())}")

full_image_list = full_image_dir.glob("*.jpg")

full_image_dict = {f.stem: f for f in full_image_list}

print(f"Total files in source: {len(full_image_dict.keys())}")
skipped = 0
label_dict = {}
with open(input_csv, "r", encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'\n\nColumn names are {", ".join(row)}')
        else:
            csv_name = row["name"]
            csv_mod_det = row["Model_Detail"].lower()
            csv_mod_det_list = csv_mod_det.split("_")
            md_target = False
            for md in csv_mod_det_list:
                if md in mod_det_target:
                    md_target = True
                    break
            if md_target:
                label_dict[csv_name] = csv_mod_det
            else:
                skipped += 1
        line_count += 1
        
        
print (f"Total annotated {len(label_dict.keys())}")
print(f"skipped {skipped}")

num_transf = 0
print("\nList to transfer")
for key, item in label_dict.items():
    if key in full_image_dict.keys() and key not in prod_dict.keys():
        #print(f"{key} - {full_image_dict[key]}")
        dest = shutil.copy(full_image_dict[key], image_dir)
        print(f"{key} copied with md {item}")
        num_transf += 1
    if num_transf >= num_to_transfer:
        break
