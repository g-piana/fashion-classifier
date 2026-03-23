import csv
from pathlib import Path

image_dir = Path("E:/fashion-data/01-RAW/shoes")

input_csv = Path("E:/OBID/NILLAB/01-CSV/TEST_shoes_var_28.csv")
output_csv = Path("E:/fashion-data/csv/shoes_label_test_01.csv")

image_list = image_dir.glob("*.jpg")
prod_dict = {}
for img_name in image_list:
    prod_dict[img_name.stem] = ""
    print(img_name.stem)

not_found = []
with open(input_csv, "r", encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'\n\nColumn names are {", ".join(row)}')
        
        csv_name = row["name"]
        csv_mod_det = row["Model_Detail"].lower()
        if csv_name in prod_dict.keys():
            line_count += 1
            prod_dict[csv_name] = csv_mod_det.replace(" ", "-")
        
print (prod_dict)
not_found = [img for img in prod_dict.keys() if prod_dict[img] == '']
print (f"\n Not found: {not_found}")
with open(output_csv, "w", encoding='utf-8', newline='') as out_csv:
    field_names = ['name', 'attributes']
    writer = csv.writer(out_csv)
    writer.writerow(field_names)
    for key, item in prod_dict.items():
        if item == 'no-spec':
            fixed_item = ''
        elif "_" in item:
            md_set = item.split("_")
            fixed_item = "|".join(md_set)
        else:
            fixed_item = item
        writer.writerow([key, fixed_item])

md_list = []
for key, item in prod_dict.items():
    md_set = item.split("_")
    for md in md_set:
        if md not in md_list and md != 'no-spec':
            md_list.append(md)

print(md_list)