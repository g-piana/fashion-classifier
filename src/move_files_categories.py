from pathlib import Path
import csv
import shutil
import argparse


source_folder = Path("E:/fashion-data/00-RAW/photo_or_draw_raw/infer")

target_folder= Path("E:/fashion-data/01-RAW/nillab_01/categories")
photo_sketch= Path("E:/fashion-data/csv/predictions_photo_or_draw.csv")


subcat_dic = {
    'flat-shoes': ["ballerina", "mule", "boat-shoe", "derby", "desert", "d-orsay-flat", "driving-shoe", "lace-up", "loafer", "monk-strap", "other-flat-shoes", "oxford", "slip-on", "slipper"],
    'heeled-shoes': ["mule", "d-orsay-pump", "decollete", "lace-up", "loafer-pump", "mary-jane-pump", "pump", "slingback", "wedge-pump"],
    'sandals': ['slider', 'sandal', 'wedge-sandal', 'mule', 'thong', 'sandal-boot' ],
    'boots-and-booties': ["boot", "over-the-knee", "ankle-boot", "shoe-boot", "lace-up-boot"],
    'sneakers': ["high-top-sneaker", "sneaker"]
}
    # 'sneakers': ["high-top-sneaker", "low-top-sneaker", "tennis-shoe", "sneaker"]
    # 'boots-and-booties': ["over-the-knee", "ankle-boot", "shoe-boot", "snow-boot"],



def move_files(category_to_move, max_per_subcat):

    pred_csv = Path(f"E:/fashion-data/csv/nillab/{category_to_move}.csv")
    if pred_csv.exists() == False:
        print(f"Labels file to move {category_to_move}.csv does not exist")
        exit(1)

    subcat_list = subcat_dic[category_to_move]


    start_id = 0 # Skip the first as already processed
    complete_set = False
    init_cat = True
    count_dic = {sc: 0 for sc in subcat_list}
    full_list = []

    sketch_list = []
    with open(photo_sketch, "r", encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for id, item in enumerate(reader):
            if item['prediction'] == 'drawing':
                sketch_list.append(item['name'])

    print(f"Found a total of {len(sketch_list)} sketches that will be skipped")

    with open(pred_csv, "r", encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for id, item in enumerate(reader):
            
            if id < start_id or item['image_stem'] in sketch_list: # skip sketches
                print(f"Skipping sketch {item['image_stem']}")
                continue

            #print(id, item)
            source = source_folder / f"{item['image_stem']}.jpg"
            category = item['category']
            if init_cat:
                init_cat = False
                cat_path = target_folder / category
                cat_path.mkdir(parents=True, exist_ok=True)

            sub_cat =  item['subcategory']

            if sub_cat:
                dest_subfolder = cat_path / sub_cat
                dest_subfolder.mkdir(exist_ok=True, parents=True)
            else:
                print("Wrong prediction at sub_cat", id)

            destination = dest_subfolder / f"{item['image_stem'].replace("+","plus")}.jpg"
            if sub_cat in count_dic.keys() and count_dic[sub_cat] < max_per_subcat:
                if item['image_stem'] not in full_list:
                    full_list.append( item['image_stem'])
                    count_dic[sub_cat] += 1

                    if source.exists():
                        #print(f"Will copy {source} to {destination}")
                        dest = shutil.copy(source, destination)
            
            complete_set = True
            for key, item in count_dic.items():
                if item < max_per_subcat:
                    complete_set = False
            
            if complete_set:
                break
    print(f"Read a total of {id} files")
    for key, item in count_dic.items():
        print(f"{key:<15} {item}")

    return

def main():
    p = argparse.ArgumentParser(
        description="Move files organized in categories/subcategories folder structure based on input csv"
    )
    p.add_argument("--category",  type=str, required=True,
                   help="Category to move - simple csv naming e.g. heeled-shoes (not heeled-shoes-women)")
    p.add_argument("--max_number", type=int, default=2,
                   help="Number of images per subcategory")
    args = p.parse_args()

    if not args.category:
        p.error(f"--category must be specified. It must be one of {subcat_dic.keys()}.")
    else:
        max_per_subcat = args.max_number
        move_files(args.category, max_per_subcat)

if __name__ == "__main__":
    main()