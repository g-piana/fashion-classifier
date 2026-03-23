import csv
from pathlib import Path
import json

tagged_file = Path("E:/fashion-data/csv/shoes_full_tagged.csv")
json_tagged_file = Path("E:/fashion-data/csv/json_shoes_full_tagged.json")

# attribute types
heel_toe = ["toe_shape","heel_type"]
md_con = ["strappy","ankle-strap","ankle-wrap","sling-back","t-bar","cross-over","platform","lace-up","zip-up","slouch"]
embell = ["bead","bow","buckle","chain","crystal","embroidery","eyelet","feather","flower","fringe","fur","hardware","lace","mesh-insert","patch","pearl","pom-pom","ribbon","sequin","stripe","stud","tassel"]
con_dic = {}

with open(tagged_file, "r", encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    for id, row in enumerate(reader):
        try:
            for item, value in row.items():
                if item == "name":
                    img_name = value
                    con_dic[value] = {
                        'model_details': [],
                        'embellishments': []
                    }
                elif item in heel_toe:
                    con_dic[img_name]['model_details'].append(value)
                elif item in md_con: # booleans -> use key if True
                    if eval(value):
                        con_dic[img_name]['model_details'].append(item)
                elif item in embell: # booleans -> use key if True
                    if eval(value):
                        con_dic[img_name]['embellishments'].append(item)
        except:
            import pdb
            pdb.set_trace()
            
with open(json_tagged_file, "w") as fp:
    json.dump(con_dic, fp, indent=4)
