emb = ["buckle", "crystal", "flower", "bead", "stud", "hardware", "bow","sequin", "fur", "pearl", "eyelet", "fringe", "mesh-insert","embroidery", "gemstone", "patch", "lace", "chain", "pom-pom","feather", "ribbon", "button"]

emb.sort()
clean_list = " ".join(emb)
print(clean_list)

"""
python src/prepare_tiers.py ` 
--vlm-csv "E:/fashion-data/csv/vlm_shoes_boot_350_corrected.csv" ` 
--category shoes ` 
--embellishments buckle, crystal, flower, bead, stud, hardware, bow, sequin, fur, pearl, eyelet, fringe, mesh-insert, embroidery, gemstone, patch, lace, chain, pom-pom, feather, ribbon, button `
--model-details ankle-strap ankle-wrap single-strap double-strap multi-strap strappy cross-over t-bar sling-back lace-up zip-up slouch `
--categoricals toe_shape heel_type heel_height `
--min-positives 8 ` 
--out-dir "E:/fashion-data/tiers"



python src/prepare_tiers.py `
     --vlm-csv   "E:/fashion-data/csv/vlm_shoes_boot_454_sequins.csv" `
     --category  shoes `
     --embellishments bead bow buckle button chain crystal embroidery eyelet feather flower fringe fur gemstone hardware lace mesh-insert patch pearl pom-pom ribbon sequin stud `
     --model-details ankle-strap ankle-wrap single-strap double-strap multi-strap strappy cross-over t-bar sling-back lace-up zip-up slouch `
     --categoricals toe_shape heel_type heel_height `
     --min-positives 8 `
     --out-dir   "E:/fashion-data/tiers"
"""