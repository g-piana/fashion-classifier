import pandas as pd

keep = ["name", "note", "toe_shape", "heel_type", "heel_height",
        # boolean model details
        "ankle-strap", "ankle-wrap", "single-strap", "double-strap", 
        "multi-strap", "strappy", "cross-over", "t-bar", "sling-back", 
        "lace-up", "zip-up", "slouch",
        # approved 22 embellishments (after dropping logo)
        "buckle", "crystal", "flower", "bead", "stud", "hardware", "bow",
        "sequin", "fur", "pearl", "eyelet", "fringe", "mesh-insert",
        "embroidery", "gemstone", "patch", "lace", "chain", "pom-pom",
        "feather", "ribbon", "button"]

#df = pd.read_csv("E:/fashion-data/csv/vlm_shoes_boot_350_corrected.csv")
df = pd.read_csv("E:/fashion-data/csv/vlm_shoes_studs.csv")
print(f"Before: {df.shape[1]} columns")
df = df[[c for c in keep if c in df.columns]]
print(f"After:  {df.shape[1]} columns")
#df.to_csv("E:/fashion-data/csv/vlm_shoes_boot_350_corrected.csv", index=False)
df.to_csv("E:/fashion-data/csv/vlm_shoes_studs_cleaned.csv", index=False)
