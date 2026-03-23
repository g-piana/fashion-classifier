# quick script — no changes to eval_vlm_shoes.py needed yet
import pandas as pd
from pathlib import Path
from taxonomy import load_normalizer, normalize_term

normalizer = load_normalizer(Path("conf/taxonomy/heel_type_norm.yaml"))

df = pd.read_csv("E:/fashion-data/csv/shoes_out_test_toe-heel_01.csv")

# Normalize predictions
df["pred_heel_canonical"] = df["pred_heel_type"].apply(
    lambda x: normalize_term(str(x), normalizer)
)

# Normalize ground truth too — your manual labels may also have variants
df["gt_heel_canonical"] = df["gt_heel_type"].apply(
    lambda x: normalize_term(str(x), normalizer)
)

# Recompute accuracy
match = (df["pred_heel_canonical"] == df["gt_heel_canonical"]).mean()
print(f"Post-normalization accuracy: {match:.3f}")

# See what's still wrong
errors = df[df["pred_heel_canonical"] != df["gt_heel_canonical"]]
print(errors[["name", "gt_heel_canonical", "pred_heel_canonical"]].to_string())

# Only evaluate rows where both sides resolved cleanly
clean = df[
    (df["gt_heel_canonical"] != "_unknown") &
    (df["pred_heel_canonical"] != "_unknown")
]
clean_accuracy = (
    clean["pred_heel_canonical"] == clean["gt_heel_canonical"]
).mean()
print(f"Clean accuracy (both resolved): {clean_accuracy:.3f}  n={len(clean)}")
print(f"Rows excluded (unknown on either side): {len(df) - len(clean)}")