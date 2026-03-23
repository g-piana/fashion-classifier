import pandas as pd
from pathlib import Path
from taxonomy import load_normalizer, normalize_term

normalizer = load_normalizer(Path("conf/taxonomy/heel_type_norm.yaml"))

# Load NEW ground truth (relabeled)
gt_df = pd.read_csv("E:/fashion-data/csv/shoes_labels.csv")[["name", "heel_type"]]
gt_df = gt_df.rename(columns={"heel_type": "gt_heel_type"})

# Load Claude predictions from eval run
pred_df = pd.read_csv("E:/fashion-data/csv/shoes_out_test_toe-heel_01.csv")[["name", "pred_heel_type"]]

# Merge on name — only evaluate images present in both
df = pd.merge(gt_df, pred_df, on="name", how="inner")
print(f"Matched rows: {len(df)}  (gt={len(gt_df)}, pred={len(pred_df)})")

# Normalize both sides
df["gt_canonical"]   = df["gt_heel_type"].apply(lambda x: normalize_term(str(x), normalizer))
df["pred_canonical"] = df["pred_heel_type"].apply(lambda x: normalize_term(str(x), normalizer))

# ── Overall accuracy ───────────────────────────────────────────────────────
match = (df["gt_canonical"] == df["pred_canonical"]).mean()
print(f"\nPost-normalization accuracy (all rows): {match:.3f}")

# ── Clean accuracy — exclude _unknown on either side ──────────────────────
clean = df[
    (df["gt_canonical"] != "_unknown") &
    (df["pred_canonical"] != "_unknown")
]
clean_match = (clean["gt_canonical"] == clean["pred_canonical"]).mean()
print(f"Clean accuracy (no _unknown):           {clean_match:.3f}  (n={len(clean)})")
print(f"Rows excluded (_unknown on either side): {len(df) - len(clean)}")

# ── Per-class breakdown ────────────────────────────────────────────────────
print("\nPer-class accuracy:")
for cls in sorted(df["gt_canonical"].unique()):
    if cls == "_unknown":
        continue
    subset = df[df["gt_canonical"] == cls]
    acc = (subset["pred_canonical"] == cls).mean()
    n = len(subset)
    bar = "█" * int(acc * 20)
    print(f"  {cls:<20} {acc:.2f}  n={n:>3}  {bar}")

# ── Errors ────────────────────────────────────────────────────────────────
errors = df[df["gt_canonical"] != df["pred_canonical"]]
print(f"\nErrors: {len(errors)} / {len(df)}")
print(errors[["name", "gt_heel_type", "gt_canonical", "pred_heel_type", "pred_canonical"]].to_string())

# ── Unknown diagnostics ───────────────────────────────────────────────────
unknown_gt = df[df["gt_canonical"] == "_unknown"]["gt_heel_type"].value_counts()
if not unknown_gt.empty:
    print("\nGT terms hitting _unknown (add to YAML):")
    print(unknown_gt.to_string())

unknown_pred = df[df["pred_canonical"] == "_unknown"]["pred_heel_type"].value_counts()
if not unknown_pred.empty:
    print("\nPred terms hitting _unknown (add to YAML):")
    print(unknown_pred.to_string())

# ── Top confusions ────────────────────────────────────────────────────────
if not errors.empty:
    confusions = (
        errors.groupby(["gt_canonical", "pred_canonical"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    print("\nTop confusions (gt → pred):")
    print(confusions.to_string(index=False))
