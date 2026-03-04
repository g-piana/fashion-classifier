import pandas as pd
df = pd.read_csv("E:/fashion-data/csv/audit_zeroshot.csv")

# See which images were flagged for each attribute
for attr in ["ankle_length", "cropped", "belted", "chest_pocket", "epaulette"]:
    positives = df[df[f"{attr}_pred"] == 1]["name"].tolist()
    print(f"\n{attr}: {len(positives)} positives")
    print("  ", positives[:10])