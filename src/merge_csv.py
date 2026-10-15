import pandas as pd


df_1 = pd.read_csv("E:/fashion-data/csv/vlm_shoes_boot_418_embroidery.csv")
df_2 = pd.read_csv("E:/fashion-data/csv/vlm_sequins_01_corrected.csv")
target_csv = "E:/fashion-data/csv/vlm_shoes_boot_454_sequins.csv"
df_2_row_num = df_2.shape[0]
if "note" in df_1.columns:
     if "note.1" not in df_2.columns:
          
          note = df_2_row_num * ['']
          df_2["note.1"] = note
          print("Added column note to csv 2")

df_1_col_num = df_1.shape[1]
df_2_col_num = df_2.shape[1]

if df_1_col_num != df_2_col_num:
    print("Different number of columns")
    missing_in_2 = [c for c in df_1.columns if c not in df_2.columns]
    missing_in_1 = [c for c in df_2.columns if c not in df_1.columns]
    if missing_in_2:
        print(f"Present in 1 but missing in 2: {missing_in_2}")
    if missing_in_1:
        print(f"Present in 2 but missing in 1: {missing_in_1}")
else:
        print(f"Both csv have same number of columns: {df_1.shape[1]}")
        df_row_reindex = pd.concat([df_1, df_2], ignore_index=True)
        
        print(f"Rows csv 1:  {df_1.shape[0]} rows")
        print(f"Rows csv 2:  {df_2.shape[0]} rows")
        print(f"Rows concatenated:  {df_row_reindex.shape[0]} rows in {target_csv}")

        df_row_reindex.to_csv(target_csv, index=False)