import sys
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

data_dir = Path('data/raw')
csv_files = list(data_dir.glob('*.csv'))

if not csv_files:
    print("No CSV files found in data/raw/")
    sys.exit(1)
print(f"Files found: {len(csv_files)}")

print("Reading files...")
dfs = pd.read_csv(
    csv_files.pop(), 
    engine='python',
    on_bad_lines='skip',
    encoding='utf-8',
    encoding_errors='ignore'
)
#2,43,55
for csv_file in tqdm(csv_files):
    if csv_file.name in ["Evil_Twin_2.csv", "Evil_Twin_43.csv", "Evil_Twin_55.csv"]:
        continue
    try:
        df = pd.read_csv(
            csv_file, 
            engine='python',
            on_bad_lines='skip',
            encoding='utf-8',
            encoding_errors='ignore'
        )
        dfs = pd.concat([dfs, df], ignore_index=True)
    except Exception as e:
        print(f"  - Error reading {csv_file.name}: {e}")
        continue

print("class imbalance...")
class_counts = dfs["Label"].value_counts()
minority_class = class_counts.idxmin()
minority_size = class_counts.min()

minority_df = dfs[dfs["Label"] == minority_class]
majority_dfs = [
    dfs[dfs["Label"] == cls].sample(n=minority_size, random_state=42)
    for cls in class_counts.index
    if cls != minority_class
]

dfs = pd.concat([minority_df, *majority_dfs]).sample(frac=1, random_state=42)

print("Saving dataset...")
output_path = Path('data/processed/data.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
dfs.to_csv(output_path, index=False, header=True)