import pandas as pd
from pathlib import Path

file_path = Path("processed/faa_sdr_clean.csv")

print("Reading dataset...")

# Read only the first 5 rows
df = pd.read_csv(file_path, nrows=5)

print("\nShape of sample:")
print(df.shape)

print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\nSample data:")
print(df.to_string())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))