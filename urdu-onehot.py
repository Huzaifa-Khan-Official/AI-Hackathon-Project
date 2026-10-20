# ===============================
# FILE: urdu-onehot.py
# Purpose: Convert Urdu text tokens into One-Hot Encoded features (Excel-friendly)
# ===============================

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os
import re

# -------------------------------
# Step 1: Load Cleaned Data
# -------------------------------
try:
    df = pd.read_csv("cleaned_urdu_data.csv", encoding='utf-8-sig')
except FileNotFoundError:
    print("❌ Error: cleaned_urdu_data.csv not found!", flush=True)
    exit()

print(f"📄 Total headlines loaded: {len(df)}", flush=True)

# Tokens column ko list me convert karo
df["Tokens"] = df["Tokens"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# -------------------------------
# Step 2: Initialize One-Hot Encoder
# -------------------------------
mlb = MultiLabelBinarizer(sparse_output=True)  # sparse matrix for memory efficiency

print("⏳ Starting optimized One-Hot Encoding...", flush=True)

# Fit and transform tokens
X_onehot = mlb.fit_transform(df["Tokens"])

# Column names = unique words
feature_names = mlb.classes_

print(f"📊 Total unique words (features): {len(feature_names)}", flush=True)

# -------------------------------
# Step 3: Convert to DataFrame (Excel-friendly)
# -------------------------------
df_onehot = pd.DataFrame.sparse.from_spmatrix(
    X_onehot,
    columns=[str(re.sub(r'\d+', '', w)).strip() for w in feature_names]  # Remove numbers from column names
)

# Combine headline and encoded features
df_final = pd.concat([df[["Headline"]], df_onehot], axis=1)

# Output folder
os.makedirs("features", exist_ok=True)
output_file = "features/urdu_onehot_features.csv"

# Remove file if already exists to avoid PermissionError
if os.path.exists(output_file):
    os.remove(output_file)

# Save CSV
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

# -------------------------------
# Step 4: Print Success
# -------------------------------
print(f"✅ One-Hot Encoded features saved successfully as '{output_file}'", flush=True)
print("🎉 Done! Column headers are clean and ready for Excel or ML tasks.", flush=True)
