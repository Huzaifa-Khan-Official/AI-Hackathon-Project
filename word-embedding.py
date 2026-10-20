# ===============================
# FILE: Urdu_TopK_Similar_Complete.py
# Har headline ke liye top 10 similar headlines + Pairwise comparison
# ===============================
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import re
from tqdm import tqdm
import os

print("🚀 Starting process...")

# -------------------------------
# STEP 1: Load Data
# -------------------------------
try:
    df = pd.read_csv("cleaned_urdu_data.csv", encoding='utf-8-sig')
    print(f"✅ {len(df)} headlines load ho gayi")
except FileNotFoundError:
    print("❌ Error: cleaned_urdu_data.csv nahi mili!")
    exit()

# -------------------------------
# STEP 2: Tokens ko text mein convert karo
# -------------------------------
df["Tokens"] = df["Tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def tokens_to_text(tokens):
    return " ".join([re.sub(r'\d+', '', w).strip() for w in tokens if w.strip() != ""])

df["Cleaned_Text"] = df["Tokens"].apply(tokens_to_text)
print("✅ Text ready ho gaya")

# -------------------------------
# STEP 3: Bag of Words (BoW)
# -------------------------------
vectorizer = CountVectorizer(lowercase=False, max_features=3000)  # 3000 words tak limit
X = vectorizer.fit_transform(df["Cleaned_Text"])
print(f"✅ BoW matrix ban gaya: {X.shape}")

# -------------------------------
# STEP 4: Top-10 Similar Headlines nikalo
# -------------------------------
print("⏳ Top-10 similar headlines nikal rahe hain...")

results = []
K = 10  # Kitne similar headlines chahiye

for i in tqdm(range(len(df)), desc="Processing"):
    # Ek headline ki saari headlines se distance nikalo
    distances = euclidean_distances(X[i], X).flatten()
    
    # Sabse chhoti distances ke indices nikalo (khud ko chhod kar)
    top_k_indices = np.argsort(distances)[1:K+1]
    
    # Similar headlines ke names
    similar_headlines = df.iloc[top_k_indices]["Headline"].tolist()
    similar_distances = distances[top_k_indices].tolist()
    
    results.append({
        "Headline": df.iloc[i]["Headline"],
        "Similar_1": similar_headlines[0] if len(similar_headlines) > 0 else "",
        "Distance_1": similar_distances[0] if len(similar_distances) > 0 else "",
        "Similar_2": similar_headlines[1] if len(similar_headlines) > 1 else "",
        "Distance_2": similar_distances[1] if len(similar_distances) > 1 else "",
        "Similar_3": similar_headlines[2] if len(similar_headlines) > 2 else "",
        "Distance_3": similar_distances[2] if len(similar_distances) > 2 else "",
        "Similar_4": similar_headlines[3] if len(similar_headlines) > 3 else "",
        "Distance_4": similar_distances[3] if len(similar_distances) > 3 else "",
        "Similar_5": similar_headlines[4] if len(similar_headlines) > 4 else "",
        "Distance_5": similar_distances[4] if len(similar_distances) > 4 else "",
    })

# -------------------------------
# STEP 5: CSV 1 - Top Similar Headlines
# -------------------------------
os.makedirs("features", exist_ok=True)

df_results = pd.DataFrame(results)
output_file_1 = "features/top10_similar_headlines.csv"
df_results.to_csv(output_file_1, index=False, encoding='utf-8-sig')
print(f"✅ File 1 saved: {output_file_1}")

# -------------------------------
# STEP 6: CSV 2 - Pairwise Distance Matrix (Tumhari image wali)
# -------------------------------
print("⏳ Creating pairwise distance matrix...")

# Define threshold
THRESHOLD = 2.0  # Distance < 2.0 = "Very similar", else "Not similar"

pairwise_results = []

# First 100 headlines ka sample (manageable size)
sample_size = 500

for i in tqdm(range(sample_size), desc="Pairwise comparison"):
    for j in range(i+1, sample_size):  # i+1 se start to avoid duplicate pairs
        headline_1 = df.iloc[i]["Headline"]
        headline_2 = df.iloc[j]["Headline"]
        
        distance = euclidean_distances(X[i].reshape(1, -1), X[j].reshape(1, -1))[0][0]
        
        similarity = "Very similar" if distance < THRESHOLD else "Not similar"
        
        pairwise_results.append({
            "Sentence Pair": f"S{i+1} - S{j+1}",
            "Headline_1": headline_1,
            "Headline_2": headline_2,
            "Euclidean Distance": round(distance, 3),
            "Similarity": similarity
        })

df_pairwise = pd.DataFrame(pairwise_results)
output_file_2 = "features/pairwise_distance_similarity.csv"
df_pairwise.to_csv(output_file_2, index=False, encoding='utf-8-sig')
print(f"✅ File 2 saved: {output_file_2}")

# -------------------------------
# STEP 7: CSV 3 - Summary Statistics
# -------------------------------
print("⏳ Creating summary statistics...")

summary_data = {
    "Metric": [
        "Total Headlines",
        "Total Unique Words (Features)",
        "Average Distance (Top-1 Similar)",
        "Threshold for 'Very Similar'",
        "Total Pairwise Comparisons (Sample)"
    ],
    "Value": [
        len(df),
        X.shape[1],
        round(df_results["Distance_1"].astype(float).mean(), 3),
        THRESHOLD,
        len(pairwise_results)
    ]
}

df_summary = pd.DataFrame(summary_data)
output_file_3 = "features/similarity_summary.csv"
df_summary.to_csv(output_file_3, index=False, encoding='utf-8-sig')
print(f"✅ File 3 saved: {output_file_3}")

# -------------------------------
# FINAL SUCCESS MESSAGE
# -------------------------------
print("\n" + "="*50)
print("🎉 ALL FILES CREATED SUCCESSFULLY!")
print("="*50)
print(f"📁 File 1: {output_file_1}")
print(f"   → Top-10 similar headlines for each headline")
print(f"   → Total rows: {len(df_results)}")
print(f"\n📁 File 2: {output_file_2}")
print(f"   → Pairwise distance comparison (first 100 headlines)")
print(f"   → Total rows: {len(pairwise_results)}")
print(f"\n📁 File 3: {output_file_3}")
print(f"   → Summary statistics")
print("="*50)