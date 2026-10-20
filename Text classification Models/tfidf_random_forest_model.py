# ===============================
# FILE: tfidf_random_forest_model.py
# Purpose: Text Classification of Urdu News using TF-IDF Features
# Algorithm: Random Forest
# ===============================

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load TF-IDF feature CSV
# -------------------------------
try:
    tfidf_df = pd.read_csv("features/urdu_tfidf_full_10k_cleaned.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ TF-IDF CSV not found!")
    exit()

# -------------------------------
# Step 2: Load labels (Categories)
# -------------------------------
try:
    labels_df = pd.read_csv("urdu-news-10k.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ Original dataset not found!")
    exit()

# -------------------------------
# Step 3: Prepare Features and Labels
# -------------------------------
# Drop Headline column from features
X = tfidf_df.drop(columns=["Headline"])
y = labels_df["Category"]

print("\n📊 Dataset Info:")
print(f"Total samples: {X.shape[0]}")
print(f"Total features: {X.shape[1]}")
print(f"Total classes: {y.nunique()}")

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 5: Train Random Forest Model
# -------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("\n✅ Random Forest model training completed!")

# -------------------------------
# Step 6: Save Model
# -------------------------------
os.makedirs("Text classification Models/models", exist_ok=True)
model_file = "Text classification Models/models/tfidf_rf_urdu.pkl"

with open(model_file, "wb") as f:
    pickle.dump(model, f)
print(f"💾 Model saved successfully as '{model_file}'")

# -------------------------------
# Step 7: Prediction on Test Set
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 8: Evaluation
# -------------------------------
print("\n🎯 Test Set Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 9: Confusion Matrix
# -------------------------------
os.makedirs("Text classification Models/results", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - TF-IDF + Random Forest (Urdu News)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(
    "Text classification Models/results/confusion_matrix_tfidf_rf.png",
    dpi=300
)
plt.close()
print("✅ Confusion Matrix saved successfully")

# -------------------------------
# Step 10: Predict Full Dataset
# -------------------------------
full_pred = model.predict(X)

# -------------------------------
# Step 11: Accuracy on Full Dataset
# -------------------------------
full_accuracy = accuracy_score(y, full_pred)
print(f"\n🎯 Accuracy on full 10k dataset: {full_accuracy * 100:.2f}%")

# -------------------------------
# Step 12: Summary
# -------------------------------
full_pred_series = pd.Series(full_pred)
category_counts = full_pred_series.value_counts()

print("\n📊 Full Dataset Prediction Summary:")
print(f"Total samples predicted: {len(full_pred_series)}")
print(f"Total unique categories: {category_counts.shape[0]}\n")

for cat, count in category_counts.items():
    print(f"{cat:25} {count}")

# -------------------------------
# Step 13: Save Final CSV (Headline + Predicted Category)
# -------------------------------
output_csv = (
    "Text classification Models/results/"
    "final_tfidf_headline_category_rf.csv"
)

df_full_pred = pd.DataFrame({
    "Headline": tfidf_df["Headline"],
    "Predicted_Category": full_pred
})

df_full_pred.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ Final CSV saved with Headline & Predicted Category: '{output_csv}'")
