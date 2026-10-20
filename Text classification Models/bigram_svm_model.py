# ===============================
# FILE: bigram_svm_model.py
# Purpose: Text Classification of Urdu News using Bigrams (N-Grams)
# Algorithm: Support Vector Machine (SVM)
# ===============================

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load pre-engineered bigram CSV
# -------------------------------
try:
    bigram_df = pd.read_csv("features/urdu_bigrams_full_10k_complete.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ Bigram CSV not found!")
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
# Step 3: Combine bigrams per headline
# -------------------------------
# Group bigrams into a single string per headline
bigram_grouped = bigram_df.groupby("Headline")["Bigram"].apply(
    lambda x: ' '.join([str(i) for i in x if pd.notnull(i)])
).reset_index()

# Merge with labels
bigram_grouped = bigram_grouped.merge(labels_df[["Headline", "Category"]], on="Headline")

# Features and labels
X_text = bigram_grouped["Bigram"]  # text features (all bigrams combined)
y = bigram_grouped["Category"]

print("\n📊 Dataset Info:")
print(f"Total samples: {len(X_text)}")
print(f"Total classes: {y.nunique()}")

# -------------------------------
# Step 4: Vectorize Bigrams
# -------------------------------
vectorizer = CountVectorizer(ngram_range=(1, 1))  # Already bigrams, so just unigrams
X = vectorizer.fit_transform(X_text)
print(f"Total bigram features after vectorization: {X.shape[1]}")

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 6: Train SVM Model
# -------------------------------
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)
print("\n✅ SVM model training completed!")

# -------------------------------
# Step 7: Save Model
# -------------------------------
os.makedirs("Text classification Models/models", exist_ok=True)
model_file = "Text classification Models/models/bigram_svm_urdu.pkl"

with open(model_file, "wb") as f:
    pickle.dump(model, f)
print(f"💾 Model saved successfully as '{model_file}'")

# -------------------------------
# Step 8: Prediction on Test Set
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 9: Evaluation
# -------------------------------
print("\n🎯 Test Set Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 10: Confusion Matrix
# -------------------------------
os.makedirs("Text classification Models/results", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - Bigrams + SVM (Urdu News)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(
    "Text classification Models/results/confusion_matrix_bigram_svm.png",
    dpi=300
)
plt.close()
print("✅ Confusion Matrix saved successfully")

# -------------------------------
# Step 11: Predict Full Dataset
# -------------------------------
full_pred = model.predict(X)

# -------------------------------
# Step 12: Accuracy on Full Dataset
# -------------------------------
full_accuracy = accuracy_score(y, full_pred)
print(f"\n🎯 Accuracy on full 10k dataset: {full_accuracy * 100:.2f}%")

# -------------------------------
# Step 13: Summary
# -------------------------------
full_pred_series = pd.Series(full_pred)
category_counts = full_pred_series.value_counts()

print("\n📊 Full Dataset Prediction Summary:")
print(f"Total samples predicted: {len(full_pred_series)}")
print(f"Total unique categories: {category_counts.shape[0]}\n")

for cat, count in category_counts.items():
    print(f"{cat:25} {count}")

# -------------------------------
# Step 14: Save Final CSV (Headline + Predicted Category)
# -------------------------------
output_csv = (
    "Text classification Models/results/"
    "final_bigram_headline_category_svm.csv"
)

df_full_pred = pd.DataFrame({
    "Headline": bigram_grouped["Headline"],
    "Predicted_Category": full_pred
})

df_full_pred.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ Final CSV saved with Headline & Predicted Category: '{output_csv}'")
