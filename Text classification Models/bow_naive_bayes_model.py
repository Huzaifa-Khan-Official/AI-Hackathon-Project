# ===============================
# FILE: bow_logistic_regression_model.py
# Purpose: Text Classification of Urdu News using Bag of Words
# Algorithm: Logistic Regression
# ===============================

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load Bag of Words Features
# -------------------------------
try:
    df = pd.read_csv(
        "features/bag_of_words_10k_features_corrected.csv",
        encoding="utf-8-sig"
    )
except FileNotFoundError:
    print("❌ Bag of Words feature file not found!")
    exit()

# -------------------------------
# Step 2: Load Labels (Categories)
# -------------------------------
try:
    labels_df = pd.read_csv("urdu-news-10k.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ Original dataset not found!")
    exit()

# -------------------------------
# Step 3: Add Category column
# -------------------------------
df["Category"] = labels_df["Category"]

# -------------------------------
# Step 4: Define X and y
# -------------------------------
X = df.drop(columns=["Category"])
y = df["Category"]

print("\n📊 Dataset Info:")
print(f"Total samples: {len(X)}")
print(f"Total classes: {y.nunique()}")

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Step 6: Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)  # Logistic Regression
model.fit(X_train, y_train)
print("\n✅ Logistic Regression model training completed!")

# -------------------------------
# Step 7: Save Model
# -------------------------------
os.makedirs("Text classification Models/models", exist_ok=True)
model_file = "Text classification Models/models/bow_logistic_regression_urdu.pkl"

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Bag of Words + Logistic Regression (Urdu News)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(
    "Text classification Models/results/confusion_matrix_bow_lr.png",
    dpi=300
)
plt.close()

print("✅ Confusion Matrix saved successfully")

# -------------------------------
# Step 11: Predict Full Dataset
# -------------------------------
print("\n⏳ Predicting categories for full 10k dataset...")
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
# Step 14: Save Final CSV (Headline + Category)
# -------------------------------
output_csv = (
    "Text classification Models/results/"
    "final_bow_logistic_headline_category.csv"
)

df_full_pred = pd.DataFrame({
    "Headline": labels_df["Headline"],
    "Predicted_Category": full_pred
})

df_full_pred.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ Final CSV saved with only Headline & Category: '{output_csv}'")
