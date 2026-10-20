# =============================== 
# FILE: onehot_naive_bayes_model.py
# Purpose: Text Classification of Urdu News using One-Hot Encoding
# Algorithm: Multinomial Naive Bayes
# ===============================

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load One-Hot Features
# -------------------------------
try:
    df = pd.read_csv("features/urdu_onehot_features.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ One-hot feature file not found!")
    exit()

# -------------------------------
# Step 2: Load Labels (Categories)
# -------------------------------
try:
    labels_df = pd.read_csv("urdu-news-10k.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ Original dataset not found!")
    exit()

# ⚠️ Make sure column name is correct
df["Category"] = labels_df["Category"]

# -------------------------------
# Step 3: Define X and y
# -------------------------------
X = df.drop(columns=["Headline", "Category"])
y = df["Category"]

print("\n📊 Dataset Info:")
print(f"Total samples: {len(X)}")
print(f"Total classes: {y.nunique()}")

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Step 5: Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)
print("\n✅ Model training completed!")

# -------------------------------
# Step 6: Save Model as .pkl
# -------------------------------
os.makedirs("Text classification Models/models", exist_ok=True)
model_file = "Text classification Models/models/onehot_naive_bayes_urdu.pkl"
with open(model_file, "wb") as f:
    pickle.dump(model, f)
print(f"💾 Model saved successfully as '{model_file}'")

# -------------------------------
# Step 7: Prediction on Test Set
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 8: Evaluation on Test Set
# -------------------------------
print("\n🎯 Test Set Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 9: Confusion Matrix on Test Set
# -------------------------------
os.makedirs("Text classification Models/results", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - One-Hot + Naive Bayes (Urdu News)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("Text classification Models/results/confusion_matrix_onehot_nb.png", dpi=300)
plt.close()
print(f"✅ Confusion Matrix saved in 'Text classification Models/results/confusion_matrix_onehot_nb.png'")

# -------------------------------
# Step 10: Predict Full 10k Dataset
# -------------------------------
print("\n⏳ Predicting categories for full 10k dataset...")
full_pred = model.predict(X)

# -------------------------------
# Step 11: Accuracy on Full Dataset
# -------------------------------
full_accuracy = accuracy_score(y, full_pred)
print(f"\n🎯 Accuracy on full 10k dataset: {full_accuracy * 100:.2f}%")

# -------------------------------
# Clean Terminal Summary for Full Dataset
# -------------------------------
full_pred_series = pd.Series(full_pred)
category_counts = full_pred_series.value_counts()

print("\n📊 Full Dataset Prediction Summary:")
print(f"Total samples predicted: {len(full_pred_series)}")
print(f"Total unique categories: {category_counts.shape[0]}")
print("\nCategories and their counts:")
for cat, count in category_counts.items():
    print(f"{cat:25} {count}")   # neat alignment

# -------------------------------
# Save full predictions CSV
# -------------------------------
output_csv = "Text classification Models/results/predicted_full_onehot_news.csv"
df_full_pred = pd.DataFrame({
    "Headline": df["Headline"],
    "Predicted_Category": full_pred
})
df_full_pred.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ Full 10k predictions saved in '{output_csv}'")
