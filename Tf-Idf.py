from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import ast
import re
import os

# 1️⃣ Load cleaned dataset
df = pd.read_csv("cleaned_urdu_data.csv", encoding="utf-8-sig")
df["Tokens"] = df["Tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x.split())
df["Text"] = df["Tokens"].apply(lambda tokens: " ".join(tokens))

# 2️⃣ Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(lowercase=False)  # lowercase=False for Urdu words
X_tfidf = vectorizer.fit_transform(df["Text"])

# 3️⃣ Get feature names and clean numbers
feature_names = vectorizer.get_feature_names_out()
cleaned_feature_names = [re.sub(r'\d+', '', w).strip() for w in feature_names]

# 4️⃣ Convert to DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=cleaned_feature_names)
tfidf_df.insert(0, "Headline", df["Headline"])

# 5️⃣ Save CSV
os.makedirs("features", exist_ok=True)
output_file = "features/urdu_tfidf_full_10k_cleaned.csv"
tfidf_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("✅ TF-IDF CSV generated successfully!")
print("Rows (headlines):", len(tfidf_df))
print("Columns (features):", len(cleaned_feature_names))
