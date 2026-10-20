import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import ast

# -------------------------------
# STEP 1: Load Cleaned Data
# -------------------------------
df = pd.read_csv("cleaned_urdu_data.csv", encoding="utf-8-sig")

# -------------------------------
# STEP 2: Convert Tokens to proper text
# -------------------------------
def clean_tokens(token_string):
    if isinstance(token_string, str):
        try:
            tokens = ast.literal_eval(token_string)  # safely parse string list
        except:
            tokens = token_string.split()
        # Remove numbers (100انڈیکس, 2, etc)
        tokens = [t for t in tokens if not t.isdigit() and not t[0].isdigit()]
        return " ".join(tokens)
    return ""

# Apply token cleaning
df["Cleaned_Text"] = df["Tokens"].apply(clean_tokens)

# Drop empty rows if any (important)
df = df[df["Cleaned_Text"].str.strip() != ""]

# -------------------------------
# STEP 3: Bag of Words
# -------------------------------
vectorizer = CountVectorizer(ngram_range=(1,1), max_features=None, lowercase=False)  
# lowercase=False ensures Urdu words remain intact

X = vectorizer.fit_transform(df["Cleaned_Text"])

# -------------------------------
# STEP 4: Feature Names
# -------------------------------
feature_names = vectorizer.get_feature_names_out()
print("Total Features (Unique Words):", len(feature_names))
print("Sample Features:", feature_names[:20])

# -------------------------------
# STEP 5: Convert BoW Matrix to DataFrame
# -------------------------------
bow_df = pd.DataFrame(X.toarray(), columns=feature_names)

# -------------------------------
# STEP 6: Save BoW Features
# -------------------------------
os.makedirs("features", exist_ok=True)
bow_file = "features/bag_of_words_10k_features_corrected.csv"
bow_df.to_csv(bow_file, index=False, encoding="utf-8-sig")
print("✅ Bag of Words feature matrix saved:", bow_file)

# -------------------------------
# STEP 7: Overall Word Frequencies
# -------------------------------
word_freq = np.sum(X.toarray(), axis=0)
freq_df = pd.DataFrame({
    "Word": feature_names,
    "Frequency": word_freq
}).sort_values(by="Frequency", ascending=False)

freq_file = "features/bow_10k_word_frequencies_corrected.csv"
freq_df.to_csv(freq_file, index=False, encoding="utf-8-sig")
print("✅ Overall word frequencies saved:", freq_file)
