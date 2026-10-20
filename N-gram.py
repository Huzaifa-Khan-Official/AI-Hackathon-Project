import pandas as pd
import ast
import os

# Load cleaned dataset
df = pd.read_csv("cleaned_urdu_data.csv", encoding="utf-8-sig")
print("📄 Total headlines loaded:", len(df))

# Convert Tokens column to list safely
def safe_eval_tokens(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x.split()
    except:
        return str(x).split()

df["Tokens"] = df["Tokens"].apply(safe_eval_tokens)

# Generate bigrams for every headline
bigram_rows = []

for idx, row in df.iterrows():
    headline = row["Headline"]
    tokens = row["Tokens"]

    if len(tokens) < 2:
        # Agar words < 2, tab bhi row add karo with Bigram=None
        bigram_rows.append([headline, None])
        continue

    # Generate bigrams
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        bigram_rows.append([headline, bigram])

# Create DataFrame
bigram_df = pd.DataFrame(bigram_rows, columns=["Headline", "Bigram"])

# Save CSV
os.makedirs("features", exist_ok=True)
output_file = "features/urdu_bigrams_full_10k_complete.csv"
bigram_df.to_csv(output_file, index=False, encoding="utf-8-sig")

# Verification
print("✅ BIGRAM generation completed")
print("📊 Total bigrams (rows) in CSV:", len(bigram_df))
print("📄 Sample rows:")
print(bigram_df.head(15))
