import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import matplotlib.ticker as ticker
import numpy as np
from wordcloud import WordCloud
from bidi.algorithm import get_display
import arabic_reshaper
import platform
import re

# Font path function for cross-platform compatibility
def get_font_path():
    system = platform.system()
    if system == 'Windows':
        return 'C:/Windows/Fonts/arial.ttf'
    elif system == 'Darwin':  # macOS
        return '/System/Library/Fonts/Supplemental/Arial.ttf'
    else:  # Linux
        return '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

# -------------------------------
# STEP 1: TOKENIZATION PROCESS
# -------------------------------
print("📝 Starting Tokenization Process...")

# Load 10k dataset
try:
    df_original = pd.read_csv("urdu-news-10k.csv")
except FileNotFoundError:
    print("❌ Error: urdu-news-10k.csv not found!")
    exit()

# Tokenization function (simple word splitting)
def tokenize_urdu_text(text):
    """
    Urdu text ko tokens (words) mein tod deta hai
    """
    if pd.isna(text):
        return []
    # Split by whitespace and remove empty strings
    tokens = text.strip().split()
    # Remove extra whitespace from each token
    tokens = [token.strip() for token in tokens if token.strip()]
    return tokens

# Apply tokenization on Headline column
print("⏳ Tokenizing headlines...")
df_original["Tokens"] = df_original["Headline"].apply(tokenize_urdu_text)

# Add token count column
df_original["Token_Count"] = df_original["Tokens"].apply(len)

# Save tokenized data to CSV
df_tokenized = df_original[['Headline', 'Tokens', 'Token_Count']]
df_tokenized.to_csv("tokenized_urdu_data.csv", index=False, encoding='utf-8-sig')

print("✅ Tokenization complete!")
print(f"📊 Total Headlines: {len(df_tokenized)}")
print(f"📊 Average Tokens per Headline: {df_tokenized['Token_Count'].mean():.2f}")
print("\n📄 Sample Tokenized Data:")
print(df_tokenized.head(3))
print("\n✅ Tokenized data saved as 'tokenized_urdu_data.csv'\n")

# -------------------------------
# STEP 2: LOAD TOKENIZED DATA FOR GRAPHS
# -------------------------------
print("📊 Creating BEFORE processing graphs...")

# Load the tokenized CSV we just created
df = pd.read_csv("tokenized_urdu_data.csv")

# Tokens column ko string se list me convert karo
df["Tokens"] = df["Tokens"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Saare tokens ko ek list me jod lo
all_tokens = [word for tokens in df["Tokens"] for word in tokens]

# Top 50 frequent words nikal lo
word_counts = Counter(all_tokens).most_common(50)
words = [wc[0] for wc in word_counts]
counts = [wc[1] for wc in word_counts]

# Graphs folder banana
os.makedirs("graphs", exist_ok=True)

# Urdu text ko properly display karne ke liye reshape function
def reshape_urdu_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# ---------- BAR GRAPH ----------
# Urdu words ko reshape karo
reshaped_words = [reshape_urdu_text(word) for word in words]

# Matplotlib mein Urdu font set karo
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

plt.figure(figsize=(14, 6))
plt.bar(reshaped_words, counts, color='steelblue')
plt.xticks(rotation=90, fontsize=10, ha='right')
plt.yticks(fontsize=10)
plt.xlabel("Top 50 Urdu Words", fontsize=12)
plt.ylabel("Word Count", fontsize=12)
plt.title("Before Processing: Top 50 Most Frequent Urdu Words in Headlines", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("graphs/bar-plot-before.png", dpi=300)
plt.close()
print("✅ Bar graph successfully saved in 'graphs/bar-plot-before.png'")

# ---------- BOX PLOT ----------
df["Token_Count"] = df["Tokens"].apply(len)

plt.figure(figsize=(8, 6))
plt.boxplot(df["Token_Count"], patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'))
plt.title("Before Processing: Distribution of Words per Headline (Box Plot)", fontsize=14)
plt.ylabel("Number of Words", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("graphs/box-plot-before.png", dpi=300)
plt.close()
print("✅ Box plot successfully saved in 'graphs/box-plot-before.png'")

# ---------- SCATTER PLOT (TOP 50 WORDS) ----------
# X-axis = word length, Y-axis = frequency
x_values = [len(word) for word in words]
y_values = counts

# Random jitter for separation
np.random.seed(42)
x_jitter = np.array(x_values) + np.random.uniform(-0.3, 0.3, size=len(x_values))
y_jitter = np.array(y_values) + np.random.uniform(-20, 20, size=len(y_values))

plt.figure(figsize=(14, 6))
plt.scatter(
    x_jitter,
    y_jitter,
    c='royalblue',
    alpha=0.8,
    s=50,
    edgecolors='black',
    linewidths=0.3
)
plt.title("Before Processing: Top 50 Urdu Words (Length vs Frequency)", fontsize=14)
plt.xlabel("Word Length (Characters)", fontsize=12)
plt.ylabel("Word Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylim(0, max(y_values) * 1.1)  # Dynamic Y-axis limit
plt.tight_layout()
plt.savefig("graphs/scatter-plot-before.png", dpi=300)
plt.close()
print("✅ Scatter plot successfully saved in 'graphs/scatter-plot-before.png'")

# ---------- WORD CLOUD (URDU) ----------
# Word frequency dictionary banana
word_freq = Counter(all_tokens)

# Urdu words ko reshape kar ke naye dictionary me daalo
reshaped_word_freq = {}
for word, freq in word_freq.items():
    reshaped_word = reshape_urdu_text(word)
    reshaped_word_freq[reshaped_word] = freq

# Word Cloud banana with Urdu font
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='white',
    font_path=get_font_path(),  # Dynamic font path
    colormap='viridis',
    relative_scaling=0.5,
    min_font_size=10,
    max_words=100
).generate_from_frequencies(reshaped_word_freq)

# Word Cloud display karo
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Before Processing: Word Cloud - Most Frequent Urdu Words in Headlines", 
          fontsize=16, pad=20, fontweight='bold')
plt.tight_layout(pad=0)
plt.savefig("graphs/word-cloud-before.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Word Cloud successfully saved in 'graphs/word-cloud-before.png'")

print("\n🎉 All BEFORE processing graphs created successfully!")
print("📁 Check 'graphs' folder for all visualizations.")

