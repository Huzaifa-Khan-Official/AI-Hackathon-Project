import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
import os
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import rcParams
import numpy as np
import matplotlib.ticker as ticker
from wordcloud import WordCloud
import platform

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
# Step 0: Load 10k dataset (already tokenized) 
# -------------------------------
try:
    df = pd.read_csv("tokenized_urdu_data.csv")
except FileNotFoundError:
    print("❌ Error: tokenized_urdu_data.csv not found!")
    exit()

df["Tokens"] = df["Tokens"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Word count before cleaning (optional, for reference)
df["Token_Count_Before"] = df["Tokens"].apply(len)

# -------------------------------
# Step 1: Preprocessing
# -------------------------------
# 1.1 Stopwords Removal
urdu_stopwords = set(["کا","کی","کے","اور","ہے","ہیں","تھا","تھی","میں","سے","پر","کو","نے",
                      "یہ","وہ","کہ","تو","ہی","بھی","ایک","یہی","تک","بعد","جب","اگر"])
df["Tokens"] = df["Tokens"].apply(lambda tokens: [w for w in tokens if w not in urdu_stopwords])

# 1.2 Punctuation Removal
def remove_punctuations(tokens):
    return [re.sub(r"[^\w\s]", "", w) for w in tokens if re.sub(r"[^\w\s]", "", w) != ""]
df["Tokens"] = df["Tokens"].apply(remove_punctuations)

# 1.3 Lowercase
df["Tokens"] = df["Tokens"].apply(lambda tokens: [w.lower() for w in tokens])

# 1.4 Standardization
standardization_map = {"ي": "ی", "ئ": "ی", "ۀ": "ہ", "ة": "ہ", "ى": "ی"}
def standardize_text(tokens):
    new_tokens = []
    for w in tokens:
        for old, new in standardization_map.items():
            w = w.replace(old, new)
        new_tokens.append(w)
    return new_tokens
df["Tokens"] = df["Tokens"].apply(standardize_text)

# 1.5 Stemming
suffixes = ["وں","یں","ی","ا","ات","ے"]
def urdu_stem(word):
    for suf in suffixes:
        if word.endswith(suf):
            return word[:-len(suf)]
    return word
df["Tokens"] = df["Tokens"].apply(lambda tokens: [urdu_stem(w) for w in tokens])

# 1.6 Lemmatization (remove single-letter or empty tokens)
df["Tokens"] = df["Tokens"].apply(lambda tokens: [w for w in tokens if len(w) > 1])

# 1.7 Remove numbers from tokens
df["Tokens"] = df["Tokens"].apply(lambda tokens: [w for w in tokens if not w.isdigit()])

# -------------------------------
# Step 2: Update Token Count After Cleaning
# -------------------------------
df["Token_Count_After"] = df["Tokens"].apply(len)

# -------------------------------
# Step 3: Save Cleaned CSV (only relevant columns)
# -------------------------------
df_cleaned = df[['Headline','Tokens','Token_Count_After']]
df_cleaned.to_csv("cleaned_urdu_data.csv", index=False, encoding='utf-8-sig')

print("✅ Cleaned CSV saved successfully as 'cleaned_urdu_data.csv'")

# -------------------------------
# AFTER PROCESSING GRAPHS
# -------------------------------

# Graphs folder bana lo agar nahi hai
os.makedirs("graphs", exist_ok=True)

# Urdu text reshape function
def reshape_urdu_text(text):
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    return bidi_text

# Top 50 frequent words
all_tokens = [word for tokens in df_cleaned["Tokens"] for word in tokens]
word_counts = Counter(all_tokens).most_common(50)
words = [w[0] for w in word_counts]
counts = [w[1] for w in word_counts]

# Urdu words ko reshape karo
reshaped_words = [reshape_urdu_text(w) for w in words]

# Font setup
rcParams['font.family'] = 'Arial'

# -------------------------------
# BAR PLOT (After Processing)
# -------------------------------
plt.figure(figsize=(14,6))
plt.bar(reshaped_words, counts, color='skyblue')
plt.xticks(rotation=90, fontsize=12, ha='right')
plt.yticks(fontsize=12)
plt.ylabel("Word Count", fontsize=14)
plt.title("After Processing: Top 50 Urdu Words", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout(pad=2)
plt.savefig("graphs/bar-plot-after.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ After Processing Bar Graph saved in 'graphs/bar-plot-after.png'")

# -------------------------------
# BOX PLOT (After Processing)
# -------------------------------
plt.figure(figsize=(10,6))
plt.boxplot(df_cleaned["Token_Count_After"], patch_artist=True,
            boxprops=dict(facecolor='lightgreen', color='green'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='green'),
            capprops=dict(color='green'))

plt.title("After Processing: Distribution of Words per Headline (Box Plot)", fontsize=16)
plt.ylabel("Number of Words", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout(pad=2)
plt.savefig("graphs/box-plot-after.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ After Processing Box Plot saved in 'graphs/box-plot-after.png'")

# -------------------------------
# SCATTER PLOT (After Processing)
# -------------------------------
# X-axis = word length, Y-axis = frequency
x_values = [len(word) for word in words]
y_values = counts

# Random jitter for separation (different seed than before)
np.random.seed(100)
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
plt.title("After Processing: Top 50 Urdu Words (Length vs Frequency)", fontsize=14)
plt.xlabel("Word Length (Characters)", fontsize=12)
plt.ylabel("Word Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylim(0, max(y_values) * 1.1)  # Dynamic Y-axis limit
plt.tight_layout()
plt.savefig("graphs/scatter-plot-after.png", dpi=300)
plt.close()

print("✅ After Processing Scatter Plot saved in 'graphs/scatter-plot-after.png'")

# -------------------------------
# WORD CLOUD (After Processing)
# -------------------------------
# Word frequency dictionary banana from cleaned tokens
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
    colormap='plasma',
    relative_scaling=0.5,
    min_font_size=10,
    max_words=100
).generate_from_frequencies(reshaped_word_freq)

# Word Cloud display karo
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("After Processing: Word Cloud - Most Frequent Urdu Words", 
          fontsize=18, pad=20, fontweight='bold')
plt.tight_layout(pad=0)
plt.savefig("graphs/word-cloud-after.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ After Processing Word Cloud saved in 'graphs/word-cloud-after.png'")

print("\n🎉 All AFTER processing graphs created successfully!")
print("📁 Check 'graphs' folder for all visualizations.")