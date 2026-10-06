#import libraries
import re
import nltk
import pdfplumber
import spacy
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#loading needed resources NLTK (takes once)
nltk.download('punkt_tab')
nltk.download('stopwords')

def extract_and_clean_pdf(pdf_path):
    """
    Step 1.1: Exctracted raw PDF-file
    """
    raw_text_pages = []

    #Used pdfplumber for correctly text two columns reading  
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                raw_text_pages.append(text)

    #United all pages to one huge text
    full_raw_text = "\n".join(raw_text_pages)
    print(f"[INFO] Success extracted {len(raw_text_pages)} pages of the text.")

    """
    Step 1.2: Cleaning text from academic "noise"
    """
    cleaned_text = full_raw_text.lower()

    cleaned_text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', cleaned_text)

    cleaned_text = re.sub(r'\b(fig\.|figure|table)\b\s?\d+', '', cleaned_text)

    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)

    """
    Step 1.3: Tokenize and stopwords filtration
    """

    tokens = word_tokenize(cleaned_text)

    stop_words = set(stopwords.words('english'))

    custom_stop_words = {
        'multimedia', 'tools', 'applications', 'pp', 'vol', 'editor', 
        'proceedings', 'ieee', 'springer', 'https', 'doi', 'al', 'et'
    }
    stop_words.update(custom_stop_words)

    final_tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words and len(word) > 2
    ]

    print(f"[INFO] Quantity tokens after filtration: {len(final_tokens)}")
    return full_raw_text, final_tokens


#push to the start code
pdf_file_path = "Mnasri2021_Article_AnomalousSoundEventDetection.pdf"

full_raw_text, cleaned_tokens = extract_and_clean_pdf(pdf_file_path)

#%%

# Counted frequency words
fdist = FreqDist(cleaned_tokens)

print("---TOP-20 KEYWORDS ON ARTICLE---")
for word, frequency in fdist.most_common(20):
    print(f"{word}: {frequency}")

#Build histogram
plt.figure(figsize=(12, 6))
fdist.plot(25, title="Top 25 Most Frequent Words in the Article", cumulative=False)
plt.show()

#Generate word cloud
wordcloud_text = " ".join(cleaned_tokens)
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=100,
    colormap='viridis'
).generate(wordcloud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Anomalous Sound Event Detection Survey", fontsize=16)
plt.show()

# %% 

# Loaded pre-trained NLP-model
nlp = spacy.load("en_core_web_sm")

# For NER, we need the raw, cleaned text as sentences rather than individual tokens.
# We take the raw_text variable returned in Step 1 and pass it to spaCy.
# To prevent the code from hanging on a massive review text, we process the first 100,000 characters
# (this is more than enough to demonstrate the algorithm in action).
doc = nlp(full_raw_text[:100000])

#Extracted entities

entities = []
for ent in doc.ents:
    if ent.label_ in ["ORG", "PRODUCT", "LAW", "WORK_OF_ART"]:
        entities.append((ent.text.strip(), ent.label_))

df_ents = pd.DataFrame(entities, columns=["Entity", "Label"])

df_ents = df_ents[df_ents["Entity"].str.len() > 1]

print("---TOP-15 EXTRACTED ENTITIES (NER)---")
print(df_ents["Entity"].value_counts().head(15))

#Visualization

top_entities = df_ents["Entity"].value_counts().head(15)

sns.barplot(x=top_entities.values, y=top_entities.index, hue=top_entities.index, palette="viridis", legend=False)

plt.title("Top 15 Extracted Named Entities (Models, Datasets, Frameworks)", fontsize=14)
plt.xlabel("Frequency Count", fontsize=12)
plt.ylabel("Entity Name", fontsize=12)
plt.tight_layout()

plt.show()

# %% 

sentences = sent_tokenize(full_raw_text[:50000])

sentences = [sent.strip() for sent in sentences if len(sent.split()) > 5]

print(f"[INFO] Sentences selected for analysis: {len(sentences)}")

# Create a similarity matrix and graph 

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(sentences)

similarity_matrix = cosine_similarity(tfidf_matrix)

nx_graph = nx.from_numpy_array(similarity_matrix)

scores = nx.pagerank(nx_graph)

# Output TOP-5 main sentences

ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

print("\n" + "="*50)
print("--- AUTOMATICALY ARTICLE SUMMARIZER (ТОП-5) ---")
print("="*50)

for i in range(min(5, len(ranked_sentences))):
    print(f"\n[{i+1}] {ranked_sentences[i][1]}")
# %%