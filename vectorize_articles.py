import json
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

CLEANED_FILE = 'data/cleaned_articles.json'
VECTORIZER_FILE = 'data/vectorizer.pkl'
FEATURES_FILE = 'data/tfidf_features.npz'

# Load cleaned articles
def load_cleaned_articles():
    with open(CLEANED_FILE, encoding='utf-8') as f:
        articles = json.load(f)
    texts = [a['cleaned_text'] for a in articles]
    return texts, articles

def main():
    texts, articles = load_cleaned_articles()
    print(f"Loaded {len(texts)} cleaned articles.")
    # TF-IDF vectorization (handles both English and Arabic)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    os.makedirs(os.path.dirname(VECTORIZER_FILE), exist_ok=True)
    # Save vectorizer
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    # Save features
    from scipy import sparse
    sparse.save_npz(FEATURES_FILE, X)
    print(f"Saved TF-IDF features to {FEATURES_FILE} and vectorizer to {VECTORIZER_FILE}")

if __name__ == '__main__':
    main()
