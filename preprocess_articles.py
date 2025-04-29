import os
import json
import re
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')

try:
    from camel_tools.utils.dediac import dediac_ar
    from camel_tools.tokenizers.word import simple_word_tokenize
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

RAW_FILE = 'data/raw_articles.json'
CLEANED_FILE = 'data/cleaned_articles.json'

EN_STOPWORDS = set(stopwords.words('english'))
AR_STOPWORDS = set(stopwords.words('arabic')) if 'arabic' in stopwords.fileids() else set()

def clean_text(text, lang):
    if not text:
        return ''
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Remove non-word chars (keep Arabic)
    text = re.sub(r'\s+', ' ', text).strip()
    if lang == 'ar' and ARABIC_SUPPORT:
        text = dediac_ar(text)
    return text

def tokenize(text, lang):
    if lang == 'ar' and ARABIC_SUPPORT:
        tokens = simple_word_tokenize(text)
    else:
        tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens, lang):
    stops = AR_STOPWORDS if lang == 'ar' else EN_STOPWORDS
    return [t for t in tokens if t.lower() not in stops]

def preprocess_article(article):
    lang = article.get('language', 'en')
    content = article.get('content') or article.get('description') or ''
    cleaned = clean_text(content, lang)
    tokens = tokenize(cleaned, lang)
    tokens = remove_stopwords(tokens, lang)
    return {
        'title': article.get('title', ''),
        'language': lang,
        'cleaned_text': ' '.join(tokens),
        'publishedAt': article.get('publishedAt'),
        'source': article.get('source', {}),
        'url': article.get('url', ''),
        'description': article.get('description', ''),
        'content': article.get('content', ''),
    }

def main():
    if not Path(RAW_FILE).exists():
        raise FileNotFoundError(f"Raw articles file not found: {RAW_FILE}")
    with open(RAW_FILE, encoding='utf-8') as f:
        articles = json.load(f)
    cleaned = [preprocess_article(a) for a in articles]
    os.makedirs(os.path.dirname(CLEANED_FILE), exist_ok=True)
    with open(CLEANED_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Preprocessed {len(cleaned)} articles -> {CLEANED_FILE}")

if __name__ == '__main__':
    main()
