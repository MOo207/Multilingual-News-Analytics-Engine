import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Load cleaned articles
with open('data/cleaned_articles.json', encoding='utf-8') as f:
    articles = json.load(f)

# Only keep English and Arabic articles
english_articles = [a for a in articles if a.get('language') == 'en']
arabic_articles = [a for a in articles if a.get('language') == 'ar']

# Define candidate topic labels (customize as needed)
candidate_labels = [
    'politics', 'sports', 'business', 'technology', 'health', 'entertainment',
    'science', 'world', 'local', 'environment', 'crime', 'education', 'opinion'
]

# Sentiment analysis pipeline (English, XLM-R)
sentiment_pipe_en = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment', tokenizer='cardiffnlp/twitter-xlm-roberta-base-sentiment', top_k=None)
# Topic classification pipeline (zero-shot, English)
topic_pipe_en = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Sentiment analysis pipeline (Arabic, updated to public model)
sentiment_pipe_ar = pipeline('sentiment-analysis', model='CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment')
# Topic classification pipeline (zero-shot, Arabic, multilingual, force slow tokenizer)
topic_pipe_ar = pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli', use_fast=False)

# Classify Arabic articles
for article in tqdm(arabic_articles, desc='Classifying Arabic articles'):
    text = article.get('cleaned_text') or article.get('title') or ''
    # Sentiment
    if text.strip():
        try:
            pred = sentiment_pipe_ar(text[:512])[0]
            article['transformer_sentiment'] = pred['label']
            article['transformer_sentiment_score'] = float(pred['score'])
        except Exception as e:
            article['transformer_sentiment'] = 'ERROR'
            article['transformer_sentiment_score'] = 0.0
    else:
        article['transformer_sentiment'] = 'EMPTY'
        article['transformer_sentiment_score'] = 0.0
    # Topic
    if text.strip():
        try:
            result = topic_pipe_ar(text[:512], candidate_labels)
            article['transformer_topic'] = result['labels'][0]
            article['transformer_topic_scores'] = dict(zip(result['labels'], map(float, result['scores'])))
        except Exception as e:
            article['transformer_topic'] = 'ERROR'
            article['transformer_topic_scores'] = {}
    else:
        article['transformer_topic'] = 'EMPTY'
        article['transformer_topic_scores'] = {}

# save ar results
with open('data/classified_articles_transformers_ar.json', 'w', encoding='utf-8') as f:
    json.dump(arabic_articles, f, ensure_ascii=False, indent=2)
    print('Saved Arabic sentiment and topic predictions to data/classified_articles_transformers_ar.json')


# Classify English articles
for article in tqdm(english_articles, desc='Classifying English articles'):
    text = article.get('cleaned_text') or article.get('title') or ''
    # Sentiment
    if text.strip():
        try:
            pred = sentiment_pipe_en(text[:512])[0]
            article['transformer_sentiment'] = pred['label'] if isinstance(pred, dict) else pred[0]['label']
            article['transformer_sentiment_score'] = float(pred['score']) if isinstance(pred, dict) else float(pred[0]['score'])

        except Exception as e:
            article['transformer_sentiment'] = 'ERROR'
            article['transformer_sentiment_score'] = 0.0
    else:
        article['transformer_sentiment'] = 'EMPTY'
        article['transformer_sentiment_score'] = 0.0
    # Topic
    if text.strip():
        try:
            result = topic_pipe_en(text[:512], candidate_labels)
            article['transformer_topic'] = result['labels'][0]
            article['transformer_topic_scores'] = dict(zip(result['labels'], map(float, result['scores'])))
        except Exception as e:
            article['transformer_topic'] = 'ERROR'
            article['transformer_topic_scores'] = {}
    else:
        article['transformer_topic'] = 'EMPTY'
        article['transformer_topic_scores'] = {}

# Save results
with open('data/classified_articles_transformers_en.json', 'w', encoding='utf-8') as f:
    json.dump(english_articles, f, ensure_ascii=False, indent=2)
    print('Saved English sentiment and topic predictions to data/classified_articles_transformers_en.json')