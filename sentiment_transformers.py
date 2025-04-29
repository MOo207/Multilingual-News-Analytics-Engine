import json
from transformers import pipeline
from tqdm import tqdm

# Load cleaned articles
with open('data/cleaned_articles.json', encoding='utf-8') as f:
    articles = json.load(f)

# Use a multilingual sentiment analysis pipeline
# Model: cardiffnlp/twitter-xlm-roberta-base-sentiment (works for English and reasonable for Arabic)
sentiment_pipe = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment', tokenizer='cardiffnlp/twitter-xlm-roberta-base-sentiment')

# Predict sentiment for each article
for article in tqdm(articles, desc='Predicting sentiment'):
    text = article.get('cleaned_text') or article.get('title') or ''
    if text.strip():
        try:
            pred = sentiment_pipe(text[:512])[0]  # Truncate to 512 tokens for speed/safety
            article['transformer_sentiment'] = pred['label']
            article['transformer_sentiment_score'] = float(pred['score'])
        except Exception as e:
            article['transformer_sentiment'] = 'ERROR'
            article['transformer_sentiment_score'] = 0.0
    else:
        article['transformer_sentiment'] = 'EMPTY'
        article['transformer_sentiment_score'] = 0.0

with open('data/classified_articles_transformers.json', 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print('Saved sentiment predictions using transformers to data/classified_articles_transformers.json')
