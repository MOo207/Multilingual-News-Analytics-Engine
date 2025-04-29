import json
from transformers import pipeline
from tqdm import tqdm

# Load cleaned articles
with open('data/cleaned_articles.json', encoding='utf-8') as f:
    articles = json.load(f)

# Use a multilingual zero-shot classification pipeline for topic
# Model: facebook/bart-large-mnli (works for English, for Arabic you can try joeddav/xlm-roberta-large-xnli)
# Define candidate labels (customize for your news domain)
candidate_labels = [
    'politics', 'sports', 'business', 'technology', 'health', 'entertainment',
    'science', 'world', 'local', 'environment', 'crime', 'education', 'opinion'
]

# Use English model for all for simplicity; for Arabic, you may want to split and use an Arabic NLI model
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

for article in tqdm(articles, desc='Classifying topic'):
    text = article.get('cleaned_text') or article.get('title') or ''
    if text.strip():
        try:
            result = classifier(text[:512], candidate_labels)
            article['transformer_topic'] = result['labels'][0]
            article['transformer_topic_scores'] = dict(zip(result['labels'], map(float, result['scores'])))
        except Exception as e:
            article['transformer_topic'] = 'ERROR'
            article['transformer_topic_scores'] = {}
    else:
        article['transformer_topic'] = 'EMPTY'
        article['transformer_topic_scores'] = {}

with open('data/classified_articles_transformers_topic.json', 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print('Saved topic predictions using transformers to data/classified_articles_transformers_topic.json')
