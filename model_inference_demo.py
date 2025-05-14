import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline
)

class MultilingualNewsAnalyzer:
    def __init__(self, 
                 sentiment_model_path='finetuned_model/portable_en_sentiment',
                 topic_model_path='finetuned_model/portable_en_topic'):
        """
        Initialize multilingual news analysis models
        
        :param sentiment_model_path: Path to fine-tuned sentiment model
        :param topic_model_path: Path to fine-tuned topic model
        """
        # Sentiment Analysis Model
        try:
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis', 
                model=self.sentiment_model, 
                tokenizer=self.sentiment_tokenizer
            )
        except Exception as e:
            print(f"Sentiment model loading error: {e}")
            self.sentiment_pipeline = None
        
        # Topic Classification Model (placeholder, as we don't have a fine-tuned topic model)
        self.topic_labels = [
        'politics', 'sports', 'business', 'technology', 'health', 'entertainment',
        'science', 'world', 'local', 'environment', 'crime', 'education', 'opinion'
    ]
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of given text
        
        :param text: Input text to analyze
        :return: Sentiment analysis results
        """
        if not self.sentiment_pipeline:
            return {"error": "Sentiment model not loaded"}
        
        try:
            # Truncate text to model's max length
            max_length = self.sentiment_tokenizer.model_max_length
            truncated_text = text[:max_length]
            
            result = self.sentiment_pipeline(truncated_text)[0]
            return {
                "label": result['label'],
                "score": result['score']
            }
        except Exception as e:
            return {"error": str(e)}
    
    def classify_topic(self, text):
        """
        Classify topic using zero-shot classification

        :param text: Input text to classify
        :return: Topic classification results
        """
        from transformers import pipeline
        
        try:
            # Use zero-shot classification
            classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
            
            result = classifier(text, self.topic_labels)
            return {
                "topic": result['labels'][0],
                "confidence": result['scores'][0]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_articles(self, articles):
        """
        Analyze a list of articles
        
        :param articles: List of article dictionaries
        :return: List of analyzed articles
        """
        analyzed_articles = []
        for article in articles:
            text = article.get('description', '') or article.get('title', '')
            
            # Skip empty texts
            if not text.strip():
                continue
            
            analyzed_article = article.copy()
            
            # Sentiment Analysis
            sentiment_result = self.analyze_sentiment(text)
            analyzed_article['sentiment_analysis'] = sentiment_result
            
            # Topic Classification
            topic_result = self.classify_topic(text)
            analyzed_article['topic_classification'] = topic_result
            
            analyzed_articles.append(analyzed_article)
        
        return analyzed_articles

def main():
    # Load existing articles from raw_articles.json
    with open('data/raw_articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Filter articles by language and content
    ar_articles = [
        article for article in articles
        if article.get('language') == 'ar' and
           article.get('content') and len(article.get('content', '')) > 5
    ][:5]  # First 5 Arabic articles
    
    en_articles = [
        article for article in articles
        if article.get('language') == 'en' and
           article.get('content') and len(article.get('content', '')) > 5
    ][:5]  # First 5 English articles
    
    # Combine articles
    filtered_articles = ar_articles + en_articles
    
    print(f"Loaded {len(filtered_articles)} articles for analysis")
    
    # Initialize analyzer
    analyzer = MultilingualNewsAnalyzer()
    
    # Analyze articles
    analyzed_articles = analyzer.analyze_articles(filtered_articles)
    analyzed_articles = analyzer.analyze_articles(articles)
    
    # Save analyzed articles
    output_file = 'data/analyzed_raw_articles.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analyzed_articles, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nAnalyzed {len(analyzed_articles)} articles:")
    for article in analyzed_articles[:10]:  # Print first 10 for demo
        print("\n---Article---")
        print(f"Language: {article.get('language', 'N/A')}")
        print(f"Title: {article.get('title', 'N/A')}")
        print(f"Sentiment: {article.get('sentiment_analysis', {}).get('label', 'N/A')}")
        print(f"Topic: {article.get('topic_classification', {}).get('topic', 'N/A')}")

if __name__ == '__main__':
    main()

# Requirements:
# 1. Fine-tuned sentiment model in finetuned_model/portable_en_sentiment
# 2. NewsAPI key in .env file
# 3. Install: transformers, torch, requests, python-dotenv, langdetect
"""
Recommended setup:
1. pip install transformers torch requests python-dotenv langdetect
2. Ensure NewsAPI key is set in .env
3. Run fine_tune_and_evaluate.py to generate sentiment model checkpoint
4. Run this script to analyze fresh news articles
"""
