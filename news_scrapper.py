import os
import json
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from langdetect import detect

# Load environment variables
load_dotenv()

# NewsAPI Configuration
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
if not NEWSAPI_KEY:
    raise ValueError("Please set NEWSAPI_KEY in your .env file")

class NewsAPIFetcher:
    BASE_URL = 'https://newsapi.org/v2/top-headlines'
    
    def __init__(self, api_key=None, output_file='newsapi_articles.json'):
        """
        Initialize NewsAPI Fetcher
        
        :param api_key: NewsAPI key (defaults to environment variable)
        :param output_file: Path to save fetched articles
        """
        self.api_key = api_key or NEWSAPI_KEY
        self.output_file = output_file
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'Multilingual News Analytics Engine'
        }
    
    def fetch_articles(self, 
                       countries=['us', 'ae'],  # US and UAE for English and Arabic
                       categories=['business', 'technology', 'science'],
                       max_articles_per_country=50):
        """
        Fetch top headlines from specified countries and categories
        
        :param countries: List of country codes
        :param categories: List of news categories
        :param max_articles_per_country: Maximum articles to fetch per country
        """
        all_articles = []
        
        for country in countries:
            for category in categories:
                params = {
                    'country': country,
                    'category': category,
                    'pageSize': max_articles_per_country,
                    'language': 'en' if country != 'ae' else 'ar'
                }
                
                try:
                    response = requests.get(self.BASE_URL, 
                                            headers=self.headers, 
                                            params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        # Language detection
                        try:
                            language = detect(article.get('description', '') or article.get('title', ''))
                        except:
                            language = 'en' if country != 'ae' else 'ar'
                        
                        processed_article = {
                            'source': article.get('source', {}),
                            'author': article.get('author', ''),
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'urlToImage': article.get('urlToImage', ''),
                            'publishedAt': article.get('publishedAt', datetime.utcnow().isoformat()),
                            'content': article.get('content', ''),
                            'language': language,
                            'fetched_at': datetime.utcnow().isoformat(),
                            'country': country,
                            'category': category
                        }
                        all_articles.append(processed_article)
                
                except requests.RequestException as e:
                    print(f"Error fetching news for {country}/{category}: {e}")
        
        return all_articles
    
    def save_articles(self, articles):
        """
        Save articles to JSON file
        
        :param articles: List of article dictionaries
        """
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(articles)} articles to {self.output_file}")
        return len(articles)

def main():
    # Example usage
    fetcher = NewsAPIFetcher()
    
    # Fetch articles
    articles = fetcher.fetch_articles(
        countries=['us', 'ae'],
        categories=['business', 'technology', 'science'],
        max_articles_per_country=50
    )
    
    # Save articles
    fetcher.save_articles(articles)

if __name__ == '__main__':
    main()