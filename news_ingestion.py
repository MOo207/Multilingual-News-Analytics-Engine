import os
import csv
import json
import requests
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
import concurrent.futures
from tqdm import tqdm

CSV_FILES = [
    r'Datasets/One Week of Global News Feeds/news-week-18aug24.csv',
    # r'Datasets/One Week of Global News Feeds/news-week-18aug24.csv'
]
OUTPUT_FILE = 'data/raw_articles.json'

# Helper to extract domain name from URL
def get_source_name(url):
    try:
        return urlparse(url).netloc.replace('www.', '')
    except Exception:
        return None

def fetch_article_content(url, timeout=10):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.ok:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Try to extract main content heuristically
            article = soup.find('article')
            if article:
                text = article.get_text(separator=' ', strip=True)
            else:
                # fallback: most text in <p> tags
                ps = soup.find_all('p')
                text = ' '.join([p.get_text(separator=' ', strip=True) for p in ps])
            # Try to extract main image
            img = soup.find('meta', property='og:image')
            urlToImage = img['content'] if img and img.has_attr('content') else ''
            return text, urlToImage
        else:
            return '', ''
    except Exception:
        return '', ''

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return ''

def convert_publish_time(ts):
    # Expecting format like 201708240000
    try:
        return datetime.strptime(ts, '%Y%m%d%H%M').isoformat()
    except Exception:
        return ''

def process_csv_row(row):
    publish_time, feed_code, source_url, headline_text = row
    content, urlToImage = fetch_article_content(source_url)
    language = detect_language(headline_text) if headline_text else ''
    return {
        'source': {
            'id': None,
            'name': get_source_name(source_url)
        },
        'author': '',
        'title': headline_text,
        'description': headline_text,
        'url': source_url,
        'urlToImage': urlToImage,
        'publishedAt': convert_publish_time(publish_time),
        'content': content,
        'fetched_at': datetime.utcnow().isoformat(),
        'language': language
    }

def process_csv_row_with_lang(row):
    headline_text = row[3]
    lang = detect_language(headline_text) if headline_text else ''
    return lang, row

def main():
    all_rows = []
    max_per_lang = 150 # for quick processing
    lang_rows = {'ar': [], 'en': []}
    for csv_file in CSV_FILES:
        with open(csv_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 4:
                    continue
                headline_text = row[3]
                lang = detect_language(headline_text) if headline_text else ''
                if lang in lang_rows and len(lang_rows[lang]) < max_per_lang:
                    lang_rows[lang].append(row)
                # Break early if both quotas are filled
                if len(lang_rows['ar']) >= max_per_lang and len(lang_rows['en']) >= max_per_lang:
                    break
        if len(lang_rows['ar']) >= max_per_lang and len(lang_rows['en']) >= max_per_lang:
            break
    print(f"Selected {len(lang_rows['ar'])} ar, {len(lang_rows['en'])} en articles for processing.")
    selected_rows = lang_rows['ar'] + lang_rows['en']
    # Fetch content in parallel
    all_articles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        articles = list(tqdm(executor.map(process_csv_row, selected_rows), total=len(selected_rows), desc='Fetching content'))
        all_articles.extend(articles)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_articles)} articles to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
