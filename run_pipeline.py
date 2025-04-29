import argparse
import subprocess
import sys
import os
from pathlib import Path
from tqdm import tqdm

def run_step(cmd, desc, cwd=None):
    print(f"\n===== {desc} =====")
    proc = subprocess.Popen([sys.executable, cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in tqdm(proc.stdout, desc=desc):
        print(line, end='')
    proc.wait()
    if proc.returncode != 0:
        print(f"Step '{desc}' failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)
    print(f"===== Finished: {desc} =====\n")

def main():
    parser = argparse.ArgumentParser(description="Run the Multilingual News Analytics Pipeline")
    parser.add_argument('--skip-ingest', action='store_true', help='Skip news ingestion step')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip-vectorize', action='store_true', help='Skip vectorization step')
    parser.add_argument('--skip-classify', action='store_true', help='Skip sentiment/topic classification step')
    parser.add_argument('--skip-topic', action='store_true', help='Skip topic classification (topic_transformers.py)')
    parser.add_argument('--skip-sentiment', action='store_true', help='Skip sentiment classification (sentiment_transformers.py)')
    args = parser.parse_args()

    steps = []
    if not args.skip_ingest:
        steps.append(('news_ingestion.py', 'Ingesting news data'))
    if not args.skip_preprocess:
        steps.append(('preprocess_articles.py', 'Preprocessing articles'))
    if not args.skip_vectorize:
        steps.append(('vectorize_articles.py', 'Vectorizing articles (TF-IDF)'))
    if not args.skip_classify:
        steps.append(('classify_sentiment_topic.py', 'Classifying sentiment & topic (transformers)'))
    if not args.skip_topic:
        steps.append(('topic_transformers.py', 'Topic classification (transformers)'))
    if not args.skip_sentiment:
        steps.append(('sentiment_transformers.py', 'Sentiment classification (transformers)'))

    cwd = Path(__file__).parent
    for script, desc in steps:
        script_path = cwd / script
        if not script_path.exists():
            print(f"Warning: {script} not found, skipping.")
            continue
        run_step(str(script_path), desc, cwd=str(cwd))
    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
