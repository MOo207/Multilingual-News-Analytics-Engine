import os
import json
import argparse
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import torch

def load_articles(json_path, label_type):
    """
    Load articles and extract cleaned_text and label (sentiment or topic).
    Only keep articles with valid labels (not EMPTY, ERROR, or None).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    articles = []
    for art in data:
        text = art.get('cleaned_text')
        if not text:
            continue
        if label_type == 'sentiment':
            label = art.get('transformer_sentiment')
        else:
            label = art.get('transformer_topic')
        # Only keep valid labels
        if label and label not in ('EMPTY', 'ERROR', None):
            articles.append({'text': text, 'label': label})
    return articles

def prepare_label_mapping(articles):
    labels = sorted(list(set(a['label'] for a in articles)))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label

def split_data(articles, seed=42):
    random.seed(seed)
    random.shuffle(articles)
    n = len(articles)
    train = articles[:int(0.7*n)]
    val = articles[int(0.7*n):int(0.85*n)]
    test = articles[int(0.85*n):]
    return train, val, test

def to_hf_dataset(data, label2id):
    return Dataset.from_list([
        {'text': a['text'], 'label': label2id[a['label']]} for a in data
    ])

def compute_metrics(pred, id2label):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    report = classification_report(labels, preds, target_names=[id2label[i] for i in sorted(id2label.keys())])
    return {'accuracy': acc, 'f1': f1, 'report': report}

def evaluate_model(trainer, dataset, id2label):
    results = trainer.predict(dataset)
    metrics = compute_metrics(results, id2label)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['en', 'ar'], default='en', help='Language to use')
    parser.add_argument('--task', choices=['sentiment', 'topic'], default='sentiment', help='Task to fine-tune')
    parser.add_argument('--model_name', type=str, default=None, help='Model to use (defaults based on language/task)')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='finetuned_model')
    args = parser.parse_args()

    # Model names exactly as used in classify_sentiment_topic.py
    MODEL_NAMES = {
        ('en', 'sentiment'): 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        ('en', 'topic'): 'facebook/bart-large-mnli',
        ('ar', 'sentiment'): 'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment',
        ('ar', 'topic'): 'joeddav/xlm-roberta-large-xnli',
    }

    # File selection
    if args.language == 'en':
        data_file = 'data/classified_articles_transformers_en.json'
    else:
        data_file = 'data/classified_articles_transformers_ar.json'
    # Use the exact model names from classify_sentiment_topic.py unless overridden
    model_name = args.model_name or MODEL_NAMES[(args.language, args.task)]
    print(f"Using model: {model_name}")

    # Load and process data
    articles = load_articles(data_file, args.task)
    if not articles:
        print(f"No valid labeled articles found for {args.language} {args.task}. Check your data file and labels.")
        return
    label2id, id2label = prepare_label_mapping(articles)
    if len(label2id) < 2:
        print(f"Not enough unique labels for classification (found {len(label2id)}: {list(label2id.keys())}). At least 2 required.")
        return
    train, val, test = split_data(articles)
    train_ds = to_hf_dataset(train, label2id)
    val_ds = to_hf_dataset(val, label2id)
    test_ds = to_hf_dataset(test, label2id)
    ds = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)
    ds = ds.map(tokenize_fn, batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Baseline evaluation (original model)
    trainer_orig = Trainer(
        model=model,
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )
    print("\nEvaluating original (pretrained) model...")
    orig_metrics = evaluate_model(trainer_orig, ds['test'], id2label)
    print("Original Model Metrics:")
    print(f"Accuracy: {orig_metrics['accuracy']:.4f}, F1: {orig_metrics['f1']:.4f}")
    print(orig_metrics['report'])

    # Fine-tuning
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        # Remove load_best_model_at_end for compatibility with your transformers version
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )
    print("\nFine-tuning model...")
    trainer.train()
    trainer.save_model(args.output_dir)

    # Evaluation after fine-tuning
    print("\nEvaluating fine-tuned model...")
    ft_metrics = evaluate_model(trainer, ds['test'], id2label)
    print("Fine-tuned Model Metrics:")
    print(f"Accuracy: {ft_metrics['accuracy']:.4f}, F1: {ft_metrics['f1']:.4f}")
    print(ft_metrics['report'])

    # Save metrics
    with open(os.path.join(args.output_dir, f'metrics_{args.language}_{args.task}.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'original': orig_metrics,
            'fine_tuned': ft_metrics
        }, f, indent=2, ensure_ascii=False)

    # Save portable model (for deployment/use elsewhere)
    portable_dir = os.path.join(args.output_dir, f'portable_{args.language}_{args.task}')
    trainer.model.save_pretrained(portable_dir)
    tokenizer.save_pretrained(portable_dir)
    print(f"\nPortable fine-tuned model saved to {portable_dir}")

    # Also save the model in the main output_dir for convenience
    trainer.save_model(args.output_dir)
    print(f"Fine-tuned model and tokenizer also saved to {args.output_dir}")

    print("\nSummary:")
    print(f"Original Model - Accuracy: {orig_metrics['accuracy']:.4f}, F1: {orig_metrics['f1']:.4f}")
    print(f"Fine-tuned Model - Accuracy: {ft_metrics['accuracy']:.4f}, F1: {ft_metrics['f1']:.4f}")
    print(f"Metrics saved to {args.output_dir}/metrics_{args.language}_{args.task}.json")

if __name__ == '__main__':
    main()
