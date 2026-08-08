
# | Member Name               | Contribution | Mail ID                    |
# |---------------------------|------|------------------------------------|
# | SARAVANAN NALLAMUTHU      | 100% | 2025ab05285@wilp.bits-pilani.ac.in |
# | S HARISH SANKARANARAYANAN | 100% | 2025aa05227@wilp.bits-pilani.ac.in |
# | BHARATH M                 | 100% | 2025aa05641@wilp.bits-pilani.ac.in |
# | SHANMUGASUNDARAM          | 100% | 2025aa05346@wilp.bits-pilani.ac.in |
# | VINOTHINI                 | 100% | 2025ab05176@wilp.bits-pilani.ac.in |

# ==========================================================
# AI-Based Text Summarization System
# Transformer Encoder-Decoder Architecture
# Dataset: CNN/DailyMail
# Models: BART, T5, PEGASUS
# ==========================================================

import argparse
import csv
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

torch = None


def _load_torch_backend():
    global torch

    if torch is not None:
        return torch

    try:
        import torch as torch_module
    except Exception as exc:  # pragma: no cover
        logging.warning("torch is unavailable: %s", exc)
        torch = None
        return None

    torch = torch_module
    return torch

AutoModelForSeq2SeqLM = None
AutoTokenizer = None
pipeline = None
transformers_version = None


def _load_transformers_backend():
    global AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, transformers_version

    if AutoModelForSeq2SeqLM is not None and AutoTokenizer is not None:
        return True

    try:
        import importlib

        module = importlib.import_module("transformers")
        AutoModelForSeq2SeqLM = module.AutoModelForSeq2SeqLM
        AutoTokenizer = module.AutoTokenizer
        pipeline = module.pipeline
        transformers_version = getattr(module, "__version__", None)
        return True
    except Exception as exc:  # pragma: no cover
        logging.warning("transformers is unavailable: %s", exc)
        AutoModelForSeq2SeqLM = None
        AutoTokenizer = None
        pipeline = None
        transformers_version = None
        return False


# Evaluation libraries

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    from nltk.translate.bleu_score import sentence_bleu
except Exception:
    sentence_bleu = None

bert_score = None


def _load_bert_score():
    global bert_score

    if bert_score is not None:
        return bert_score

    try:
        from bert_score import score as bert_score_module
    except Exception as exc:  # pragma: no cover
        logging.warning("bert_score is unavailable: %s", exc)
        bert_score = None
        return None

    bert_score = bert_score_module
    return bert_score


# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)

# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = Path("cnn_dailymail")

MODELS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-base",
    "pegasus": "google/pegasus-cnn_dailymail",
}

MODEL_CACHE = {}
TOKENIZER_CACHE = {}


def _get_device() -> str:
    torch_module = _load_torch_backend()
    if torch_module is None:
        return "cpu"

    return "cuda" if torch_module.cuda.is_available() else "cpu"

# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text: str) -> str:
    """Remove extra whitespace from the text."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [s.strip() for s in sentences if s.strip()]

# ==========================================================
# Dataset Loading
# ==========================================================


def load_dataset(file_path: str) -> List[Dict]:
    data = []

    with open(file_path, encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            data.append(
                {
                    "id": row["id"],
                    "article": clean_text(row["article"]),
                    "summary": clean_text(row["highlights"]),
                }
            )

    return data

# ==========================================================
# Tokenization
# ==========================================================

def load_tokenizer(model_name: str):
    if model_name in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[model_name]

    if not _load_transformers_backend():
        logging.error("transformers is not available; tokenizer could not be loaded")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        logging.error("tokenizer could not be loaded for %s: %s", model_name, exc)
        return None

    TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def tokenize_document(text: str, model_name: str, max_length: int = 1024):
    tokenizer = load_tokenizer(model_name)
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokens

# ==========================================================
# Extractive Summarization Baseline
# ==========================================================

def tfidf_summary(article: str, sentences_count: int = 3) -> str:
    sentences = split_sentences(article)

    if len(sentences) <= sentences_count:
        return " ".join(sentences)

    if not sentences:
        return ""

    tokenized_sentences = [re.findall(r"\b[a-zA-Z]{2,}\b", sentence.lower()) for sentence in sentences]
    if not any(tokenized_sentences):
        return " ".join(sentences[:sentences_count])

    word_freq = Counter(token for tokens in tokenized_sentences for token in tokens)
    scored_sentences = []
    for sentence, tokens in zip(sentences, tokenized_sentences):
        score = sum(word_freq[token] for token in tokens)
        scored_sentences.append((score, sentence))

    ranked = sorted(scored_sentences, reverse=True)
    selected = [sentence for _, sentence in ranked[:sentences_count]]
    return " ".join(selected)

# ==========================================================
# Transformer Encoder-Decoder Model
# ==========================================================

def load_seq2seq_model(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if not _load_transformers_backend():
        logging.error("transformers is not available; model could not be loaded")
        return None

    torch_module = _load_torch_backend()
    if torch_module is None:
        logging.error("torch is not available; model could not be loaded")
        return None

    logging.info(f"Loading model: {model_name}")

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as exc:
        logging.error("model could not be loaded for %s: %s", model_name, exc)
        return None

    device = _get_device()

    try:
        model.to(device)
        model.eval()
    except Exception as exc:
        logging.error("model could not be moved to device %s: %s", device, exc)
        return None

    MODEL_CACHE[model_name] = model
    return model

# ==========================================================
# Encoder-Decoder Summary Generation
# ==========================================================

def generate_summary(article: str, model_name: str, max_length: int = 120, min_length: int = 40):
    torch_module = _load_torch_backend()
    if torch_module is None:
        return tfidf_summary(article, sentences_count=3)

    model = load_seq2seq_model(model_name)
    tokenizer = load_tokenizer(model_name)

    if model is None or tokenizer is None:
        return tfidf_summary(article, sentences_count=3)

    inputs = tokenizer(
        article,
        max_length=1024,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    inputs = {key: value.to(_get_device()) for key, value in inputs.items()}

    with torch_module.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return clean_text(summary)





# ==========================================================
# Attention Mechanism Extraction
# ==========================================================


def extract_attention(text: str, model_name: str):
    """
    Extracts encoder and decoder attention weights.

    Used for explaining the transformer encoder-decoder flow.
    """
    torch_module = _load_torch_backend()
    if torch_module is None:
        return {"encoder_attention": None, "decoder_attention": None}

    model = load_seq2seq_model(model_name)
    tokenizer = load_tokenizer(model_name)

    tokens = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    tokens = {key: value.to(_get_device()) for key, value in tokens.items()}

    with torch_module.no_grad():
        outputs = model(
            **tokens,
            output_attentions=True,
            return_dict=True,
        )

    encoder_attention = outputs.encoder_attentions
    decoder_attention = outputs.decoder_attentions

    return {
        "encoder_attention": encoder_attention,
        "decoder_attention": decoder_attention,
    }




# ==========================================================
# Model Comparison Helper
# ==========================================================


def compare_models():
    print("\nAvailable Transformer Models\n")

    for name, path in MODELS.items():
        print(name, "---->", path)


# ==========================================================
# Evaluation Metrics
# ==========================================================

def calculate_rouge(reference: str, generated: str):
    if rouge_scorer is None:
        return None

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, generated)





# ==========================================================
# BLEU Score
# ==========================================================


def calculate_bleu(reference: str, generated: str):
    if sentence_bleu is None:
        return None

    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    return sentence_bleu(reference_tokens, generated_tokens)





# ==========================================================
# Perplexity
# ==========================================================


def calculate_perplexity(text: str, model_name: str):
    torch_module = _load_torch_backend()
    if torch_module is None:
        return None

    model = load_seq2seq_model(model_name)
    tokenizer = load_tokenizer(model_name)

    if model is None or tokenizer is None:
        return None

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    tokens = {key: value.to(_get_device()) for key, value in tokens.items()}

    with torch_module.no_grad():
        output = model(**tokens, labels=tokens["input_ids"])

    loss = output.loss.item()

    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None

    return perplexity





# ==========================================================
# BERTScore
# ==========================================================


def calculate_bert_score(reference: str, generated: str):
    bert_score_module = _load_bert_score()
    if bert_score_module is None:
        return None

    try:
        _, _, f1 = bert_score_module([generated], [reference], lang="en", model_type="distilbert-base-uncased")
        return float(f1.mean())
    except Exception as exc:
        logging.warning("BERTScore could not be computed: %s", exc)
        return None





# ==========================================================
# Complete Evaluation
# ==========================================================

def evaluate_model(dataset, model_name, limit=10):
    results = []
    samples = dataset[:limit]

    for index, item in enumerate(samples, 1):
        article = item["article"]
        reference = item["summary"]

        logging.info(f"Processing example {index}")

        generated = generate_summary(article, model_name)
        rouge = calculate_rouge(reference, generated)
        bleu = calculate_bleu(reference, generated)
        perplexity = calculate_perplexity(generated, model_name)
        bert = calculate_bert_score(reference, generated)

        row = {
            "id": item["id"],
            "reference": reference,
            "generated": generated,
            "bleu": bleu,
            "perplexity": perplexity,
            "bert_score": bert,
        }

        if rouge:
            row.update(
                {
                    "rouge1": rouge["rouge1"].fmeasure,
                    "rouge2": rouge["rouge2"].fmeasure,
                    "rougeL": rouge["rougeL"].fmeasure,
                }
            )

        results.append(row)

        print("\n======================")
        print("Example:", index)
        print("\nReference:\n", reference)
        print("\nGenerated:\n", generated)

        if rouge:
            print("\nROUGE-1:", round(rouge["rouge1"].fmeasure, 4))
            print("ROUGE-2:", round(rouge["rouge2"].fmeasure, 4))
            print("ROUGE-L:", round(rouge["rougeL"].fmeasure, 4))

        print("BLEU:", bleu)
        print("Perplexity:", perplexity)
        print("BERTScore:", bert)

    return results





# ==========================================================
# Console Output Helpers
# ==========================================================


def print_model_summary(model_name: str, results: List[Dict]):
    """Print a compact summary for a model run to the console."""
    print(f"\n=== {model_name.upper()} ===")
    for idx, row in enumerate(results, 1):
        print(f"Example {idx}:")
        print(f"  Summary: {row.get('generated', '')}")
        print(f"  BLEU: {row.get('bleu')}")
        print(f"  ROUGE-1: {row.get('rouge1')}")
        print(f"  ROUGE-2: {row.get('rouge2')}")
        print(f"  ROUGE-L: {row.get('rougeL')}")
        print("-" * 40)


# ==========================================================
# Model Comparison Experiment
# ==========================================================


def run_model_suite(dataset, limit=10):
    """Run all supported models, print results to the console, and skip CSV output."""
    all_results = {}

    for model_key, model_path in MODELS.items():
        display_name = {
            "bart": "BART",
            "t5": "T5",
            "pegasus": "PEGASUS",
        }.get(model_key, model_key.upper())

        print(f"\nRunning Model: {display_name}")
        print(f"Model Path: {model_path}")

        results = evaluate_model(dataset, model_path, limit)
        all_results[model_key] = results
        print_model_summary(display_name, results)
        print(f"Completed {display_name}")

    return all_results





# ==========================================================
# Command Line Arguments
# ==========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Based Text Summarization using Transformer Encoder Decoder")
    parser.add_argument("--dataset", default="validation.csv", help="CNN/DailyMail dataset split")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles")
    parser.add_argument("--model", default="bart", choices=["bart", "t5", "pegasus", "all"], help="Transformer model")
    return parser.parse_args()





# ==========================================================
# Main Execution
# ==========================================================

def main():
    args = parse_arguments()
    dataset_file = DATA_PATH / args.dataset

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    print("\nLoading Dataset...")
    dataset = load_dataset(dataset_file)
    print("Total Documents:", len(dataset))

    if args.model != "all":
        model_path = MODELS[args.model]
        results = evaluate_model(dataset, model_path, args.limit)
        print_model_summary(args.model.upper(), results)
    else:
        run_model_suite(dataset, args.limit)

    print("\nExecution Completed")





# ==========================================================
# Program Entry Point
# ==========================================================


if __name__ == "__main__":
    main()


# ==========================================================
# Real execution log (2026-08-08), after fixing the
# local_files_only=True bug that previously made every model
# silently fall back to the extractive tfidf_summary() path.
# Command: python summarization.py --model all --limit 50 --dataset validation.csv
# (run in chunked batches per model to fit CPU runtime per execution)
# Dataset: 50 real CNN/DailyMail validation examples (incl. the
# Sally Forrest article) pulled from the abisee/cnn_dailymail
# dataset on Hugging Face.
# ==========================================================

# ======================
# Example: 1 (Sally Forrest)

# Reference:
#  Sally Forrest, an actress-dancer who graced the silver screen throughout the '40s and '50s in MGM musicals and films died on March 15 . Forrest, whose birth name was Katherine Feeney, had long battled cancer . A San Diego native, Forrest became a protege of Hollywood trailblazer Ida Lupino, who cast her in starring roles in films .

# === BART (facebook/bart-large-cnn) ===
#  Forrest, whose birth name was Katherine Feeney, was 86 and had long battled cancer. A San Diego native, Forrest became a protege of Hollywood trailblazer Ida Lupino, who cast her in starring roles in films including Not Wanted, Never Fear and Hard, Fast and Beautiful. Some of Forrest's other film credits included Bannerline, Son of Sinbad, and Excuse My Dust.
#   ROUGE-1: 0.6154  ROUGE-2: 0.5217  ROUGE-L: 0.5470  BLEU: 0.4565  Perplexity: 1.2958  BERTScore: 0.8481

# === T5 (t5-base) ===
#  Sally Forrest, an actress-dancer who graced the silver screen throughout the '40s and '50s in MGM musicals and films such as the 1956 noir While the City Sleeps, died on March 15 at her home in Beverly Hills, California . Forrest, whose birth name was Katherine Feeney, was 86 and had long battled cancer .
#   ROUGE-1: 0.6727  ROUGE-2: 0.5741  ROUGE-L: 0.6364  BLEU: 0.5323

# === PEGASUS (google/pegasus-cnn_dailymail) ===
#  Sally Forrest graced the silver screen throughout the '40s and '50s in MGM musicals and films such as the 1956 noir While the City Sleeps .<n>Forrest, whose birth name was Katherine Feeney, was 86 and had long cancer .<n>A San Diego native, Forrest became a protege of Hollywood trailblazer Ida Lupino, who cast her in starring roles in films .
#   ROUGE-1: 0.8034  ROUGE-2: 0.7130  ROUGE-L: 0.8034  BLEU: 0.6402

# Note: PEGASUS emits a literal "<n>" token in place of newlines;
# this is a known tokenizer artifact, not a data error.

# ==========================================================
# 50-example averages per model (see Group_15_NLP_Assignment2.pdf
# section 3.2 for the full table and section 3.3 for the ROUGE-1
# distribution across all 50 examples):
#   BART:    ROUGE-1 0.3942  ROUGE-2 0.1998  ROUGE-L 0.3074  BLEU 0.1070  Perplexity 1.2483  BERTScore 0.8182
#   T5:      ROUGE-1 0.3179  ROUGE-2 0.1356  ROUGE-L 0.2311  BLEU 0.0605  Perplexity 1.2490  BERTScore 0.7899
#   PEGASUS: ROUGE-1 0.3841  ROUGE-2 0.1751  ROUGE-L 0.2904  BLEU 0.0903  Perplexity 1.2095  BERTScore 0.8048
# ==========================================================

# Execution Completed