Dataset link: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

# AI-Based Text Summarization System

An AI-based text summarization system built around the CNN/DailyMail dataset, using pre-trained
Transformer encoder-decoder models (BART, T5, PEGASUS) from Hugging Face for abstractive
summarization, with an extractive (TF-IDF sentence-scoring) fallback when `torch`/`transformers`
or the models themselves are unavailable.

## What is included?
- `summarization.py`: loads a CNN/DailyMail-format CSV, generates abstractive summaries with
  BART (`facebook/bart-large-cnn`), T5 (`t5-base`), or PEGASUS (`google/pegasus-cnn_dailymail`),
  and evaluates them with ROUGE-1/2/L, BLEU, Perplexity, and BERTScore.
- `tests/test_summarization.py`: unit tests covering the extractive fallback path.
- `run_chunk.py`: helper for running `evaluate_model()` over a slice of the dataset (`start:end`),
  used to process large runs in resumable batches on CPU.
- `parsed_results_50.json`: raw per-example results (reference, generated summary, and all
  metrics) from a 50-article run per model.
- `Group_15_NLP_Assignment2.pdf`: the project report, including the real model comparison.

## Setup
Install dependencies (no `requirements.txt` is currently pinned in this repo):
```bash
pip install torch transformers rouge_score nltk bert_score sentencepiece
```
Models are downloaded from the Hugging Face Hub on first use and cached locally.

## Dataset
`summarization.py` expects a CSV with `id`, `article`, and `highlights` columns, placed at
`cnn_dailymail/<name>.csv` (this folder is gitignored, so bring your own copy). Get it from
Kaggle (link above) or the Hugging Face mirror, e.g.:
```python
from datasets import load_dataset
load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation[:50]")
```

## Run
```bash
python summarization.py --model bart --limit 10 --dataset validation.csv
python summarization.py --model all --limit 50 --dataset validation.csv   # run all 3 models
```
`--model` accepts `bart`, `t5`, `pegasus`, or `all`. `--limit` caps how many articles to process.

## Notes
If `torch`/`transformers` aren't installed, or a model fails to load (e.g. no network access and
nothing cached), `generate_summary()` silently falls back to `tfidf_summary()`, a simple
extractive baseline. A previous version of this code always hit that fallback due to
`from_pretrained(..., local_files_only=True)` blocking downloads entirely; that flag has since
been removed so the abstractive models run for real.
