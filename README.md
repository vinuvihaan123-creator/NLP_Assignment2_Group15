Dataset link: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

# AI-Based Text Summarization System

This workspace contains a lightweight transformer-style text summarization demo built around the CNN/DailyMail CSV files in the `cnn_dailymail` folder.

## What is included?
- A Python script that loads the validation split and generates sample summaries from article text.
- A simple extractive summarization approach that selects the most informative sentences.
- A clear entry point for extending the solution to pretrained transformer models such as T5, BART, or PEGASUS.

## Files
- `text_summarization_demo.py`: runnable summarization demo
- `cnn_dailymail/validation.csv`: dataset used for the sample run
- https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

## Run
```bash
cd /Users/vinothinivinu/Downloads/NLP2
/opt/homebrew/bin/python3.13 text_summarization_demo.py

```

## Notes
The current implementation uses an extractive approach for reliable local execution. It can be upgraded to pretrained abstractive models such as `facebook/bart-large-cnn` or `google/pegasus-cnn_dailymail` for more human-like summaries.
