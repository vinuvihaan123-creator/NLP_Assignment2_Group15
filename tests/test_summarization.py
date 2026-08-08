import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarization


def test_generate_summary_falls_back_to_extractive_when_transformers_are_unavailable():
    article = "A major storm hit the city yesterday. Power outages affected thousands of residents. Schools were closed for the day."
    summary = summarization.generate_summary(article, "facebook/bart-large-cnn")
    assert isinstance(summary, str)
    assert summary
    assert "storm" in summary.lower()


def test_model_and_tokenizer_loaders_fall_back_when_pretrained_download_fails(monkeypatch):
    class FailingTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("download failed")

    class FailingModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("download failed")

    monkeypatch.setattr(summarization, "AutoTokenizer", FailingTokenizer)
    monkeypatch.setattr(summarization, "AutoModelForSeq2SeqLM", FailingModel)
    monkeypatch.setattr(summarization, "TOKENIZER_CACHE", {})
    monkeypatch.setattr(summarization, "MODEL_CACHE", {})

    assert summarization.load_tokenizer("fake-model") is None
    assert summarization.load_seq2seq_model("fake-model") is None
