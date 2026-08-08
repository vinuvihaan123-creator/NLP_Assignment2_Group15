import builtins
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarization as summarization_module
from text_summarization_demo import clean_text, get_model_settings, simple_extractive_summary, summarize_text


def test_clean_text_collapses_whitespace():
    text = "Hello   world\n\nfrom  python"
    assert clean_text(text) == "Hello world from python"


def test_simple_extractive_summary_returns_sentence_subset():
    article = "A major storm hit the city yesterday. Power outages affected thousands of residents. Schools were closed for the day."
    summary = simple_extractive_summary(article, max_sentences=2)
    assert len(summary.split(".")) >= 1
    assert "storm" in summary.lower()


def test_summarize_text_returns_non_empty_summary():
    article = "A major storm hit the city yesterday. Power outages affected thousands of residents. Schools were closed for the day."
    summary = summarize_text(article, use_transformer=False, max_sentences=2)
    assert isinstance(summary, str)
    assert summary
    assert "storm" in summary.lower()


def test_get_model_settings_supports_t5_and_pegasus():
    t5_settings = get_model_settings("t5-small")
    pegasus_settings = get_model_settings("google/pegasus-cnn_dailymail")

    assert t5_settings["model_name"] == "t5-small"
    assert t5_settings["prefix"] == "summarize:"
    assert pegasus_settings["model_name"] == "google/pegasus-cnn_dailymail"
    assert pegasus_settings["prefix"] is None


def test_importing_summarization_does_not_fail_when_torch_interrupts(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise KeyboardInterrupt
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("summarization", None)

    module = importlib.import_module("summarization")

    assert module.torch is None


def test_run_model_suite_prints_results_without_csv(monkeypatch, capsys):
    sample_dataset = [{"article": "A storm hit the town.", "summary": "The town was hit by a storm."}]

    def fake_evaluate_model(dataset, model_name, limit):
        return [{"model": model_name, "limit": limit, "sample": dataset[0]["summary"]}]

    monkeypatch.setattr(summarization_module, "evaluate_model", fake_evaluate_model)

    results = summarization_module.run_model_suite(sample_dataset, limit=1)
    captured = capsys.readouterr()

    assert results["bart"][0]["model"] == "facebook/bart-large-cnn"
    assert "Running Model: BART" in captured.out
    assert "Completed BART" in captured.out
