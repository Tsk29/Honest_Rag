import json

import pytest

from eval.run_eval import DATASET_PATH, compute_metrics, load_dataset, mock_complete, parse_relevance


def test_parse_relevance_strips_think_blocks():
    assert parse_relevance("<think>maybe yes, maybe no</think>\nno") == "no"
    assert parse_relevance("<think>hmm</think> Yes, it is relevant.") == "yes"
    assert parse_relevance("NO") == "no"


def test_compute_metrics():
    results = [
        {"label": "yes", "prediction": "yes"},  # tp
        {"label": "yes", "prediction": "no"},   # fn
        {"label": "no", "prediction": "yes"},   # fp
        {"label": "no", "prediction": "no"},    # tn
        {"label": "no", "prediction": "no"},    # tn
    ]
    m = compute_metrics(results)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 2, 1)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_compute_metrics_empty_is_all_zero():
    m = compute_metrics([])
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == 0.0


def test_shipped_dataset_is_well_formed():
    examples = load_dataset()
    assert len(examples) >= 10
    for ex in examples:
        assert ex["query"].strip() and ex["document_chunk"].strip()
        assert ex["label"] in {"yes", "no"}
    labels = {ex["label"] for ex in examples}
    assert labels == {"yes", "no"}, "dataset needs both positive and negative examples"


def test_load_dataset_skips_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    row = {"query": "q", "document_chunk": "c", "label": "yes"}
    path.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n")
    assert len(load_dataset(str(path))) == 2


def test_mock_complete_is_deterministic():
    assert mock_complete("capital city France", "The capital city of France is Paris") == "yes"
    assert mock_complete("capital city France", "Bananas are yellow") == "no"
    assert DATASET_PATH.endswith("dataset.jsonl")
