import math, sys
from pathlib import Path
import numpy as np
import pytest

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base if (base / "common.py").exists() else base / "code"))
from common import (
    retrieval_metrics,
    triple_metrics,
    order_scores,
    rrf,
    rerank_prefix,
    inference_claims,
)
from retrieval import BM25
from training import sample_triplets
from verification import classify


def test_hand_calculable_metrics_and_nei_exclusion():
    claims = [
        {
            "id": 1,
            "evidence": [
                {"doc_id": 9, "label": "SUPPORT"},
                {"doc_id": 4, "label": "CONTRADICT"},
            ],
        },
        {"id": 2, "evidence": []},
    ]
    score = retrieval_metrics({"1": [8, 9, 4], "2": [9, 4, 8]}, claims)
    assert score["ndcg10"] == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )
    assert score["recall100"] == 1 and score["mrr100"] == 0.5
    assert score["denominator"] == 1 and score["excluded_nei"] == 1


def test_wrong_label_counts_fp_and_fn_empty_nei_exact():
    claims = [
        {"id": 1, "evidence": [{"doc_id": 9, "label": "SUPPORT"}]},
        {"id": 2, "evidence": []},
    ]
    score = triple_metrics([(1, 9, "CONTRADICT")], claims)
    assert score["false_positives"] == score["false_negatives"] == 1
    assert score["f1"] == 0 and score["exact_set_accuracy"] == 0.5
    assert triple_metrics([], claims)["precision"] == 0


def test_ties_empty_queries_query_frequency_and_formula():
    docs = ["a a", "b"]
    index = BM25(docs, 0.9, 0.4)
    assert np.array_equal(index.score("a a"), index.score("a"))
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * 1.9 / (2 + 0.9 * (0.6 + 0.4 * 2 / 1.5))
    assert index.score("a")[0] == pytest.approx(expected)
    assert order_scores(index.score(""), [9, 4]).tolist() == [1, 0]
    assert order_scores(np.array([3, 3, 1]), [9, 4, 1]).tolist() == [1, 0, 2]


def test_negative_pools_exclude_all_gold_including_contradictions():
    docs = [{"doc_id": i} for i in range(1, 6)]
    claims = [
        {
            "id": 7,
            "evidence": [
                {"doc_id": 1, "label": "SUPPORT"},
                {"doc_id": 2, "label": "CONTRADICT"},
            ],
        }
    ]
    for method in ["random", "hard"]:
        samples = sample_triplets(method, docs, claims, {"7": [1, 2, 3, 4]})
        assert len(samples) == 2 and all(
            t["negative_doc_id"] not in {1, 2} for t in samples
        )
        assert samples == sample_triplets(method, docs, claims, {"7": [1, 2, 3, 4]})


def test_rrf_and_reranker_tail_preservation():
    assert rrf([1, 2, 3], [3, 2, 1])[0] == 1  # 1 and 3 tie; document-ID break
    ranked = rerank_prefix([1, 2, 3, 4, 5], {1: 1, 2: 3, 3: 3}, 3)
    assert ranked == [2, 3, 1, 4, 5] and len(set(ranked)) == 5


def test_gold_dropped_at_inference_boundary():
    assert inference_claims(
        [{"id": 1, "claim": "text", "evidence": [{"doc_id": 3}]}]
    ) == [{"id": 1, "claim": "text"}]


def test_nli_neutral_threshold_and_cap():
    rows = [
        {
            "claim_id": 1,
            "doc_id": i,
            "retrieval_rank": i,
            "probabilities": [0.05, 0.9 - i * 0.01, 0.05 + i * 0.01],
        }
        for i in range(1, 5)
    ]
    rows.append(
        {
            "claim_id": 2,
            "doc_id": 9,
            "retrieval_rank": 1,
            "probabilities": [0.1, 0.1, 0.8],
        }
    )
    triples, trace = classify(rows, 5, 0.8)
    assert len(triples) == 3 and all(t[0] == 1 and t[2] == "SUPPORT" for t in triples)
    assert trace[3]["decision"] == "three_row_cap" and trace[4]["decision"] == "neutral"
    assert not classify(rows, 5, 0.99)[0]


def test_human_review_import_rejects_blank_or_incomplete_coverage(
    tmp_path, monkeypatch
):
    import verification
    from common import save_jsonl

    monkeypatch.setattr(verification, "ART", tmp_path)
    monkeypatch.setattr(verification, "PRED", tmp_path)
    save_jsonl(tmp_path / "validation_error_review.jsonl", [{"claim_id": 1}])
    review = tmp_path / "review.csv"
    review.write_text("claim_id,final_category,reviewer_note\n1,,\n")
    with pytest.raises(ValueError, match="missing or invalid"):
        verification.export_review(review)
    assert not (tmp_path / "t5_val_errors.csv").exists()
    review.write_text("claim_id,final_category,reviewer_note\n2,other,test fixture\n")
    with pytest.raises(ValueError, match="Coverage mismatch"):
        verification.export_review(review)
    assert not (tmp_path / "t5_val_errors.csv").exists()


def test_window_visibility_uses_document_indices(monkeypatch):
    import experiment_calibration

    doc = {"doc_id": 1, "title": "Title", "abstract": ["alpha", "beta", "gamma"]}
    monkeypatch.setattr(experiment_calibration, "load_data", lambda: ([doc], {}))
    rows = [
        {
            "claim_id": 1,
            "doc_id": 1,
            "retrieval_rank": 1,
            "premise": "Title gamma",
            "hypothesis": "claim",
            "window": {"start_sentence": 2, "end_sentence": 3, "text": "Title gamma"},
            "tokens": {"premise_visible_char_end": 11},
            "probabilities": [0.1, 0.8, 0.1],
            "logits": [1, 3, 1],
        }
    ]
    aggregated = experiment_calibration.aggregate_windows(rows, "max")
    assert aggregated[0]["tokens"]["visible_document_sentence_indices"] == [2]
    assert len(aggregated[0]["window_traces"]) == 1


def test_two_stage_gate_is_distinct_from_neutral_argmax():
    from experiment_calibration import calibrated_classify

    rows = [
        {
            "claim_id": 1,
            "doc_id": 1,
            "retrieval_rank": 1,
            "probabilities": [0.01, 0.04, 0.95],
        }
    ]
    config = {
        "rule": "two_stage",
        "evidence_threshold": 0.01,
        "conditional_threshold": 0.7,
    }
    triples, trace = calibrated_classify(rows, 1, config)
    assert triples == [(1, 1, "SUPPORT")]
    assert trace[0]["decision_rule"] == config
    assert not classify(rows, 1, 0.3)[0]
    assert not calibrated_classify(rows, 1, {**config, "evidence_threshold": 0.1})[0]


def test_selected_resume_scores_validation_when_weights_already_exist(
    tmp_path, monkeypatch
):
    import run_selected
    import sys

    exp = tmp_path / "experiments"
    for name in ["finetuned_full", "finetuned_lexical_windows"]:
        checkpoint = exp / "checkpoints" / name / "model.safetensors"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture")
    (tmp_path / "code").mkdir()
    (tmp_path / "code/selected_recipe.json").write_text("{}")
    monkeypatch.setattr(run_selected, "ROOT", tmp_path)
    monkeypatch.setattr(run_selected, "EXP", exp)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_selected.py",
            "--device",
            "cpu",
            "--from-stage",
            "nli_training",
            "--through-stage",
            "nli_training",
        ],
    )
    calls = []
    monkeypatch.setattr(
        run_selected.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    run_selected.main()
    assert len(calls) == 2
    assert all("infer_variant(" in call[0][-1] for call in calls)
    assert all(call[1]["env"]["NLP_DEVICE"] == "cpu" for call in calls)
