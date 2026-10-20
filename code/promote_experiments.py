"""Promote only validation-selected experiments; version review before replacing outputs."""

import argparse, json, shutil, time
from common import *
from extended_experiments import EXP, BASE, infer_variant
from experiment_calibration import aggregate_windows, sweep, calibrated_classify
from verification import make_review, write_verification

NLI_VARIANTS = {
    "finetuned_full": ("full", EXP / "checkpoints" / "finetuned_full"),
    "finetuned_lexical_windows": (
        "lexical_windows",
        EXP / "checkpoints" / "finetuned_lexical_windows",
    ),
    "zeroshot_all_windows": ("all_windows", None),
    "zeroshot_lexical_windows": ("lexical_windows", None),
}


def nli_on_selected_retrieval():
    # Compare the two trained encoders on the actual final retrieval, with reuse at the pair level.
    docs, claims = load_data()
    new = json.loads((EXP / "improved_rerank_val_rankings.json").read_text())
    k = 10
    for name in ["finetuned_full", "finetuned_lexical_windows"]:
        mode, checkpoint = NLI_VARIANTS[name]
        old = jsonl(EXP / f"{name}_hybrid_val_windows.jsonl")
        lookup = {}
        for row in old:
            lookup.setdefault((row["claim_id"], row["doc_id"]), []).append(row)
        reused = []
        missing = {}
        byclaim = {c["id"]: c for c in inference_claims(claims["val"])}
        for cid, ranking in new.items():
            missing[cid] = []
            for rank, d in enumerate(ranking[:k], 1):
                if (int(cid), d) in lookup:
                    reused.extend(
                        {**r, "retrieval_rank": rank} for r in lookup[(int(cid), d)]
                    )
                else:
                    missing[cid].append(d)
        # Score only new claim/doc inputs, then attach actual ranks without exposing gold labels.
        system = "new_pairs_" + name
        save_json(EXP / f"{system}_val_rankings.json", missing)
        if any(missing.values()):
            infer_variant(name, mode, checkpoint, system, k, "val")
            additional = jsonl(EXP / f"{name}_{system}_val_windows.jsonl")
            for r in additional:
                r["retrieval_rank"] = new[str(r["claim_id"])].index(r["doc_id"]) + 1
            reused += additional
        for aggregation in ["max", "mean"]:
            key = f"{name}_improved_rerank_{aggregation}"
            pairs = aggregate_windows(reused, aggregation)
            save_jsonl(EXP / f"{key}_val_pairs.jsonl", pairs)
            selection = sweep(pairs, key)
            selection.update(
                name=name,
                mode=mode,
                aggregation=aggregation,
                checkpoint=str(checkpoint),
            )
            save_json(EXP / f"{key}_final_candidate.json", selection)
    choices = [
        json.loads(p.read_text()) for p in sorted(EXP.glob("*_final_candidate.json"))
    ]
    best = max(
        choices,
        key=lambda x: (
            x["metrics"]["f1"],
            x["metrics"]["precision"],
            -x["candidate_k"],
            x["name"] == "finetuned_full",
        ),
    )
    save_json(EXP / "final_verification_selection.json", best)
    print(
        "Final NLI selection",
        best["name"],
        best["metrics"]["f1"],
        best["candidate_k"],
        best["decision_config"],
        flush=True,
    )


def materialize():
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    hybrid = json.loads((EXP / "hybrid_selection.json").read_text())
    rerank = json.loads((EXP / "medcpt_rerank_selection.json").read_text())
    selection = json.loads((EXP / "final_verification_selection.json").read_text())
    name = selection["name"]
    mode = selection["mode"]
    aggregation = selection["aggregation"]
    k = selection["candidate_k"]
    checkpoint = selection["checkpoint"]
    # Freeze all verification decisions before evaluation encoding.
    save_json(
        EXP / "final_selection_frozen.json",
        {
            "hybrid": hybrid,
            "rerank": rerank,
            "verification": selection,
            "evaluation_used_for_selection": False,
        },
    )
    # Archive current review state, including directly edited CSVs and predictions.
    old_reviews = [
        ART / f"validation_error_review.{suffix}" for suffix in ["html", "csv", "jsonl"]
    ] + [ART / "nli_selection.json", PRED / "t5_val_verification.csv"]
    present = [p for p in old_reviews if p.exists()]
    fingerprint = digest({p.name: file_hash(p) for p in present})[:16]
    archive = ART / "reviews" / ("before_promotion_" + fingerprint)
    archive.mkdir(parents=True, exist_ok=True)
    for old in present:
        shutil.copy2(old, archive / old.name)
    for old in [ART / "human_review.json", PRED / "t5_val_errors.csv"]:
        if old.exists():
            destination = archive / (old.stem + "_" + file_hash(old)[:16] + old.suffix)
            old.rename(destination)
    for split in ["val", "eval"]:
        rank = json.loads((EXP / f"improved_hybrid_{split}_rankings.json").read_text())
        raw = json.loads((EXP / f"improved_hybrid_{split}_raw_scores.json").read_text())
        save_run("hybrid", split, rank, raw)
        save_run(
            "rerank",
            split,
            json.loads((EXP / f"improved_rerank_{split}_rankings.json").read_text()),
        )
    hybrid["config"]["dense"] = "scientific"
    record_stage("hybrid", hybrid)
    save_json(ART / "first_stage_selection.json", {"winner": "hybrid"})
    info = {
        "config": {
            "first_stage": "hybrid",
            "k": rerank["k"],
            "max_length": 512,
            "truncation": "longest_first",
            "export_score": "101-rank",
            "model": rerank["model"],
        },
        "metrics": rerank["metrics"],
        "rerank_query_latency_ms": rerank["fresh_rerank_query_latency_ms"],
        "query_latency_ms": hybrid["query_latency_ms"]
        + rerank["fresh_rerank_query_latency_ms"],
        "latency_convention": "sum of measured component means and fresh warmed30-query reranking overhead",
    }
    record_stage("rerank", info)
    save_json(
        ART / "rerank_selection.json",
        {
            "selected_k": rerank["k"],
            "first_stage": "hybrid",
            "model": rerank["model"],
            "frozen_before_evaluation": True,
        },
    )
    val_pairs = jsonl(EXP / f"{name}_improved_rerank_{aggregation}_val_pairs.jsonl")
    triples, val_decisions = calibrated_classify(
        val_pairs, k, selection["decision_config"]
    )
    model_meta = json.loads((EXP / f"{name}_training.json").read_text())["model"]
    model_meta["source"] = "cross-encoder/nli-deberta-v3-small"
    model_meta["checkpoint"] = checkpoint
    model_meta["training_device"] = model_meta["device"]
    model_meta["device"] = os.environ.get("NLP_DEVICE", "mps")
    model_meta["inference_padding"] = "fixed512"
    model_meta["inference_batch_size"] = 2
    nli_info = {
        "retriever": "rerank",
        "first_stage": "hybrid",
        "candidate_k": k,
        "threshold": selection["decision_config"].get(
            "threshold", selection["decision_config"].get("evidence_threshold")
        ),
        "decision_config": selection["decision_config"],
        "aggregation": aggregation,
        "input_mode": mode,
        "max_rows": 3,
        "model": model_meta,
        "metrics": triple_metrics(triples, claims["val"]),
        "zero_shot": False,
        "frozen_before_evaluation": True,
    }
    assert nli_info["metrics"] == selection["metrics"]
    save_json(ART / "nli_selection.json", nli_info)
    write_verification("val", triples)
    save_jsonl(ART / "nli_val_trace.jsonl", val_decisions)
    # Evaluation only runs with the frozen NLI candidate count/decision rule.
    infer_variant(name, mode, checkpoint, "improved_rerank", k, "eval")
    window_rows = jsonl(EXP / f"{name}_improved_rerank_eval_windows.jsonl")
    eval_pairs = aggregate_windows(window_rows, aggregation)
    eval_triples, eval_decisions = calibrated_classify(
        eval_pairs, k, selection["decision_config"]
    )
    write_verification("eval", eval_triples)
    save_jsonl(ART / "nli_eval_trace.jsonl", eval_decisions)
    review = make_review(
        claims["val"],
        bydoc,
        read_run("rerank", "val"),
        val_decisions,
        triples,
        nli_info,
    )
    version = ART / "reviews" / "window_20261013T124402Z"
    version.mkdir(parents=True, exist_ok=True)
    for suffix in ["html", "csv", "jsonl"]:
        shutil.copy2(
            ART / f"validation_error_review.{suffix}",
            version / f"validation_error_review.{suffix}",
        )
    record_stage("nli", {"config": nli_info, "metrics": nli_info["metrics"]})
    for split in ["val", "eval"]:
        first = read_run("hybrid", split)
        final = read_run("rerank", split)
        raw = json.loads((ART / f"hybrid_{split}_raw_scores.json").read_text())
        save_jsonl(
            ART / f"verification_{split}_retrieval_trace.jsonl",
            [
                {
                    "claim_id": int(cid),
                    "doc_id": d,
                    "first_stage_rank": first[cid].index(d) + 1,
                    "first_stage_raw_score": raw[cid][first[cid].index(d)]
                    if d in first[cid][:100]
                    else None,
                    "final_rank": r,
                    "final_rank_score": 101 - r,
                    "sent_to_nli": r <= k,
                }
                for cid, rank in final.items()
                for r, d in enumerate(rank[:100], 1)
            ],
        )
    print("Promoted pipeline; human review pending", len(review), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["validate_nli", "materialize"])
    a = p.parse_args()
    if a.stage == "validate_nli":
        nli_on_selected_retrieval()
    else:
        materialize()
