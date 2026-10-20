"""Cached-logit calibration diagnostics; never fits encoder weights."""

import argparse, collections, json
import numpy as np
from common import *
from extended_experiments import EXP, BASE, LABELS
from verification import classify


def calibrated_classify(rows, k, config):
    if config["rule"] == "neutral_gate":
        return classify(rows, k, config["threshold"])
    decisions = []
    grouped = collections.defaultdict(list)
    for r in rows:
        if r["retrieval_rank"] > k:
            continue
        p = np.asarray(r["probabilities"])
        label = int(np.argmax(p[:2]))
        evidence = float(p[:2].sum())
        conditional = float(p[label] / max(evidence, 1e-12))
        reason = (
            "below_evidence_threshold"
            if evidence < config["evidence_threshold"]
            else "below_conditional_threshold"
            if conditional < config["conditional_threshold"]
            else "accepted_before_cap"
        )
        item = {
            **r,
            "chosen_label": LABELS[label],
            "chosen_probability": float(p[label]),
            "evidence_probability": evidence,
            "conditional_label_probability": conditional,
            "threshold": config["evidence_threshold"],
            "decision": reason,
            "decision_rule": config,
        }
        decisions.append(item)
        if reason == "accepted_before_cap":
            grouped[r["claim_id"]].append(item)
    triples = []
    for cid, items in grouped.items():
        items.sort(
            key=lambda x: (-x["chosen_probability"], x["retrieval_rank"], x["doc_id"])
        )
        for rank, item in enumerate(items):
            item["decision"] = "accepted" if rank < 3 else "three_row_cap"
            if rank < 3:
                triples.append((cid, item["doc_id"], item["chosen_label"]))
    return sorted(triples, key=lambda x: x[0]), decisions


def aggregate_windows(window_rows, aggregation):
    docs, _ = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    groups = collections.defaultdict(list)
    for row in window_rows:
        groups[(row["claim_id"], row["doc_id"])].append(row)
    result = []
    for key, items in groups.items():
        best = max(
            items,
            key=lambda r: (
                max(r["probabilities"][:2]),
                -r["window"]["start_sentence"]
                if r["window"]["start_sentence"] is not None
                else 0,
            ),
        )
        if aggregation == "mean":
            prob = np.mean([x["probabilities"] for x in items], axis=0).tolist()
        else:
            prob = best["probabilities"]
        visible = set()
        for r in items:
            doc = bydoc[r["doc_id"]]
            first = r["window"]["start_sentence"]
            last = r["window"]["end_sentence"]
            first = 0 if first is None else first
            last = len(doc["abstract"]) if last is None else last
            position = len(doc["title"]) + 1
            for index in range(first, last):
                end = position + len(doc["abstract"][index])
                if end <= r["tokens"]["premise_visible_char_end"]:
                    visible.add(index)
                position = end + 1
        result.append(
            {
                **best,
                "tokens": {
                    **best["tokens"],
                    "visible_document_sentence_indices": sorted(visible),
                },
                "probabilities": prob,
                "aggregation": aggregation,
                "windows_examined": len(items),
                "window_traces": items,
            }
        )
    return result


def sweep(rows, name):
    _, claims = load_data()
    results = []
    ks = [k for k in [1, 3, 5, 10, 20] if k <= max(r["retrieval_rank"] for r in rows)]
    configs = [
        {"rule": "neutral_gate", "threshold": t}
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ]
    configs += [
        {"rule": "two_stage", "evidence_threshold": e, "conditional_threshold": c}
        for e in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
        for c in [0.5, 0.7, 0.9]
    ]
    for k in ks:
        for cfg in configs:
            pred, _ = calibrated_classify(rows, k, cfg)
            m = triple_metrics(pred, claims["val"])
            results.append({"candidate_k": k, "decision_config": cfg, "metrics": m})
    best = max(
        enumerate(results),
        key=lambda ix: (
            ix[1]["metrics"]["f1"],
            ix[1]["metrics"]["precision"],
            -ix[1]["candidate_k"],
            -ix[0],
        ),
    )[1]
    save_json(EXP / f"{name}_calibration_sweep.json", results)
    save_json(EXP / f"{name}_calibration_selection.json", best)
    print(
        name,
        best["metrics"]["f1"],
        best["candidate_k"],
        best["decision_config"],
        flush=True,
    )
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name")
    a = p.parse_args()
    baseline = jsonl(BASE / "nli_val_trace.jsonl")
    sweep(baseline, "baseline_full")
    paths = list(EXP.glob("*_val_windows.jsonl"))
    for path in paths:
        if a.name and not path.name.startswith(a.name):
            continue
        rows = jsonl(path)
        name = path.name.removesuffix("_val_windows.jsonl")
        for aggregation in ["max", "mean"]:
            sweep(aggregate_windows(rows, aggregation), name + "_" + aggregation)
    winners = []
    for path in EXP.glob("*_calibration_selection.json"):
        d = json.loads(path.read_text())
        winners.append(
            {"name": path.name.removesuffix("_calibration_selection.json"), **d}
        )
    if winners:
        save_json(
            EXP / "verification_experiment_leaderboard.json",
            sorted(
                winners,
                key=lambda x: (
                    -x["metrics"]["f1"],
                    -x["metrics"]["precision"],
                    x["candidate_k"],
                    x["name"],
                ),
            ),
        )


if __name__ == "__main__":
    main()
