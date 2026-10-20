"""Incrementally score validation prefixes and freeze cutoff before evaluation."""

import argparse, json, time
import numpy as np
from common import *
from neural import Encoder, sync


def pair_cache(model, public, rankings, bydoc, k, split, system):
    config = {
        "model": {
            key: value
            for key, value in model.metadata.items()
            if key != "load_time_sec"
        },
        "input_hash": digest([public, bydoc]),
        "system": system,
        "first_stage_hash": digest(rankings),
        "max_length": 512,
        "truncation": "only_second",
        "prefix": k,
    }
    # k is stored in each checkpoint; earlier scores are reusable only for exact inputs/model.
    identity = {key: value for key, value in config.items() if key != "prefix"}
    path = ART / "cache" / f"rerank-{split}-{digest(identity)}.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    missing = []
    inputs = []
    for c in public:
        for r, d in enumerate(rankings[str(c["id"])][:k], 1):
            key = f"{c['id']}:{d}"
            if key not in cached:
                missing.append((key, c["id"], d, r))
                inputs.append((c["claim"], document_text(bydoc[d])))
    if inputs:
        tokens = model.token_metadata(inputs)
        t = time.perf_counter()
        logits = model.encode(inputs, 8, True).reshape(-1)
        sync(model.device)
        elapsed = time.perf_counter() - t
        for (key, cid, d, r), value, tok in zip(missing, logits, tokens):
            cached[key] = {
                "claim_id": cid,
                "doc_id": d,
                "first_stage_rank": r,
                "raw_logit": float(value),
                "tokens": tok,
            }
        save_json(path, cached)
        print(f"Scored {len(inputs)} new {split} pairs in {elapsed:.2f}s", flush=True)
    return cached


def main(k, device, finalize):
    seed_all()
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    system = json.loads((ART / "first_stage_selection.json").read_text())["winner"]
    model = Encoder("reranker", device)
    rankings = read_run(system, "val")
    public = inference_claims(claims["val"])
    scores = pair_cache(model, public, rankings, bydoc, k, "val", system)
    reranked = {
        cid: rerank_prefix(
            rank, {d: scores[f"{cid}:{d}"]["raw_logit"] for d in rank[:k]}, k
        )
        for cid, rank in rankings.items()
    }
    metrics = retrieval_metrics(reranked, claims["val"])
    # Fresh overhead measurement for the same 30 sorted claims, excluding retrieval.
    model.encode(
        [
            (public[0]["claim"], document_text(bydoc[d]))
            for d in rankings[str(public[0]["id"])][:k]
        ],
        8,
    )
    times = []
    for c in public[:30]:
        inputs = [
            (c["claim"], document_text(bydoc[d])) for d in rankings[str(c["id"])][:k]
        ]
        sync(device)
        t = time.perf_counter()
        model.encode(inputs, 8)
        sync(device)
        times.append((time.perf_counter() - t) * 1000)
    baseline = json.loads((ART / f"{system}_metadata.json").read_text())
    overhead = float(np.mean(times))
    info = {
        "config": {
            "first_stage": system,
            "k": k,
            "max_length": 512,
            "truncation": "only_second",
            "export_score": "101-rank",
            "model": model.metadata,
        },
        "metrics": metrics,
        "rerank_query_latency_ms": overhead,
        "query_latency_ms": baseline["query_latency_ms"] + overhead,
        "latency_samples_ms": times,
        "latency_convention": "fresh warmed pair tokenization/inference overhead; total latency is sum with first-stage mean",
    }
    save_json(ART / f"rerank_k{k}_metadata.json", info)
    save_jsonl(
        ART / f"rerank_k{k}_val_scores.jsonl",
        [scores[f"{c['id']}:{d}"] for c in public for d in rankings[str(c["id"])][:k]],
    )
    save_json(ART / f"rerank_k{k}_val_rankings.json", reranked)
    print(
        f"Rerank k={k}: nDCG {metrics['ndcg10']:.6f}, overhead {overhead:.2f} ms",
        flush=True,
    )
    if finalize:
        candidates = sorted(
            ART.glob("rerank_k*_metadata.json"),
            key=lambda p: int(p.name.split("_")[1][1:]),
        )
        complete = [json.loads(p.read_text()) for p in candidates]
        best = max(x["metrics"]["ndcg10"] for x in complete)
        winner = min(
            [x for x in complete if abs(x["metrics"]["ndcg10"] - best) <= 1e-8],
            key=lambda x: x["rerank_query_latency_ms"],
        )
        chosen = winner["config"]["k"]
        save_json(
            ART / "rerank_selection.json",
            {
                "selected_k": chosen,
                "first_stage": system,
                "selection_split": "validation",
                "frozen_before_evaluation": True,
            },
        )
        save_run(
            "rerank",
            "val",
            json.loads((ART / f"rerank_k{chosen}_val_rankings.json").read_text()),
        )
        eval_rank = read_run(system, "eval")
        eval_public = inference_claims(claims["eval"])
        eval_scores = pair_cache(
            model, eval_public, eval_rank, bydoc, chosen, "eval", system
        )
        result = {
            cid: rerank_prefix(
                rank,
                {d: eval_scores[f"{cid}:{d}"]["raw_logit"] for d in rank[:chosen]},
                chosen,
            )
            for cid, rank in eval_rank.items()
        }
        save_run("rerank", "eval", result)
        save_jsonl(
            ART / "rerank_eval_scores.jsonl",
            [
                eval_scores[f"{c['id']}:{d}"]
                for c in eval_public
                for d in eval_rank[str(c["id"])][:chosen]
            ],
        )
        record_stage("rerank", winner)
    model.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=20, choices=[20, 50, 100])
    p.add_argument("--device", default="mps")
    p.add_argument("--finalize", action="store_true")
    a = p.parse_args()
    main(a.k, a.device, a.finalize)
