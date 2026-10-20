"""Resumable MedCPT comparison; fixed shapes, tiny batches, atomic checkpoints."""

import argparse, json, time
import numpy as np
import torch
from common import *
from neural import sync
from safe_inference import SafeEncoder
from extended_experiments import EXP, BASE


def hybrid_eval():
    from score_fusion import generate

    return generate(
        "eval", json.loads((EXP / "hybrid_selection.json").read_text())["config"]
    )


def main(finalize, val_k=50, device="mps", batch_size=2):
    seed_all()
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    ref = json.loads((EXP / "medcpt_cross_encoder_reference.json").read_text())
    model = SafeEncoder(
        "reranker",
        device,
        checkpoint=ref["model"],
        revision=ref["revision"],
        batch_size=batch_size,
    )
    model.metadata.update(license=ref["license"], authors_input=ref["authors_input"])

    # Exactly follow the authors' query/article pair and longest-first truncation.
    def tokenize(inputs):
        return model.tokenizer(
            [x[0] for x in inputs],
            [x[1] for x in inputs],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    model.tokenize = tokenize
    sample = [(claims["val"][0]["claim"], document_text(d)) for d in docs[:2]]
    gpu = model.encode(sample)
    model.model.to("cpu")
    model.device = "cpu"
    cpu = model.encode(sample)
    model.model.to(device)
    model.device = device
    difference = float(np.max(np.abs(gpu - cpu)))
    assert difference < 0.01
    sources = {
        "baseline_hybrid": json.loads((BASE / "hybrid_val_rankings.json").read_text()),
        "improved_hybrid": json.loads(
            (EXP / "improved_hybrid_val_rankings.json").read_text()
        ),
    }
    pairs = {}
    for ranks in sources.values():
        for c in inference_claims(claims["val"]):
            for d in ranks[str(c["id"])][:val_k]:
                pairs[(c["id"], d)] = (c["claim"], document_text(bydoc[d]))
    identity = {
        "revision": ref["revision"],
        "inputs": digest([inference_claims(claims["val"]), docs]),
        "parameters": model.parameters,
        "pair_order": "claim, article",
        "max_length": 512,
        "truncation": "longest_first",
    }
    path = EXP / "medcpt_val_scores.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    if path.exists():
        assert json.loads((EXP / "medcpt_val_config.json").read_text()) == identity

    def score_incrementally(keys, inputs, cache, cache_path, label):
        missing = [key for key in keys if key not in cache]
        start = time.perf_counter()
        completed = 0
        for i in range(0, len(missing), 32):
            chunk = missing[i : i + 32]
            values = model.encode([inputs[key] for key in chunk]).reshape(-1)
            cache.update({key: float(value) for key, value in zip(chunk, values)})
            save_json(cache_path, cache)
            completed += len(chunk)
            save_json(
                EXP / f"{label}_progress.json",
                {
                    "missing_at_start": len(missing),
                    "new_completed": completed,
                    "total_cached": len(cache),
                    "elapsed_sec": time.perf_counter() - start,
                    "model": model.metadata,
                    "cpu_mps_max_abs_difference": difference,
                },
            )
            print(
                label,
                completed,
                "/",
                len(missing),
                "cached",
                len(cache),
                "sec",
                round(time.perf_counter() - start, 1),
                flush=True,
            )
        return cache

    inputs = {f"{c}:{d}": text for (c, d), text in pairs.items()}
    save_json(EXP / "medcpt_val_config.json", identity)
    scores = score_incrementally(sorted(inputs), inputs, scores, path, "medcpt_val")
    comparisons = []
    for system, ranks in sources.items():
        for k in [cutoff for cutoff in [20, 50] if cutoff <= val_k]:
            result = {
                cid: rerank_prefix(rank, {d: scores[f"{cid}:{d}"] for d in rank[:k]}, k)
                for cid, rank in ranks.items()
            }
            metrics = retrieval_metrics(result, claims["val"])
            name = f"medcpt_{system}_k{k}"
            save_json(EXP / f"{name}_val_rankings.json", result)
            entry = {
                "name": name,
                "system": system,
                "k": k,
                "metrics": metrics,
                "model": model.metadata,
            }
            comparisons.append(entry)
            print(name, metrics["ndcg10"], flush=True)
    save_json(EXP / "medcpt_rerank_comparisons.json", comparisons)
    winner = max(
        [x for x in comparisons if x["system"] == "improved_hybrid"],
        key=lambda x: (x["metrics"]["ndcg10"], -x["k"]),
    )
    assert (
        winner["metrics"]["ndcg10"]
        >= json.loads((BASE / "rerank_metadata.json").read_text())["metrics"]["ndcg10"]
    )
    save_json(EXP / "medcpt_rerank_selection.json", winner)
    # Make validation ranking available only after selection; no eval influenced it.
    save_json(
        EXP / "improved_rerank_val_rankings.json",
        json.loads((EXP / f"{winner['name']}_val_rankings.json").read_text()),
    )
    if finalize:
        k = winner["k"]
        ranks = sources["improved_hybrid"]
        times = []
        model.encode(
            [
                (claims["val"][0]["claim"], document_text(bydoc[d]))
                for d in ranks[str(claims["val"][0]["id"])][:k]
            ]
        )
        for c in inference_claims(claims["val"])[:30]:
            inputs = [
                (c["claim"], document_text(bydoc[d])) for d in ranks[str(c["id"])][:k]
            ]
            sync(device)
            start = time.perf_counter()
            model.encode(inputs)
            sync(device)
            times.append((time.perf_counter() - start) * 1000)
        winner["fresh_rerank_query_latency_ms"] = float(np.mean(times))
        winner["latency_samples_ms"] = times
        save_json(EXP / "medcpt_rerank_selection.json", winner)
        save_json(
            EXP / "retrieval_selection_frozen.json",
            {
                "selected_reranker": winner,
                "selected_hybrid": json.loads(
                    (EXP / "hybrid_selection.json").read_text()
                ),
                "evaluation_used_for_selection": False,
            },
        )
        ranks = hybrid_eval()
        inputs = {
            f"{c['id']}:{d}": (c["claim"], document_text(bydoc[d]))
            for c in inference_claims(claims["eval"])
            for d in ranks[str(c["id"])][:k]
        }
        eval_path = EXP / f"medcpt_eval_k{k}_scores.json"
        eval_scores = json.loads(eval_path.read_text()) if eval_path.exists() else {}
        eval_scores = score_incrementally(
            sorted(inputs), inputs, eval_scores, eval_path, "medcpt_eval"
        )
        save_json(EXP / "medcpt_eval_scores.json", eval_scores)
        result = {
            cid: rerank_prefix(
                rank, {d: eval_scores[f"{cid}:{d}"] for d in rank[:k]}, k
            )
            for cid, rank in ranks.items()
        }
        save_json(EXP / "improved_rerank_eval_rankings.json", result)
    model.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--val-k", type=int, default=50, choices=[20, 50])
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--batch-size", type=int, default=2)
    a = p.parse_args()
    main(a.finalize, a.val_k, a.device, a.batch_size)
