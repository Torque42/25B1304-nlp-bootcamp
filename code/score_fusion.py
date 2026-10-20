"""Identical non-oracle normalized score fusion for train, validation, and eval."""

import json, time
import numpy as np
from common import *
from retrieval import BM25, rank_matrix
from extended_experiments import EXP, BASE

COMPONENTS = ["bm25", "scientific", "general"]


def component_scores(split):
    docs, claims = load_data()
    cache = ART / "cache"
    metadata = {}
    for p in cache.glob("*.json"):
        value = json.loads(p.read_text())
        if "shape" in value:
            metadata[p.stem] = value
    cfg = json.loads((BASE / "bm25_metadata.json").read_text())["config"]
    index = BM25(
        [
            document_text(d)
            if cfg["text"] == "title_abstract"
            else " ".join(d["abstract"])
            for d in docs
        ],
        cfg["k1"],
        cfg["b"],
    )
    scores = {
        "bm25": np.array(
            [index.score(c["claim"]) for c in inference_claims(claims[split])]
        )
    }
    for name in ["scientific", "general"]:
        q = [
            k
            for k, v in metadata.items()
            if v["model"]["kind"] == ("query" if name == "scientific" else "general")
            and v["configuration"].get("split") == split
            and "checkpoint_sha256" not in v["configuration"]
        ]
        d = [
            k
            for k, v in metadata.items()
            if v["model"]["kind"] == ("article" if name == "scientific" else "general")
            and v["shape"][0] == 5183
            and "checkpoint_sha256" not in v["configuration"]
        ]
        assert len(q) == len(d) == 1, (name, split)
        scores[name] = (
            np.load(cache / (q[0] + ".npy")) @ np.load(cache / (d[0] + ".npy")).T
        )
    return docs, claims[split], scores


def normalize(scores, method):
    if method == "minmax":
        lo = scores.min(-1, keepdims=True)
        scale = scores.max(-1, keepdims=True) - lo
    elif method == "zscore":
        lo = scores.mean(-1, keepdims=True)
        scale = scores.std(-1, keepdims=True)
    else:
        cutoff = int(method.split("_")[0][3:])
        lo = np.sort(scores, axis=-1)[..., -cutoff, None]
        scale = scores.max(-1, keepdims=True) - lo
    return (scores - lo) / np.maximum(scale, 1e-12)


def generate(split, config):
    docs, claims, scores = component_scores(split)
    combined = sum(
        w * normalize(scores[n], config["normalization"])
        for w, n in zip(config["weights"], COMPONENTS)
    )
    rankings, raw = rank_matrix(
        combined, inference_claims(claims), [d["doc_id"] for d in docs]
    )
    save_json(EXP / f"improved_hybrid_{split}_rankings.json", rankings)
    save_json(EXP / f"improved_hybrid_{split}_raw_scores.json", raw)
    return rankings


def adopt():
    imported = json.loads((EXP / "planning_score_fusion_broad_sweep.json").read_text())
    winner = imported["best"]
    current = json.loads((EXP / "hybrid_selection.json").read_text())
    save_json(EXP / "rrf_hybrid_selection.json", current)
    if winner["metrics"]["ndcg10"] > current["metrics"]["ndcg10"] + 1e-8:
        selected = {
            "config": {
                "method": "score_fusion",
                "normalization": winner["normalization"],
                "weights": winner["weights"],
                "components": COMPONENTS,
                "clip_normalized_scores": False,
            },
            "metrics": winner["metrics"],
            "source": "planning CPU sweep, independently replicated here",
        }
        ranks = generate("val", selected["config"])
        _, claims = load_data()
        measured = retrieval_metrics(ranks, claims["val"])
        assert abs(measured["ndcg10"] - winner["metrics"]["ndcg10"]) < 1e-8
        selected["metrics"] = measured
        save_json(EXP / "hybrid_selection.json", selected)
        generate("train", selected["config"])
        _, _, scores = component_scores("val")
        times = []
        for i in range(30):
            start = time.perf_counter()
            combined = sum(
                w * normalize(scores[n][i : i + 1], selected["config"]["normalization"])
                for w, n in zip(selected["config"]["weights"], COMPONENTS)
            )
            np.lexsort((np.arange(combined.shape[1]), -combined[0]))
            times.append((time.perf_counter() - start) * 1000)
        selected["fusion_overhead_ms"] = float(np.mean(times))
        selected["query_latency_ms"] = (
            sum(
                json.loads((ART / f"{n}_metadata.json").read_text())["query_latency_ms"]
                for n in COMPONENTS
            )
            + selected["fusion_overhead_ms"]
        )
        selected["latency_convention"] = (
            "sum of component fresh-query means plus fresh fusion normalization/sort overhead on30 sorted validation queries"
        )
        save_json(EXP / "hybrid_selection.json", selected)
        print(
            "Replicated score fusion",
            selected["config"],
            measured["ndcg10"],
            flush=True,
        )


if __name__ == "__main__":
    adopt()
