"""The eight BM25 settings, exact dense retrieval, and complete-ranking RRF."""

import argparse, collections, json, re, time
import numpy as np
from common import *


class BM25:
    def __init__(self, texts, k1, b):
        tokens = [re.findall(r"[^\W_]+", t.lower()) for t in texts]
        self.n = len(texts)
        self.k1 = k1
        self.b = b
        self.lengths = np.array([len(t) for t in tokens])
        self.avgdl = self.lengths.mean()
        postings = collections.defaultdict(list)
        for i, ts in enumerate(tokens):
            for term, tf in collections.Counter(ts).items():
                postings[term].append((i, tf))
        self.postings = {}
        for term, rows in postings.items():
            indices, tf = np.array(rows).T
            idf = np.log(1 + (self.n - len(rows) + 0.5) / (len(rows) + 0.5))
            values = (
                idf
                * tf
                * (k1 + 1)
                / (tf + k1 * (1 - b + b * self.lengths[indices] / self.avgdl))
            )
            self.postings[term] = (indices, values)

    def score(self, query):
        scores = np.zeros(self.n, dtype=np.float64)
        for term in set(re.findall(r"[^\W_]+", query.lower())):
            if term in self.postings:
                indices, values = self.postings[term]
                scores[indices] += values
        return scores


def rank_matrix(scores, claims, doc_ids):
    rankings = {}
    top_scores = {}
    for c, row in zip(claims, scores):
        order = order_scores(row, doc_ids)
        cid = str(c["id"])
        rankings[cid] = [int(doc_ids[j]) for j in order]
        top_scores[cid] = [float(row[j]) for j in order[:100]]
    return rankings, top_scores


def bm25():
    docs, claims = load_data()
    ids = np.array([d["doc_id"] for d in docs])
    sweep = []
    best = None
    for text_mode in ["title_abstract", "abstract"]:
        texts = [
            document_text(d)
            if text_mode == "title_abstract"
            else " ".join(d["abstract"])
            for d in docs
        ]
        for k1 in [0.9, 1.5]:
            for b in [0.4, 0.75]:
                start = time.perf_counter()
                index = BM25(texts, k1, b)
                ranked, _ = rank_matrix(
                    [index.score(c["claim"]) for c in inference_claims(claims["val"])],
                    claims["val"],
                    ids,
                )
                metrics = retrieval_metrics(ranked, claims["val"])
                times = []
                for c in inference_claims(claims["val"])[:30]:
                    startq = time.perf_counter()
                    order_scores(index.score(c["claim"]), ids)[:100]
                    times.append((time.perf_counter() - startq) * 1000)
                config = {
                    "text": text_mode,
                    "k1": k1,
                    "b": b,
                    "tokenizer": "lowercase Unicode alphanumeric runs excluding underscore",
                    "idf": "log(1+(N-df+0.5)/(df+0.5))",
                    "query_term_frequency": False,
                }
                item = {
                    "config": config,
                    "metrics": metrics,
                    "query_latency_ms": float(np.mean(times)),
                    "sweep_time_sec": time.perf_counter() - start,
                }
                sweep.append(item)
                print("BM25", config, metrics["ndcg10"], flush=True)
                if (
                    best is None
                    or metrics["ndcg10"] > best["metrics"]["ndcg10"] + 1e-8
                    or abs(metrics["ndcg10"] - best["metrics"]["ndcg10"]) <= 1e-8
                    and item["query_latency_ms"] < best["query_latency_ms"]
                ):
                    best = item
    save_json(ART / "bm25_sweep.json", sweep)
    config = best["config"]
    index = BM25(
        [
            document_text(d)
            if config["text"] == "title_abstract"
            else " ".join(d["abstract"])
            for d in docs
        ],
        config["k1"],
        config["b"],
    )
    for split, rows in claims.items():
        rankings, scores = rank_matrix(
            [index.score(c["claim"]) for c in inference_claims(rows)], rows, ids
        )
        save_run("bm25", split, rankings, scores)
    record_stage("bm25", best)


def dense(name, device, checkpoint=None, fresh_index=False):
    from neural import Encoder, cache_embeddings, peak_rss_mb, sync

    seed_all()
    docs, claims = load_data()
    ids = np.array([d["doc_id"] for d in docs])
    start = time.perf_counter()
    model = Encoder(
        "article" if name == "scientific" else "general", device, checkpoint
    )
    doc_inputs = (
        [(d["title"], " ".join(d["abstract"])) for d in docs]
        if name == "scientific"
        else [document_text(d) for d in docs]
    )
    doc_meta = model.token_metadata(doc_inputs)
    config = {
        "document_text": "title/abstract pair"
        if name == "scientific"
        else "title + space + abstract sentences joined with spaces",
        "exact_search": True,
        "seed": 42,
    }
    if checkpoint:
        config["checkpoint_sha256"] = file_hash(Path(checkpoint) / "model.safetensors")
    sync(device)
    index_start = time.perf_counter()
    matrix, cached, key = cache_embeddings(model, doc_inputs, config, force=fresh_index)
    matrix = np.ascontiguousarray(matrix)
    sync(device)
    index_time = time.perf_counter() - index_start
    previous_path = ART / f"{name}_metadata.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    measurement_origin = "fresh indexing in this process"
    if cached:
        if (
            previous.get("embedding_cache_key") == key
            and previous.get("index_time_sec") is not None
        ):
            index_time = previous["index_time_sec"]
            measurement_origin = (
                "retained prior cold measurement for identical embedding cache key"
            )
        else:
            # A cache without a matching measured run cannot supply a cold timing.
            sync(device)
            index_start = time.perf_counter()
            matrix, cached, key = cache_embeddings(
                model, doc_inputs, config, force=True
            )
            matrix = np.ascontiguousarray(matrix)
            sync(device)
            index_time = time.perf_counter() - index_start
    doc_model = model.metadata
    if name == "scientific":
        model.close()
        model = Encoder("query", device)
    truncation = {
        "documents": {
            "truncated": sum(x["truncated"] for x in doc_meta),
            "total": len(doc_meta),
        }
    }
    save_jsonl(
        ART / f"{name}_document_tokens.jsonl",
        [{"doc_id": d["doc_id"], **m} for d, m in zip(docs, doc_meta)],
    )
    for split, rows in claims.items():
        public = inference_claims(rows)
        inputs = [c["claim"] for c in public]
        meta = model.token_metadata(inputs)
        truncation[split] = {
            "truncated": sum(x["truncated"] for x in meta),
            "total": len(meta),
        }
        queries, _, qkey = cache_embeddings(
            model, inputs, {**config, "split": split, "role": "claim"}
        )
        scores = queries @ matrix.T
        rankings, tops = rank_matrix(scores, public, ids)
        save_run(name, split, rankings, tops)
    # These are fresh uncached single-query executions, including tokenize/encode/search/select.
    model.encode([claims["val"][0]["claim"]])
    sync(device)
    latencies = []
    for c in inference_claims(claims["val"])[:30]:
        sync(device)
        t = time.perf_counter()
        query = model.encode([c["claim"]])
        score = query @ matrix.T
        order_scores(score[0], ids)[:100]
        sync(device)
        latencies.append((time.perf_counter() - t) * 1000)
    metrics = retrieval_metrics(read_run(name, "val"), claims["val"])
    info = {
        "training_retrieval_metrics": retrieval_metrics(
            read_run(name, "train"), claims["train"]
        ),
        "metrics": metrics,
        "document_encoder": doc_model,
        "query_encoder": model.metadata,
        "config": config,
        "index_time_sec": index_time,
        "index_measurement_origin": measurement_origin,
        "index_from_cache": cached,
        "embedding_cache_key": key,
        "query_latency_ms": float(np.mean(latencies)),
        "latency_samples_ms": latencies,
        "latency_convention": "30 sorted validation claims, warmed, fresh tokenization/encoding/scoring/top100; synchronized MPS; excludes model load",
        "memory_mb": previous["memory_mb"] if cached else peak_rss_mb(),
        "query_process_peak_rss_mb": peak_rss_mb(),
        "memory_measurement_origin": "prior cold index/query process"
        if cached
        else "fresh index/query process",
        "memory_convention": "peak process RSS MiB over model loading/index/query, macOS ru_maxrss bytes",
        "truncation": truncation,
        "stage_time_sec": time.perf_counter() - start,
    }
    if device == "mps":
        import torch

        info.update(
            mps_allocated_bytes=torch.mps.current_allocated_memory(),
            mps_driver_bytes=torch.mps.driver_allocated_memory(),
        )
    record_stage(name, info)
    model.close()


def hybrid():
    docs, claims = load_data()
    dense_name = choose_best(["general", "scientific"])
    for split in claims:
        a, b = read_run("bm25", split), read_run(dense_name, split)
        ranks = {cid: rrf(a[cid], b[cid]) for cid in a}
        positions_a = {cid: {d: r for r, d in enumerate(a[cid], 1)} for cid in a}
        positions_b = {cid: {d: r for r, d in enumerate(b[cid], 1)} for cid in b}
        scores = {
            cid: [
                1 / (60 + positions_a[cid][d]) + 1 / (60 + positions_b[cid][d])
                for d in ranks[cid][:100]
            ]
            for cid in ranks
        }
        save_run("hybrid", split, ranks, scores)
    metrics = retrieval_metrics(read_run("hybrid", "val"), claims["val"])
    times = []
    a, b = read_run("bm25", "val"), read_run(dense_name, "val")
    for c in claims["val"][:30]:
        t = time.perf_counter()
        rrf(a[str(c["id"])], b[str(c["id"])])
        times.append((time.perf_counter() - t) * 1000)
    retrieval_latency = sum(
        json.loads((ART / f"{n}_metadata.json").read_text())["query_latency_ms"]
        for n in ["bm25", dense_name]
    ) + float(np.mean(times))
    record_stage(
        "hybrid",
        {
            "metrics": metrics,
            "config": {
                "dense": dense_name,
                "k": 60,
                "weights": [1, 1],
                "fuse_complete_corpus": True,
            },
            "query_latency_ms": retrieval_latency,
            "fusion_overhead_ms": float(np.mean(times)),
            "latency_convention": "sum of fresh constituent means plus cached-ranking fusion overhead",
        },
    )
    save_json(
        ART / "first_stage_selection.json",
        {"winner": choose_best(["bm25", "general", "scientific", "hybrid"])},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["bm25", "general", "scientific", "hybrid", "random", "hard"]
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--checkpoint")
    p.add_argument(
        "--fresh-index",
        action="store_true",
        help="Re-encode corpus to refresh cold index/memory measurements",
    )
    args = p.parse_args()
    if args.stage == "bm25":
        bm25()
    elif args.stage == "hybrid":
        hybrid()
    else:
        dense(args.stage, args.device, args.checkpoint, args.fresh_index)
