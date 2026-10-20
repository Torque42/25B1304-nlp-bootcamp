"""Data validation, deterministic metrics, ranking, and artifact IO."""

import csv, hashlib, json, math, os, random, subprocess, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PRED = ROOT / "predictions"
SEED = 42
CATEGORIES = {
    "retrieval_miss",
    "negation",
    "numerical",
    "entity_mismatch",
    "needs_multiple_docs",
    "other",
}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)


def load_data(data_dir=None):
    data_dir = Path(data_dir or os.environ.get("NLP_DATA_DIR") or ROOT / "data")
    docs = sorted(jsonl(data_dir / "corpus.jsonl"), key=lambda x: x["doc_id"])
    claims = {
        s: sorted(jsonl(data_dir / f"{s}_claims.jsonl"), key=lambda x: x["id"])
        for s in ["train", "val", "eval"]
    }
    assert len(docs) == 5183 and len({d["doc_id"] for d in docs}) == len(docs)
    bydoc = {d["doc_id"]: d for d in docs}
    for split, rows in claims.items():
        assert len(rows) == {"train": 647, "val": 162, "eval": 300}[split]
        assert len({c["id"] for c in rows}) == len(rows)
        for c in rows:
            assert type(c["id"]) is int and isinstance(c["claim"], str)
            if split == "eval":
                assert "evidence" not in c
            for e in c.get("evidence", []):
                assert e["doc_id"] in bydoc and e["label"] in {"SUPPORT", "CONTRADICT"}
                assert all(
                    0 <= s < len(bydoc[e["doc_id"]]["abstract"]) for s in e["sentences"]
                )
    return docs, claims


def inference_claims(rows):
    """Gold labels are deliberately dropped at the inference boundary."""
    return [{"id": c["id"], "claim": c["claim"]} for c in rows]


def document_text(d):
    return d["title"] + " " + " ".join(d["abstract"])


def gold_pairs(c):
    return {(e["doc_id"], e["label"]) for e in c["evidence"]}


def order_scores(scores, doc_ids):
    assert np.isfinite(scores).all()
    return np.lexsort((np.asarray(doc_ids), -np.asarray(scores)))


def retrieval_metrics(rankings, claims):
    per = []
    for c in claims:
        gold = {e["doc_id"] for e in c["evidence"]}
        ranked = rankings[str(c["id"])][:100]
        if not gold:
            per.append({"claim_id": c["id"], "excluded": True})
            continue
        dcg = sum((d in gold) / math.log2(r + 2) for r, d in enumerate(ranked[:10]))
        ideal = sum(1 / math.log2(r + 2) for r in range(min(len(gold), 10)))
        rr = next((1 / (r + 1) for r, d in enumerate(ranked) if d in gold), 0)
        per.append(
            {
                "claim_id": c["id"],
                "excluded": False,
                "ndcg10": dcg / ideal,
                "recall100": len(gold & set(ranked)) / len(gold),
                "mrr100": rr,
            }
        )
    eligible = [p for p in per if not p["excluded"]]
    means = {
        k: float(np.mean([p[k] for p in eligible]))
        for k in ["ndcg10", "recall100", "mrr100"]
    }
    return {
        **means,
        "denominator": len(eligible),
        "excluded_nei": len(per) - len(eligible),
        "per_claim": per,
    }


def triple_metrics(pred, claims):
    gold = {(c["id"], d, l) for c in claims for d, l in gold_pairs(c)}
    pred = set(tuple(x) for x in pred)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0
    recall = tp / len(gold) if gold else 0
    errors = [
        c["id"]
        for c in claims
        if {(d, l) for cid, d, l in pred if cid == c["id"]} != gold_pairs(c)
    ]
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0,
        "true_positives": tp,
        "false_positives": len(pred - gold),
        "false_negatives": len(gold - pred),
        "exact_set_accuracy": 1 - len(errors) / len(claims),
        "error_count": len(errors),
        "error_ids": errors,
        "denominator_claims": len(claims),
    }


def save_run(name, split, rankings, scores=None):
    ART.mkdir(exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    save_json(ART / f"{name}_{split}_rankings.json", rankings)
    if scores is not None:
        save_json(ART / f"{name}_{split}_raw_scores.json", scores)
    filenames = {
        ("bm25", "eval"): "t1_bm25.trec",
        ("bm25", "val"): "val_bm25.trec",
        ("general", "eval"): "t2_general.trec",
        ("general", "val"): "val_general.trec",
        ("scientific", "eval"): "t2_scientific.trec",
        ("hybrid", "eval"): "t3_hybrid.trec",
        ("rerank", "eval"): "t3_rerank.trec",
        ("random", "eval"): "t4_random.trec",
        ("hard", "eval"): "t4_hard.trec",
    }
    filename = filenames.get((name, split))
    if filename:
        with (PRED / filename).open("w") as f:
            for cid in sorted(rankings, key=int):
                for r, did in enumerate(rankings[cid][:100], 1):
                    score = scores[cid][r - 1] if scores is not None else 101 - r
                    f.write(f"{cid} Q0 {did} {r} {score:.9g} {name}\n")


def read_run(name, split):
    return json.loads((ART / f"{name}_{split}_rankings.json").read_text())


def record_stage(name, info):
    save_json(ART / f"{name}_metadata.json", info)
    print(
        json.dumps(
            {
                "stage": name,
                "metrics": {
                    k: v for k, v in info.get("metrics", {}).items() if k != "per_claim"
                },
                "query_latency_ms": info.get("query_latency_ms"),
            }
        ),
        flush=True,
    )


def choose_best(names):
    data = [(n, json.loads((ART / f"{n}_metadata.json").read_text())) for n in names]
    best = max(d["metrics"]["ndcg10"] for _, d in data)
    tied = [(n, d) for n, d in data if abs(d["metrics"]["ndcg10"] - best) <= 1e-8]
    return min(
        enumerate(tied),
        key=lambda x: (x[1][1].get("query_latency_ms", float("inf")), x[0]),
    )[1][0]


def rrf(rank_a, rank_b, k=60, limit=None):
    scores = {}
    for ranking in [rank_a, rank_b]:
        for r, d in enumerate(ranking[:limit] if limit else ranking, 1):
            scores[d] = scores.get(d, 0) + 1 / (k + r)
    return sorted(scores, key=lambda d: (-scores[d], d))


def rerank_prefix(ranking, score_map, k):
    prefix = sorted(
        enumerate(ranking[:k]), key=lambda x: (-score_map[x[1]], x[0], x[1])
    )
    return [d for _, d in prefix] + ranking[k:]


def initialize_manifest():
    """Capture provenance for a fresh reproduction before running any stage."""
    path = ART / "input_manifest.json"
    if path.exists():
        return
    data_dir = Path(os.environ.get("NLP_DATA_DIR") or ROOT / "data")
    protected = [
        ROOT / name
        for name in [
            "specs.md",
            "NLP_PS.pdf",
            "check_format.py",
            "results_template.json",
        ]
        if (ROOT / name).exists()
    ]
    protected += list(data_dir.glob("*.jsonl"))
    files = {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): file_hash(p)
        for p in protected
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        status = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
        )
    except FileNotFoundError:
        commit = status = None  # Git is optional in an extracted submission.
    save_json(
        path,
        {
            "protected_file_hashes": files,
            "code_base_commit": commit.stdout.strip()
            if commit is not None and commit.returncode == 0
            else None,
            "initial_git_status": status.stdout.strip()
            if status is not None and status.returncode == 0
            else None,
            "original_spec_preserved": (ROOT / "specs.md").is_file(),
            "nondeterminism": "Seed 42; eager float32. MPS is not guaranteed bitwise reproducible.",
        },
    )
