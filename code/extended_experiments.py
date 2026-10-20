"""Versioned experiments authorized separately from the original specification."""

import argparse, collections, datetime, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from common import *
from neural import Encoder, sync, cache_embeddings
from safe_inference import SafeEncoder
from transformers import get_linear_schedule_with_warmup
from verification import classify, sentence_spans, LABELS, THRESHOLDS

DEVICE = os.environ.get("NLP_DEVICE", "mps")
EXP = ART / "experiments" / "window_20261013T124402Z"
BASE = ART / "snapshots" / "baseline_20261013T124402Z" / "artifacts"
EXP.mkdir(parents=True, exist_ok=True)


def softmax(logits):
    p = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def window_candidates(doc):
    s = doc["abstract"]
    return [
        {
            "start_sentence": i,
            "end_sentence": min(i + 3, len(s)),
            "text": doc["title"] + " " + " ".join(s[i : i + 3]),
        }
        for i in range(0, len(s), 2)
    ] or [{"start_sentence": 0, "end_sentence": 0, "text": doc["title"]}]


def selected_windows(doc, claim, count=3):
    import re

    tokens = lambda t: set(re.findall(r"[^\W_]+", t.lower()))
    query = tokens(claim)
    windows = window_candidates(doc)
    # Pure claim-text overlap; sentence annotations never enter selection.
    ranked = sorted(
        enumerate(windows),
        key=lambda iw: (
            -len(query & tokens(iw[1]["text"])),
            -len(
                query
                & tokens(
                    " ".join(
                        doc["abstract"][iw[1]["start_sentence"] : iw[1]["end_sentence"]]
                    )
                )
            ),
            iw[0],
        ),
    )
    return [w for _, w in ranked[:count]]


def threshold_sweep(rows, name, ks):
    results = []
    _, claims = load_data()
    for k in ks:
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            triples, decisions = classify(rows, k, threshold)
            metric = triple_metrics(triples, claims["val"])
            results.append(
                {"candidate_k": k, "threshold": threshold, "metrics": metric}
            )
    winner = max(
        results,
        key=lambda x: (
            x["metrics"]["f1"],
            x["metrics"]["precision"],
            x["threshold"],
            -x["candidate_k"],
        ),
    )
    save_json(EXP / f"{name}_sweep.json", results)
    save_json(EXP / f"{name}_selection.json", winner)
    print(
        name,
        "F1",
        winner["metrics"]["f1"],
        "k",
        winner["candidate_k"],
        "threshold",
        winner["threshold"],
        flush=True,
    )
    return winner


def audit():
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    rankings = json.loads((BASE / "hybrid_val_rankings.json").read_text())
    rows = jsonl(BASE / "nli_val_trace.jsonl")
    trace = {(r["claim_id"], r["doc_id"]): r for r in rows}
    cover = {}
    total = sum(len(gold_pairs(c)) for c in claims["val"])
    for k in [1, 3, 5, 10, 20, 100]:
        oracle = [
            (c["id"], d, l)
            for c in claims["val"]
            for d, l in sorted(gold_pairs(c))
            if d in rankings[str(c["id"])][:k]
        ][:]
        cover[k] = {
            "covered_gold_pairs": len(oracle),
            "total_gold_pairs": total,
            "pair_coverage": len(oracle) / total,
            "oracle_exact_triple_metrics": triple_metrics(oracle, claims["val"]),
        }
    confusion = {lab: {pred: 0 for pred in LABELS} for lab in ["SUPPORT", "CONTRADICT"]}
    visible = {
        "fully_visible": collections.Counter(),
        "truncated": collections.Counter(),
    }
    rejects = collections.Counter()
    for c in claims["val"]:
        for d, gold in gold_pairs(c):
            r = trace.get((c["id"], d))
            if r:
                pred = LABELS[int(np.argmax(r["probabilities"]))]
                confusion[gold][pred] += 1
                rejects[r["decision"]] += 1
                es = [
                    s for e in c["evidence"] if e["doc_id"] == d for s in e["sentences"]
                ]
                full = all(
                    sentence_spans(bydoc[d])[s][1]
                    <= r["tokens"]["premise_visible_char_end"]
                    for s in es
                )
                visible["fully_visible" if full else "truncated"][pred] += 1
    pred = [
        (r["claim_id"], r["doc_id"], r["chosen_label"])
        for r in rows
        if r["decision"] == "accepted"
    ]
    nei = {c["id"] for c in claims["val"] if not c["evidence"]}
    neifp = [x for x in pred if x[0] in nei]
    model = SafeEncoder("nli", DEVICE)
    checks = []
    for r in rows[:20]:
        encoded = model.tokenizer(
            r["premise"], r["hypothesis"], max_length=512, truncation="only_first"
        )
        checks.append(encoded["input_ids"] == r["tokens"]["input_ids"])
        assert r["hypothesis"] == next(
            c["claim"] for c in claims["val"] if c["id"] == r["claim_id"]
        )
        assert r["premise"] == document_text(bydoc[r["doc_id"]])
    toy = [
        ("A person is eating food.", "A person eats."),
        ("No person is eating.", "A person is eating."),
        ("A person is eating food.", "The weather is sunny."),
    ]
    logits = model.encode(toy, 8)
    model.close()
    seen = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[l][l] for l in confusion)
    result = {
        "input_order": "premise=title+abstract, hypothesis=claim; verified against corpus and claims",
        "label_mapping": {0: "CONTRADICT", 1: "SUPPORT", 2: "NEUTRAL"},
        "input_id_reproduction_checks": sum(checks),
        "checked_pairs": len(checks),
        "truncation": "only_first; hypothesis preserved",
        "softmax_checks": bool(
            all(
                abs(sum(r["probabilities"]) - 1) < 1e-5
                and np.allclose(
                    softmax(np.array(r["logits"])), r["probabilities"], atol=1e-6
                )
                for r in rows
            )
        ),
        "neutral_gating": "argmax neutral rejects regardless of threshold",
        "threshold": 0.5,
        "cap": 3,
        "coverage_and_oracle_bounds": cover,
        "conditional_gold_label_confusion": confusion,
        "conditional_gold_label_accuracy": correct / seen,
        "gold_candidate_count": seen,
        "gold_pair_decisions": dict(rejects),
        "gold_visibility_confusion": {k: dict(v) for k, v in visible.items()},
        "nei_false_positive_claims": len({cid for cid, d, l in neifp}),
        "nei_claims": len(nei),
        "nei_false_positive_triples": len(neifp),
        "toy_logits": logits.tolist(),
        "toy_argmax_labels": [LABELS[int(i)] for i in np.argmax(logits, axis=1)],
        "baseline_metrics": triple_metrics(pred, claims["val"]),
        "finding": "Input/mapping/gating implementation is consistent; many fully visible gold pairs are rejected as neutral. Test shorter inputs and supervised domain adaptation.",
    }
    save_json(EXP / "verification_audit.json", result)
    print("Audit", confusion, "conditional accuracy", correct / seen, flush=True)


def infer_variant(name, mode, checkpoint=None, system="hybrid", k=10, split="val"):
    seed_all()
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    rankings = (
        json.loads((BASE / f"{system}_{split}_rankings.json").read_text())
        if system == "hybrid"
        else json.loads((EXP / f"{system}_{split}_rankings.json").read_text())
    )
    public = inference_claims(claims[split])
    model = SafeEncoder("nli", DEVICE, checkpoint=checkpoint)
    pairs = []
    references = []
    for c in public:
        for r, d in enumerate(rankings[str(c["id"])][:k], 1):
            windows = (
                [
                    {
                        "start_sentence": None,
                        "end_sentence": None,
                        "text": document_text(bydoc[d]),
                    }
                ]
                if mode == "full"
                else selected_windows(bydoc[d], c["claim"])
                if mode == "lexical_windows"
                else window_candidates(bydoc[d])
            )
            for w in windows:
                pairs.append((w["text"], c["claim"]))
                references.append(
                    {"claim_id": c["id"], "doc_id": d, "retrieval_rank": r, "window": w}
                )
    identity = {
        "name": name,
        "mode": mode,
        "split": split,
        "system": system,
        "k": k,
        "checkpoint_sha256": file_hash(Path(checkpoint) / "model.safetensors")
        if checkpoint
        else model.revision,
        "input_sha256": digest(pairs),
        "max_length": 512,
        "truncation": "only_first",
    }
    path = EXP / f"{name}_{system}_{split}_windows.jsonl"
    start = time.perf_counter()
    fresh = not path.exists()
    if path.exists():
        window_rows = jsonl(path)
        assert (
            json.loads((EXP / f"{name}_{system}_{split}_config.json").read_text())
            == identity
        )
    else:
        cache_path = EXP / f"{name}_{system}_{split}_incremental_logits.json"
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        cache_identity = EXP / f"{name}_{system}_{split}_incremental_config.json"
        if cache_identity.exists():
            assert json.loads(cache_identity.read_text()) == identity
        save_json(cache_identity, identity)
        missing = [i for i in range(len(pairs)) if str(i) not in cached]
        for offset in range(0, len(missing), 32):
            chunk = missing[offset : offset + 32]
            values = model.encode([pairs[i] for i in chunk])
            cached.update({str(i): v.tolist() for i, v in zip(chunk, values)})
            save_json(cache_path, cached)
            print(
                name,
                system,
                split,
                "new windows",
                min(offset + 32, len(missing)),
                "/",
                len(missing),
                flush=True,
            )
        logits = np.array([cached[str(i)] for i in range(len(pairs))])
        probabilities = softmax(logits)
        window_rows = []
        for ref, pair, logit, prob in zip(references, pairs, logits, probabilities):
            encoded = model.tokenizer(
                pair[0],
                pair[1],
                truncation="only_first",
                max_length=512,
                return_offsets_mapping=True,
            )
            end = max(
                (
                    b
                    for (a, b), seq in zip(
                        encoded["offset_mapping"], encoded.sequence_ids()
                    )
                    if seq == 0
                ),
                default=0,
            )
            window_rows.append(
                {
                    **ref,
                    "premise": pair[0],
                    "hypothesis": pair[1],
                    "logits": logit.tolist(),
                    "probabilities": prob.tolist(),
                    "probability_order": LABELS,
                    "tokens": {
                        "input_ids": encoded["input_ids"],
                        "premise_visible_char_end": end,
                        "truncation": "only_first",
                        "tokenizer_revision": model.revision,
                        "untruncated_tokens": len(
                            model.tokenizer(pair[0], pair[1], truncation=False)[
                                "input_ids"
                            ]
                        ),
                        "truncated": len(
                            model.tokenizer(pair[0], pair[1], truncation=False)[
                                "input_ids"
                            ]
                        )
                        > 512,
                    },
                }
            )
        save_jsonl(path, window_rows)
        save_json(EXP / f"{name}_{system}_{split}_config.json", identity)
    grouped = collections.defaultdict(list)
    for row in window_rows:
        grouped[(row["claim_id"], row["doc_id"])].append(row)
    aggregated = []
    for key, items in grouped.items():
        # Select by model non-neutral confidence, not evidence labels/sentence indices.
        best = max(
            items,
            key=lambda r: (
                max(r["probabilities"][:2]),
                -r["window"]["start_sentence"]
                if r["window"]["start_sentence"] is not None
                else 0,
            ),
        )
        aggregated.append(
            {
                **best,
                "windows_examined": len(items),
                "aggregation": "max non-neutral confidence; neutral argmax still rejects",
            }
        )
    save_jsonl(EXP / f"{name}_{system}_{split}_pairs.jsonl", aggregated)
    save_json(
        EXP / f"{name}_{system}_{split}_measurement.json",
        {
            "elapsed_sec": time.perf_counter() - start,
            "window_pairs": len(window_rows),
            "document_pairs": len(aggregated),
            "model": model.metadata,
            "mode": mode,
            "fresh_inference": fresh,
        },
    )
    model.close()
    if split == "val" and not system.startswith("new_pairs_"):
        return threshold_sweep(
            aggregated, f"{name}_{system}", [x for x in [1, 3, 5, 10, 20] if x <= k]
        )
    return aggregated


def fit_nli(mode):
    seed_all()
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    ids = [d["doc_id"] for d in docs]
    rank = json.loads((BASE / "hybrid_train_rankings.json").read_text())
    rng = random.Random(42)
    examples = []
    for c in claims["train"]:
        gold = {d for d, l in gold_pairs(c)}

        def add(d, label, origin, assumed=False):
            if mode == "lexical_windows" and not assumed:
                evidence = sorted(
                    {
                        index
                        for e in c["evidence"]
                        if e["doc_id"] == d
                        for index in e["sentences"]
                    }
                )
                first = max(0, evidence[0] - 1)
                last = min(len(bydoc[d]["abstract"]), evidence[-1] + 2)
                windows = [
                    {
                        "text": bydoc[d]["title"]
                        + " "
                        + " ".join(bydoc[d]["abstract"][first:last]),
                        "start_sentence": first,
                        "end_sentence": last,
                        "selection": "training annotated evidence plus one neighbor",
                    }
                ]
            else:
                windows = (
                    selected_windows(bydoc[d], c["claim"], 1)
                    if mode == "lexical_windows"
                    else [
                        {
                            "text": document_text(bydoc[d]),
                            "start_sentence": None,
                            "end_sentence": None,
                        }
                    ]
                )
            examples.append(
                {
                    "claim_id": c["id"],
                    "doc_id": d,
                    "label": label,
                    "label_id": LABELS.index(label),
                    "premise": windows[0]["text"],
                    "hypothesis": c["claim"],
                    "origin": origin,
                    "assumed_neutral": assumed,
                    "window": windows[0],
                }
            )

        for d, label in sorted(gold_pairs(c)):
            add(d, label, "annotated claim-document pair")
        pool = [d for d in ids if d not in gold]
        hard = [d for d in rank[str(c["id"])][:20] if d not in gold]
        if gold:
            add(rng.choice(pool), "NEUTRAL", "uniform nongold assumption", True)
        add(rng.choice(hard or pool), "NEUTRAL", "mined nongold assumption", True)
    # Deduplicate paired inputs; retain label provenance. No validation labels fit weights.
    examples = list(
        {(x["claim_id"], x["doc_id"], x["label_id"]): x for x in examples}.values()
    )
    name = "finetuned_" + mode
    save_jsonl(EXP / f"{name}_training_examples.jsonl", examples)
    model = Encoder("nli", DEVICE)
    model.model.train()
    groups = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if not (n.endswith("bias") or "LayerNorm" in n)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if n.endswith("bias") or "LayerNorm" in n
            ],
            "weight_decay": 0,
        },
    ]
    opt = torch.optim.AdamW(groups, lr=2e-5)
    batch = 8
    steps = math.ceil(len(examples) / batch)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(steps * 0.1), steps)
    rng.shuffle(examples)
    losses = []
    start = time.perf_counter()
    from training import state_hash

    initial = state_hash(model.model)
    import gc

    micro_batch = 2
    for i in range(0, len(examples), batch):
        rows = examples[i : i + batch]
        opt.zero_grad(set_to_none=True)
        total_loss = 0
        for j in range(0, len(rows), micro_batch):
            micro = rows[j : j + micro_batch]
            features = model.tokenizer(
                [x["premise"] for x in micro],
                [x["hypothesis"] for x in micro],
                padding="max_length",
                max_length=512,
                truncation="only_first",
                return_tensors="pt",
            )
            logits = model.forward(features)
            loss = torch.nn.functional.cross_entropy(
                logits, torch.tensor([x["label_id"] for x in micro], device=DEVICE)
            )
            assert torch.isfinite(loss)
            (loss * len(micro) / len(rows)).backward()
            total_loss += float(loss.item()) * len(micro) / len(rows)
            del loss, logits, features
        norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1)
        assert torch.isfinite(norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        sync(DEVICE)
        losses.append(
            {
                "step": len(losses) + 1,
                "loss": total_loss,
                "batch_size": len(rows),
                "micro_batch": micro_batch,
            }
        )
        del norm, rows, micro
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        if len(losses) % 10 == 0:
            print(
                name,
                len(losses),
                "/",
                steps,
                "loss",
                total_loss,
                "sec",
                time.perf_counter() - start,
                "device",
                DEVICE,
                flush=True,
            )

    checkpoint = EXP / "checkpoints" / name
    model.model.save_pretrained(checkpoint)
    model.tokenizer.save_pretrained(checkpoint)
    final = state_hash(model.model)
    assert initial != final
    save_json(
        EXP / f"{name}_training.json",
        {
            "mode": mode,
            "model": model.metadata,
            "checkpoint": str(checkpoint),
            "epochs": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "batch_size": 8,
            "micro_batch_size": 2,
            "gradient_accumulation": 4,
            "padding": "fixed 512 during training; dynamic at inference",
            "allocator_cleanup": "after each effective batch",
            "warmup_steps": math.ceil(steps * 0.1),
            "steps": steps,
            "seed": 42,
            "temperature": "ordinary three-class NLI softmax",
            "examples": len(examples),
            "label_counts": dict(collections.Counter(x["label"] for x in examples)),
            "neutral_labels_are_assumed": True,
            "exclude_all_known_gold_documents": True,
            "training_gold_sentence_indices_used": mode == "lexical_windows",
            "validation_eval_gold_sentence_indices_used": False,
            "training_time_sec": time.perf_counter() - start,
            "initial_state_sha256": initial,
            "final_state_sha256": final,
        },
    )
    save_jsonl(EXP / f"{name}_loss.jsonl", losses)
    model.close()
    infer_variant(name, mode, checkpoint, "hybrid", 10)


def hybrid_sweep():
    docs, claims = load_data()
    ids = np.array([d["doc_id"] for d in docs])
    look = {d: i for i, d in enumerate(ids)}
    val = claims["val"]
    systems = ["bm25", "scientific", "general"]
    ranks = {
        n: json.loads((BASE / f"{n}_val_rankings.json").read_text()) for n in systems
    }
    positions = {}
    for n in systems:
        matrix = np.empty((len(val), len(ids)), np.int32)
        for i, c in enumerate(val):
            for r, d in enumerate(ranks[n][str(c["id"])], 1):
                matrix[i, look[d]] = r
        positions[n] = matrix
    results = []
    best = None
    score_vectors = []
    configs = []
    for constant in [10, 30, 60, 100]:
        for weight in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]:
            configs.append(
                {
                    "constant": constant,
                    "weights": [weight, 1, 0],
                    "family": "two_retriever",
                }
            )
        configs.append(
            {"constant": constant, "weights": [1, 0, 0], "family": "two_retriever"}
        )
        for a in range(5):
            for b in range(5 - a):
                configs.append(
                    {
                        "constant": constant,
                        "weights": [a / 4, b / 4, (4 - a - b) / 4],
                        "family": "three_retriever",
                    }
                )
    for idx, cfg in enumerate(configs):
        scores = sum(
            w / (cfg["constant"] + positions[n])
            for w, n in zip(cfg["weights"], systems)
        )
        ranked = {
            str(c["id"]): [int(ids[j]) for j in order_scores(scores[i], ids)]
            for i, c in enumerate(val)
        }
        metrics = retrieval_metrics(ranked, val)
        entry = {"index": idx, "config": cfg, "metrics": metrics}
        results.append(entry)
        score_vectors.append(
            [p["ndcg10"] for p in metrics["per_claim"] if not p["excluded"]]
        )
        if best is None or metrics["ndcg10"] > best["metrics"]["ndcg10"] + 1e-8:
            best = entry
    save_json(EXP / "hybrid_sweep.json", results)
    save_json(EXP / "hybrid_selection.json", best)
    cfg = best["config"]
    for split in ["train", "val"]:
        s = {
            n: json.loads((BASE / f"{n}_{split}_rankings.json").read_text())
            for n in systems
        }
        out = {}
        for c in claims[split]:
            score = {}
            for n, w in zip(systems, cfg["weights"]):
                for r, d in enumerate(s[n][str(c["id"])], 1):
                    score[d] = score.get(d, 0) + w / (cfg["constant"] + r)
            out[str(c["id"])] = sorted(score, key=lambda d: (-score[d], d))
        save_json(EXP / f"improved_hybrid_{split}_rankings.json", out)
    baseline = np.array(
        [
            p["ndcg10"]
            for p in json.loads((BASE / "hybrid_metadata.json").read_text())["metrics"][
                "per_claim"
            ]
            if not p["excluded"]
        ]
    )
    matrix = np.array(score_vectors)
    rng = np.random.default_rng(42)
    bootstrap = rng.integers(0, len(baseline), (1000, len(baseline)))
    gains = (matrix[best["index"]] - baseline)[bootstrap].mean(1)
    means = matrix[:, bootstrap].mean(2)
    wins = np.argmax(means, axis=0)
    save_json(
        EXP / "hybrid_bootstrap.json",
        {
            "replicates": 1000,
            "seed": 42,
            "paired_gain": best["metrics"]["ndcg10"] - baseline.mean(),
            "paired_gain_95_percentile_ci": np.quantile(gains, [0.025, 0.975]).tolist(),
            "selection_stability": dict(collections.Counter(map(int, wins))),
            "limitation": "Finite validation sweep; bootstrap is descriptive and does not correct selection optimism",
            "top_configurations": sorted(
                results, key=lambda x: (-x["metrics"]["ndcg10"], x["index"])
            )[:10],
        },
    )
    print(
        "Hybrid sweep", len(results), "best", cfg, best["metrics"]["ndcg10"], flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["audit", "windows", "ft_full", "ft_windows", "hybrid"]
    )
    a = p.parse_args()
    if a.stage == "audit":
        audit()
    elif a.stage == "windows":
        infer_variant("zeroshot_all_windows", "all_windows", None, "hybrid", 10)
        infer_variant("zeroshot_lexical_windows", "lexical_windows", None, "hybrid", 10)
    elif a.stage == "ft_full":
        fit_nli("full")
    elif a.stage == "ft_windows":
        fit_nli("lexical_windows")
    else:
        hybrid_sweep()
