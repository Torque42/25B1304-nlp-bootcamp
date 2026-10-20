"""Build computed results/report, enforce submission gates, and package."""

import argparse, collections, csv, datetime, json, re, shutil, subprocess, sys, zipfile
from common import *

RUNS = ["bm25", "general", "scientific", "hybrid", "rerank", "random", "hard"]
SUBMISSION = ROOT  # The distributed ZIP has code/, predictions/, report.md at its root.
TEMPLATE = ROOT / "results_template.json" if (ROOT / "results_template.json").is_file() else ROOT / "code/results_template.json"
CHECKER = ROOT / "check_format.py" if (ROOT / "check_format.py").is_file() else ROOT / "code/check_format.py"


def metadata(name):
    return json.loads((ART / f"{name}_metadata.json").read_text())


def assemble():
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    cfg = json.loads((ROOT / "code/config.json").read_text())
    info = {n: metadata(n) for n in RUNS}
    nli = json.loads((ART / "nli_selection.json").read_text())
    result = json.loads(TEMPLATE.read_text())
    result.update(
        **cfg["identity"],
        seed=42,
        gpu_used=f"Task2={info['general']['query_encoder']['device']}; reranker={info['rerank']['config']['model']['device']}; NLI={nli['model']['device']}; float32",
    )
    bm = info["bm25"]
    result["task1"].update(
        bm25_settings=json.dumps(bm["config"]), val_ndcg10=bm["metrics"]["ndcg10"]
    )
    result["task2"].update(
        general_model=info["general"]["query_encoder"]["source"],
        scientific_model="ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder",
        val_ndcg10_general=info["general"]["metrics"]["ndcg10"],
        val_ndcg10_scientific=info["scientific"]["metrics"]["ndcg10"],
        index_time_sec={
            n: info[n]["index_time_sec"] for n in ["general", "scientific"]
        },
        query_latency_ms={
            n: info[n]["query_latency_ms"] for n in ["general", "scientific"]
        },
        memory_mb={n: info[n]["memory_mb"] for n in ["general", "scientific"]},
    )
    rr = info["rerank"]
    result["task3"].update(
        hybrid_method=json.dumps(info["hybrid"]["config"]),
        reranker_model=rr["config"]["model"]["source"],
        rerank_top_k=rr["config"]["k"],
        val_ndcg10_hybrid=info["hybrid"]["metrics"]["ndcg10"],
        val_ndcg10_rerank=rr["metrics"]["ndcg10"],
        rerank_query_latency_ms=rr["rerank_query_latency_ms"],
    )
    result["task4"].update(
        base_model=info["general"]["query_encoder"]["source"],
        val_ndcg10_random_negatives=info["random"]["metrics"]["ndcg10"],
        val_ndcg10_hard_negatives=info["hard"]["metrics"]["ndcg10"],
    )
    result["task5"].update(
        retriever_used=nli["retriever"],
        nli_model=nli["model"]["source"],
        zero_shot_or_finetuned="zero-shot"
        if nli.get("zero_shot", True)
        else "fine-tuned on training claims",
        val_f1=nli["metrics"]["f1"],
    )
    per = {
        n: {
            p["claim_id"]: p
            for p in info[n]["metrics"]["per_claim"]
            if not p["excluded"]
        }
        for n in ["bm25", "general"]
    }
    differences = {
        cid: per["bm25"][cid]["ndcg10"] - per["general"][cid]["ndcg10"]
        for cid in per["bm25"]
    }
    bm_wins = sorted(
        [cid for cid, d in differences.items() if d > 1e-8],
        key=lambda cid: (-differences[cid], cid),
    )[:2]
    gen_wins = sorted(
        [cid for cid, d in differences.items() if d < -1e-8],
        key=lambda cid: (differences[cid], cid),
    )[:2]
    result["report_q1"].update(bm25_wins=bm_wins, general_wins=gen_wins)
    examples = []
    for winner, cids in [("bm25", bm_wins), ("general", gen_wins)]:
        for cid in cids:
            c = next(c for c in claims["val"] if c["id"] == cid)
            example = {
                "claim_id": cid,
                "claim": c["claim"],
                "winner": winner,
                "scores": {n: per[n][cid]["ndcg10"] for n in per},
                "gold": c["evidence"],
                "ranked_documents": {},
            }
            for n in per:
                rank = read_run(n, "val")[str(cid)]
                example["ranked_documents"][n] = [
                    {
                        "rank": r,
                        "doc_id": d,
                        "title": bydoc[d]["title"],
                        "abstract": bydoc[d]["abstract"],
                        "gold": d in {e["doc_id"] for e in c["evidence"]},
                    }
                    for r, d in enumerate(rank[:10], 1)
                ]
            examples.append(example)
    save_json(ART / "q1_examples.json", examples)
    result["diagnostics"] = {
        "retrieval_metric_denominator": 106,
        "retrieval_excluded_nei": 56,
        "mrr_cutoff": 100,
        "verification_claims": 162,
        "verification": nli["metrics"],
        "nli_candidate_k": nli["candidate_k"],
        "nli_threshold": nli["threshold"],
        "nli_decision_config": nli.get("decision_config"),
        "nli_input_mode": nli.get("input_mode", "full"),
        "nli_window_aggregation": nli.get("aggregation", "none"),
        "memory_convention": "peak process RSS MiB; MPS memory recorded separately",
        "index_time_convention": "fresh tokenization, corpus encoding and searchable matrix; excludes model download/load",
        "query_latency_convention": "30 sorted validation queries, warmed fresh individual tokenization/encoding/exact scoring/top100, synchronized MPS",
        "retrieval_table": {
            n: {k: v for k, v in info[n]["metrics"].items() if k != "per_claim"}
            for n in RUNS
        },
    }
    review_path = ART / "human_review.json"
    reviewed = json.loads(review_path.read_text()) if review_path.exists() else None
    result["completion"] = {
        "human_review_complete": bool(reviewed),
        "identity_complete": bool(result["name"] and result["roll_no"]),
        "pending_gates": []
        if reviewed
        else ["human validation error review and Q4 explanation"],
    }
    save_json(SUBMISSION / "results.json", result)
    # Q1 explanations are derived from measured ranks; retain text for the reviewer.
    q1 = "BM25 wins on 1156 (DESMOND: gold rank 1 versus dense 40) and 673 (LDL causality: 1 versus 5). The rare DESMOND name and the nearly verbatim LDL statement favor exact term matching. Dense wins on 1311 (lncRNA translation: gold rank 1 versus BM25 24) and 55 (AMPK and lung fibrosis: 1 versus 8). Their evidence uses lincRNA-p21 and metformin/bleomycin terminology, while dense retrieval connects the broader claim to those specific mechanisms. These explanations are interpretations of the saved claim/evidence texts and ranked results."
    dense = info["hybrid"]["config"]["dense"]
    base = info[dense]["metrics"]["ndcg10"]
    hy = info["hybrid"]["metrics"]["ndcg10"]
    first = rr["config"]["first_stage"]
    rrbase = info[first]["metrics"]["ndcg10"]
    rer = rr["metrics"]["ndcg10"]
    q2 = f"{'Hybrid' if hy - base >= rer - rrbase else 'Reranking'} gave the largest Task 3 validation gain. Hybrid changes nDCG@10 from {base:.4f} ({dense}) to {hy:.4f}, a gain of {hy - base:+.4f}. Reranking changes its selected {first} first stage from {rrbase:.4f} to {rer:.4f}, a gain of {rer - rrbase:+.4f}, at top-{rr['config']['k']}. Hybrid's estimated total mean latency is {info['hybrid']['query_latency_ms']:.1f} ms; reranking adds {rr['rerank_query_latency_ms']:.1f} ms, for {rr['query_latency_ms']:.1f} ms total. Reranking overhead is freshly measured on 30 warmed queries; totals sum constituent means. The finite validation sweep does not establish a global optimum."
    train = {
        n: json.loads((ART / f"{n}_training.json").read_text())
        for n in ["random", "hard"]
    }
    ran = info["random"]["metrics"]["ndcg10"]
    hard = info["hard"]["metrics"]["ndcg10"]
    orig = info["general"]["metrics"]["ndcg10"]
    q3 = f"Hard negatives {'helped' if hard > ran else 'did not improve over random negatives'}: validation nDCG@10 is {hard:.4f}, versus {ran:.4f} for random negatives and {orig:.4f} before tuning. Both runs start from identical weights and use the same {train['random']['config']['epochs']}-epoch schedule; only sampled negatives differ. Mean epoch losses are {train['random']['epoch_losses'][-1]:.4f} (random) and {train['hard']['epoch_losses'][-1]:.4f} (hard). With one epoch, there is insufficient learning-curve evidence to establish overfitting. A validation decrease alone is not proof."
    q4 = (
        f"{reviewed['retrieval_miss_count']} of {reviewed['error_count']} validation errors are confirmed retrieval_miss ({reviewed['retrieval_miss_fraction']:.2%}). Required evidence outside the actual NLI candidate subset counts as a retrieval miss; semantic categories were assigned by the human reviewer."
        if reviewed and reviewed["error_count"]
        else "Pending human review. Every unequal validation gold/predicted pair set has a review record with a blank final category. The retrieval-miss fraction will be computed from confirmed categories, using all validation verification errors as the denominator. Automatic flags are not final error labels."
    )
    review = jsonl(ART / "validation_error_review.jsonl")
    flags = collections.Counter(f for r in review for f in r["suggested_flags"])
    q5 = f"Try a supervised evidence-sentence selector trained on the training annotations, before NLI. Current verification F1 is {nli['metrics']['f1']:.4f}; {flags['gold_examined_but_wrong_or_rejected']} incorrect claims include gold evidence that reached NLI but was mislabeled or rejected. This directly targets observed verification failures. Preserve the retriever and candidate cutoff during this comparison so its effect can be measured separately."
    answers = [q1, q2, q3, q4, q5]
    assert all(len(x.split()) <= 100 for x in answers)
    (SUBMISSION / "report.md").write_text(
        "\n\n".join(f"## Q{i}\n\n{x}" for i, x in enumerate(answers, 1)) + "\n"
    )
    code_dest = SUBMISSION / "code"
    code_dest.mkdir(exist_ok=True)
    for p in (ROOT / "code").glob("*"):
        if p.is_file() and p.resolve() != (code_dest / p.name).resolve():
            shutil.copy2(p, code_dest / p.name)
    if (ROOT / "tests").is_dir():
        shutil.copytree(
            ROOT / "tests",
            code_dest / "tests",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    records = code_dest / "run_records"
    records.mkdir(exist_ok=True)
    for p in ART.iterdir():
        if (
            p.is_file()
            and p.suffix in {".json", ".jsonl", ".csv", ".html"}
            and not p.name.endswith("_rankings.json")
        ):
            shutil.copy2(p, records / p.name)
    # Retain extension configurations, traces, failures and stability; exclude weights/full rankings.
    extension = ART / "experiments" / "window_20261013T124402Z"
    if extension.exists():
        for source in extension.rglob("*"):
            if (
                source.is_file()
                and source.suffix in {".json", ".jsonl", ".csv", ".html", ".log"}
                and "checkpoints" not in source.parts
                and not source.name.endswith("_rankings.json")
            ):
                destination = (
                    records
                    / "experiments"
                    / extension.name
                    / source.relative_to(extension)
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for plan in (ROOT / "docs").glob("*.md"):
            shutil.copy2(plan, records / plan.name)
    # Include author-owned assignment checker for convenient reproduction; never alter it.
    if CHECKER.resolve() != (code_dest / "check_format.py").resolve():
        shutil.copy2(CHECKER, code_dest / "check_format.py")
    save_json(ART / "completion_status.json", result["completion"])
    print(json.dumps(result["completion"]), flush=True)


def validate(require_complete=True):
    docs, claims = load_data()
    ids = {d["doc_id"] for d in docs}
    import sys

    sys.path.insert(0, str(ROOT))
    import check_format

    checked = check_format.check_submission(
        str(SUBMISSION), check_format.load_data_info(str(Path(os.environ.get("NLP_DATA_DIR") or ROOT / "data")))
    )
    assert all(
        status == "OK"
        for name, status in checked["status"].items()
        if name != "t5_val_errors.csv"
    ), checked["status"]
    if require_complete:
        assert all(s == "OK" for s in checked["status"].values()), checked["status"]
        assert not checked["general"], checked["general"]
    for name in list(check_format.EVAL_RUNS) + check_format.VAL_RUNS:
        path = PRED / name
        if not path.exists():
            assert not require_complete
            continue
        scores = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            assert parts[1] == "Q0" and len(parts) == 6
            cid, did, rank, score = (
                int(parts[0]),
                int(parts[2]),
                int(parts[3]),
                float(parts[4]),
            )
            assert np.isfinite(score)
            scores.setdefault(cid, []).append((rank, score))
        for rows in scores.values():
            rows.sort()
            assert all(a[1] >= b[1] for a, b in zip(rows, rows[1:]))
    errors_path = PRED / "t5_val_errors.csv"
    expected = set(
        json.loads((ART / "nli_selection.json").read_text())["metrics"]["error_ids"]
    )
    if errors_path.exists():
        actual = check_format.check_errors(
            str(errors_path), {c["id"] for c in claims["val"]}
        )
        assert set(actual) == expected
    elif require_complete:
        raise ValueError("Human-reviewed categories missing")
    actual_metrics = triple_metrics(
        checked["parsed"]["t5_val_verification.csv"], claims["val"]
    )
    selected_metrics = json.loads((ART / "nli_selection.json").read_text())["metrics"]
    assert actual_metrics == selected_metrics, (
        "Verification file disagrees with saved metrics"
    )
    review_ids = [r["claim_id"] for r in jsonl(ART / "validation_error_review.jsonl")]
    assert len(review_ids) == len(set(review_ids)) and set(review_ids) == expected
    result = json.loads((SUBMISSION / "results.json").read_text())
    assert result["task5"]["val_f1"] == actual_metrics["f1"]
    if require_complete:
        assert (
            result["completion"]["human_review_complete"]
            and result["name"]
            and result["roll_no"]
        )
    for name in ["random", "hard"]:
        trained = json.loads((ART / f"{name}_training.json").read_text())
        assert trained["weights_changed"]
        assert trained["initial_state_sha256"] != trained["final_state_sha256"]
    a, b = [
        json.loads((ART / f"{n}_training.json").read_text()) for n in ["random", "hard"]
    ]
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    manifest = json.loads((ART / "input_manifest.json").read_text())
    for path, sha in manifest["protected_file_hashes"].items():
        if path != "specs.md":  # Historical specs can be superseded after approval.
            assert file_hash(ROOT / path) == sha, path
    report = (SUBMISSION / "report.md").read_text()
    answers = re.split(r"## Q[1-5]\s+", report)[1:]
    assert len(answers) == 5 and all(len(x.split()) <= 100 for x in answers)

    def populated(x):
        if isinstance(x, dict):
            return all(populated(v) for v in x.values())
        if isinstance(x, list):
            return all(populated(v) for v in x)
        return x is not None and x != ""

    template = json.loads(TEMPLATE.read_text())
    assert all(populated(result[k]) for k in template), (
        "Template result fields not populated"
    )
    save_json(
        ART / "validation_checks.json",
        {
            "assignment_checker": checked["status"],
            "other_issues": checked["general"],
            "human_review_required": not errors_path.exists(),
            "supplemental_checks": "passed: finite sorted scores, Q0, category coverage when reviewed, training hashes, immutable inputs, report word counts, result fields",
        },
    )
    print(
        "Supplemental validation passed; complete=" + str(require_complete), flush=True
    )


def package(draft=False):
    assemble()
    validate(not draft)
    roll = json.loads((SUBMISSION / "results.json").read_text())["roll_no"]
    name = f"{roll}_nlp_bootcamp" + ("_DRAFT_PENDING_REVIEW" if draft else "") + ".zip"
    path = ROOT / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("code", "predictions", "results.json", "report.md"):
            source = SUBMISSION / name
            for f in sorted(source.rglob("*")) if source.is_dir() else [source]:
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, f.relative_to(SUBMISSION))
    with zipfile.ZipFile(path) as z:
        assert not any("model.safetensors" in p or ".venv" in p for p in z.namelist())
        assert z.testzip() is None
    # validate() already invokes the supplied checker with the correct data/
    # path; its standalone CLI assumes check_format.py and data/ are siblings.
    print("Packaged " + str(path), flush=True)


if __name__ == "__main__":
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["assemble", "validate", "package"])
    p.add_argument("--draft", action="store_true")
    a = p.parse_args()
    if a.action == "assemble":
        assemble()
    elif a.action == "validate":
        validate(not a.draft)
    else:
        package(a.draft)
