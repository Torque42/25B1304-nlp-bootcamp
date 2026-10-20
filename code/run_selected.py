"""Reproduce or resume the selected pipeline without planning-chat paths."""

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from common import ROOT, ART, save_json, load_data, retrieval_metrics
from extended_experiments import EXP, BASE

STAGES = [
    "baseline",
    "fusion",
    "nli_training",
    "reranking",
    "verification_selection",
    "materialize",
    "package",
]


def run(args, device):
    env = {**os.environ, "NLP_DEVICE": device}
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, check=True)


def baseline(device):
    required = [
        f"{name}_metadata.json"
        for name in [
            "bm25",
            "general",
            "scientific",
            "hybrid",
            "rerank",
            "random",
            "hard",
        ]
    ] + [
        "hybrid_train_rankings.json",
        "hybrid_val_rankings.json",
        "hybrid_eval_rankings.json",
        "nli_val_trace.jsonl",
        "nli_selection.json",
    ]
    if (
        BASE.exists()
        and all((BASE / name).is_file() for name in required)
        and (BASE.parent / "snapshot_manifest.json").is_file()
    ):
        print("Using verified baseline snapshot:", BASE, flush=True)
        return
    run(
        [
            "code/run.py",
            "--device",
            device,
            "--from-stage",
            "bm25",
            "--through-stage",
            "nli5",
        ],
        device,
    )
    snapshot = BASE.parent
    snapshot.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(exist_ok=True)
    for p in ART.iterdir():
        if p.is_file():
            shutil.copy2(p, BASE / p.name)
    # Extracted packages have a flat submission root. Snapshot only the
    # deliverables, avoiding recursive copying of artifacts into themselves.
    snapshot_submission = snapshot / "submission"
    shutil.copytree(ROOT / "predictions", snapshot_submission / "predictions", dirs_exist_ok=True)
    for name in ("results.json", "report.md"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, snapshot_submission / name)
    save_json(
        snapshot / "snapshot_manifest.json",
        {
            "purpose": "baseline created from original seven systems before selected-pipeline promotion",
            "seed": 42,
            "device": device,
            "complete": True,
            "required_files": required,
        },
    )


def prepare_fusion(recipe):
    from score_fusion import generate, component_scores, normalize, COMPONENTS
    import numpy as np

    config = recipe["hybrid"]
    ranks = generate("val", config)
    _, claims = load_data()
    metrics = retrieval_metrics(ranks, claims["val"])
    generate("train", config)
    docs, _, scores = component_scores("val")
    times = []
    ids = np.array([d["doc_id"] for d in docs])
    for i in range(30):
        start = time.perf_counter()
        combined = sum(
            w * normalize(scores[n][i : i + 1], config["normalization"])
            for w, n in zip(config["weights"], COMPONENTS)
        )
        np.lexsort((ids, -combined[0]))
        times.append((time.perf_counter() - start) * 1000)
    overhead = float(np.mean(times))
    total = (
        sum(
            json.loads((ART / f"{n}_metadata.json").read_text())["query_latency_ms"]
            for n in COMPONENTS
        )
        + overhead
    )
    save_json(
        EXP / "hybrid_selection.json",
        {
            "config": config,
            "metrics": metrics,
            "fusion_overhead_ms": overhead,
            "query_latency_ms": total,
            "source": "fixed configuration from recorded validation experiments",
            "latency_convention": "sum of fresh component means and fresh query-local normalization/sort overhead",
        },
    )
    save_json(EXP / "medcpt_cross_encoder_reference.json", recipe["reranker"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--from-stage", default="baseline", choices=STAGES)
    p.add_argument("--through-stage", default="package", choices=STAGES)
    a = p.parse_args()
    recipe = json.loads((ROOT / "code/selected_recipe.json").read_text())
    for stage in STAGES[STAGES.index(a.from_stage) : STAGES.index(a.through_stage) + 1]:
        if stage == "baseline":
            baseline(a.device)
        elif stage == "fusion":
            prepare_fusion(recipe)
        elif stage == "nli_training":
            for name in ["full", "windows"]:
                checkpoint = (
                    EXP
                    / "checkpoints"
                    / (
                        "finetuned_full"
                        if name == "full"
                        else "finetuned_lexical_windows"
                    )
                    / "model.safetensors"
                )
                if not checkpoint.exists():
                    run(["code/extended_experiments.py", "ft_" + name], a.device)
                else:
                    # Saved weights do not imply that post-training validation finished.
                    mode = "full" if name == "full" else "lexical_windows"
                    env = {**os.environ, "NLP_DEVICE": a.device}
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            f"import sys;sys.path.insert(0,'code');from extended_experiments import infer_variant,EXP;infer_variant('finetuned_{mode}','{mode}',EXP/'checkpoints'/'finetuned_{mode}','hybrid',10)",
                        ],
                        cwd=ROOT,
                        env=env,
                        check=True,
                    )
        elif stage == "reranking":
            run(
                [
                    "code/biomedical_rerank.py",
                    "--val-k",
                    "50",
                    "--finalize",
                    "--device",
                    a.device,
                    "--batch-size",
                    "2",
                ],
                a.device,
            )
        elif stage == "verification_selection":
            run(["code/promote_experiments.py", "validate_nli"], a.device)
        elif stage == "materialize":
            run(["code/promote_experiments.py", "materialize"], a.device)
        else:
            run(["code/submission.py", "package", "--draft"], a.device)


if __name__ == "__main__":
    main()
