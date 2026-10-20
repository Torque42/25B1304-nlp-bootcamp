"""Sequential full/resume runner. Each neural stage uses a fresh process."""

import argparse, json, subprocess, sys
from pathlib import Path

from common import initialize_manifest

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "benchmark",
    "bm25",
    "general",
    "scientific",
    "hybrid",
    "rerank20",
    "rerank50",
    "rerank100",
    "train_random",
    "random",
    "train_hard",
    "hard",
    "nli5",
    "assemble",
]


def command(stage, device):
    py = sys.executable
    if stage == "benchmark":
        return [py, "code/benchmark.py", "--device", device]
    if stage in {"bm25", "general", "scientific", "hybrid"}:
        return [py, "code/retrieval.py", stage, "--device", device]
    if stage.startswith("rerank"):
        return [py, "code/reranking.py", "--device", device, "--k", stage[6:]] + (
            ["--finalize"] if stage == "rerank100" else []
        )
    if stage.startswith("train_"):
        return [py, "code/training.py", stage[6:], "--device", device]
    if stage in {"random", "hard"}:
        return [
            py,
            "code/retrieval.py",
            stage,
            "--device",
            device,
            "--checkpoint",
            f"artifacts/checkpoints/{stage}/epoch-1",
        ]
    if stage == "nli5":
        return [py, "code/verification.py", "--device", device, "--k", "5"]
    if stage == "assemble":
        return [py, "code/submission.py", "assemble"]
    raise ValueError(stage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--from-stage", default="benchmark", choices=STAGES)
    p.add_argument("--through-stage", default="assemble", choices=STAGES)
    a = p.parse_args()
    initialize_manifest()
    for stage in STAGES[STAGES.index(a.from_stage) : STAGES.index(a.through_stage) + 1]:
        cmd = command(stage, a.device)
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
