"""Matched one-epoch two-candidate contrastive training experiments."""

import argparse, json, math, random, time
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from common import *
from neural import Encoder, sync


def state_hash(model):
    import hashlib

    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def sample_triplets(method, docs, claims, rankings=None):
    ids = [d["doc_id"] for d in docs]
    rng = random.Random(SEED)
    result = []
    for c in sorted(claims, key=lambda c: c["id"]):
        gold = sorted({e["doc_id"] for e in c["evidence"]})
        if not gold:
            continue
        pool = [d for d in ids if d not in gold]
        if method == "hard":
            pool = [d for d in rankings[str(c["id"])] if d not in gold][:20]
        fallback = not pool
        if fallback:
            pool = [d for d in ids if d not in gold]
        for positive in gold:
            negative = rng.choice(pool)
            assert negative not in gold
            result.append(
                {
                    "claim_id": c["id"],
                    "positive_doc_id": positive,
                    "negative_doc_id": negative,
                    "pool_size": len(pool),
                    "random_fallback": fallback,
                }
            )
    return result


def train(method, device, epochs):
    seed_all()
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    byclaim = {c["id"]: c for c in claims["train"]}
    rankings = None
    if method == "hard":
        systems = [read_run(n, "train") for n in ["bm25", "general", "scientific"]]
        # Equal weights over our Task 1 and both Task 2 top-100 lists.
        rankings = {}
        for c in claims["train"]:
            scores = {}
            for system in systems:
                for r, d in enumerate(system[str(c["id"])][:100], 1):
                    scores[d] = scores.get(d, 0) + 1 / (60 + r)
            rankings[str(c["id"])] = sorted(scores, key=lambda d: (-scores[d], d))
        save_json(ART / "hard_mining_rankings.json", rankings)
    triplets = sample_triplets(method, docs, claims["train"], rankings)
    assert len(triplets) == 440
    save_jsonl(ART / f"{method}_triplets.jsonl", triplets)
    model = Encoder("general", device)
    initial_hash = state_hash(model.model)
    base_metadata = model.metadata.copy()
    model.model.train()
    decay = []
    no_decay = []
    for name, param in model.model.named_parameters():
        (
            no_decay
            if name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name
            else decay
        ).append(param)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=2e-5,
    )
    batch_size = 16
    steps = math.ceil(len(triplets) / batch_size) * epochs
    warmup = math.ceil(0.1 * steps)
    scheduler = get_linear_schedule_with_warmup(opt, warmup, steps)
    rng = random.Random(SEED)
    losses = []
    start = time.perf_counter()
    checkpoint = ART / "checkpoints" / method
    for epoch in range(epochs):
        shuffled = list(triplets)
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            q = model.forward(
                model.tokenize([byclaim[t["claim_id"]]["claim"] for t in batch])
            )
            p = model.forward(
                model.tokenize(
                    [document_text(bydoc[t["positive_doc_id"]]) for t in batch]
                )
            )
            n = model.forward(
                model.tokenize(
                    [document_text(bydoc[t["negative_doc_id"]]) for t in batch]
                )
            )
            logits = torch.stack([(q * p).sum(1), (q * n).sum(1)], 1) / 0.05
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(batch), dtype=torch.long, device=device)
            )
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            assert torch.isfinite(norm)
            opt.step()
            scheduler.step()
            sync(device)
            losses.append(
                {
                    "epoch": epoch + 1,
                    "step": len(losses) + 1,
                    "batch_size": len(batch),
                    "loss": float(loss.item()),
                    "learning_rate_after_step": scheduler.get_last_lr()[0],
                    "gradient_norm_before_clipping": float(norm.item()),
                }
            )
            print(
                f"{method} epoch {epoch + 1} step {len(losses)}/{steps} loss {loss.item():.4f}",
                flush=True,
            )
        epoch_path = checkpoint / f"epoch-{epoch + 1}"
        model.model.save_pretrained(epoch_path, safe_serialization=True)
        model.tokenizer.save_pretrained(epoch_path)
    final_hash = state_hash(model.model)
    assert final_hash != initial_hash
    other = ART / f"{'hard' if method == 'random' else 'random'}_training.json"
    if other.exists():
        assert json.loads(other.read_text())["initial_state_sha256"] == initial_hash
    save_jsonl(ART / f"{method}_losses.jsonl", losses)
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "weight_decay_exclusions": ["bias", "LayerNorm"],
        "warmup_steps": warmup,
        "total_steps": steps,
        "schedule": "linear warmup and decay",
        "gradient_clip": 1.0,
        "temperature": 0.05,
        "loss": "two-candidate cross entropy, positive target zero, no in-batch negatives",
        "seed": 42,
        "positives": len(triplets),
        "excluded_nei": 248,
        "hard_mining": "equal weight RRF k=60 across BM25 and both pretrained dense top100 lists"
        if method == "hard"
        else None,
    }
    epoch_losses = [
        sum(x["loss"] * x["batch_size"] for x in losses if x["epoch"] == e)
        / sum(x["batch_size"] for x in losses if x["epoch"] == e)
        for e in range(1, epochs + 1)
    ]
    save_json(
        ART / f"{method}_training.json",
        {
            "config": config,
            "base_model": base_metadata,
            "initial_state_sha256": initial_hash,
            "final_state_sha256": final_hash,
            "weights_changed": True,
            "training_time_sec": time.perf_counter() - start,
            "epoch_losses": epoch_losses,
            "checkpoint": str(checkpoint / f"epoch-{epochs}"),
            "triplet_sha256": file_hash(ART / f"{method}_triplets.jsonl"),
        },
    )
    model.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["random", "hard"])
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=1)
    a = p.parse_args()
    train(a.method, a.device, a.epochs)
