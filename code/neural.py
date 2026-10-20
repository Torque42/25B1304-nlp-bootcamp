"""Eager float32 encoder inference and measurements on MPS or CPU."""

import gc, json, os, resource, time

os.environ.setdefault(
    "HF_HOME",
    str(__import__("pathlib").Path(__file__).resolve().parents[1] / "artifacts" / "hf"),
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from common import ART, digest, save_json

MODELS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    "query": "ncbi/MedCPT-Query-Encoder",
    "article": "ncbi/MedCPT-Article-Encoder",
    "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "nli": "cross-encoder/nli-deberta-v3-small",
}


def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def release():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class Encoder:
    def __init__(self, kind, device="mps", checkpoint=None, revision=None):
        self.kind = kind
        self.device = device
        self.source = str(checkpoint or MODELS[kind])
        start = time.perf_counter()
        if revision is None and checkpoint is None:
            revision = json.loads(
                (
                    __import__("pathlib").Path(__file__).parent / "config.json"
                ).read_text()
            )["model_revisions"][kind]
        self.tokenizer = AutoTokenizer.from_pretrained(self.source, revision=revision)
        cls = (
            AutoModelForSequenceClassification
            if kind in {"reranker", "nli"}
            else AutoModel
        )
        self.model = (
            cls.from_pretrained(
                self.source, revision=revision, attn_implementation="eager"
            )
            .float()
            .to(device)
            .eval()
        )
        self.load_time = time.perf_counter() - start
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.parameters = sum(p.numel() for p in self.model.parameters())
        assert self.parameters <= 400_000_000
        self.max_length = {
            "general": 256,
            "query": 64,
            "article": 512,
            "reranker": 512,
            "nli": 512,
        }[kind]
        self.metadata = {
            "kind": kind,
            "source": self.source,
            "revision": self.revision,
            "parameters": self.parameters,
            "device": device,
            "dtype": "float32",
            "max_length": self.max_length,
            "pooling": "mean_l2"
            if kind == "general"
            else "cls_raw"
            if kind in {"query", "article"}
            else "raw_logits",
            "load_time_sec": self.load_time,
        }
        if kind == "nli":
            mapping = {
                int(k): str(v).lower() for k, v in self.model.config.id2label.items()
            }
            assert mapping == {0: "contradiction", 1: "entailment", 2: "neutral"}, (
                mapping
            )
            self.metadata["label_mapping"] = mapping

    def tokenize(self, inputs):
        paired = isinstance(inputs[0], (list, tuple))
        kwargs = {
            "padding": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
            "truncation": "only_second"
            if self.kind == "reranker"
            else "only_first"
            if self.kind == "nli"
            else True,
        }
        if paired:
            return self.tokenizer(
                [x[0] for x in inputs], [x[1] for x in inputs], **kwargs
            )
        return self.tokenizer(inputs, **kwargs)

    def forward(self, features):
        out = self.model(**{k: v.to(self.device) for k, v in features.items()})
        if self.kind in {"reranker", "nli"}:
            return out.logits
        if self.kind == "general":
            mask = features["attention_mask"].to(self.device).unsqueeze(-1)
            embedding = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(embedding, p=2, dim=1)
        return out.last_hidden_state[:, 0, :]

    def encode(self, inputs, batch_size=16, progress=False):
        result = []
        start = time.perf_counter()
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + batch_size]
            try:
                with torch.inference_mode():
                    out = self.forward(self.tokenize(batch)).cpu().numpy()
            except RuntimeError as e:
                if "memory" not in str(e).lower() or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                release()
                print(f"Reducing inference batch to {batch_size}", flush=True)
                continue
            assert np.isfinite(out).all()
            result.append(out)
            i += len(batch)
            if progress and (i % 256 == 0 or i == len(inputs)):
                print(
                    f"{self.kind}: {i}/{len(inputs)} in {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
        return np.concatenate(result)

    def token_metadata(self, inputs):
        rows = []
        for x in inputs:
            first, second = x if isinstance(x, (tuple, list)) else (x, None)
            a = self.tokenizer(first, add_special_tokens=False)["input_ids"]
            b = (
                self.tokenizer(second, add_special_tokens=False)["input_ids"]
                if second is not None
                else []
            )
            total = (
                len(a)
                + len(b)
                + self.tokenizer.num_special_tokens_to_add(pair=second is not None)
            )
            rows.append(
                {
                    "untruncated_tokens": total,
                    "truncated": total > self.max_length,
                    "first_tokens": len(a),
                    "second_tokens": len(b),
                    "max_length": self.max_length,
                }
            )
        return rows

    def close(self):
        del self.model
        release()


def cache_embeddings(encoder, inputs, key_config, batch_size=16, force=False):
    key = digest({"input": inputs, "model": encoder.metadata, "config": key_config})
    # Loading time is measurement metadata, not model semantics; excluded from stable key.
    stable = {k: v for k, v in encoder.metadata.items() if k != "load_time_sec"}
    key = digest({"inputs": inputs, "model": stable, "config": key_config})
    path = ART / "cache" / f"{key}.npy"
    if path.exists() and not force:
        return np.load(path), True, key
    emb = encoder.encode(inputs, batch_size, progress=True)
    path.parent.mkdir(exist_ok=True, parents=True)
    np.save(path, emb)
    save_json(
        path.with_suffix(".json"),
        {
            "model": stable,
            "configuration": key_config,
            "input_sha256": digest(inputs),
            "shape": list(emb.shape),
        },
    )
    return emb, False, key
