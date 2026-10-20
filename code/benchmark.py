import argparse, platform, subprocess, time
import numpy as np
import torch
from common import (
    ART,
    load_data,
    document_text,
    save_json,
    seed_all,
    initialize_manifest,
)
from neural import Encoder, MODELS, sync, peak_rss_mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--kind", choices=list(MODELS))
    args = p.parse_args()
    initialize_manifest()
    docs, claims = load_data()
    seed_all()
    kinds = [args.kind] if args.kind else list(MODELS)
    hardware = {
        "platform": platform.platform(),
        "cpu": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "dtype": "float32",
        "seed": 42,
        "implementation_start_utc": "2026-10-13T11:04:52Z",
    }
    save_json(ART / "hardware.json", hardware)
    print(hardware, flush=True)
    for kind in kinds:
        model = Encoder(kind, args.device)
        texts = [document_text(d) for d in docs[:32]]
        queries = [c["claim"] for c in claims["val"][:32]]
        inputs = (
            list(zip(queries, texts))
            if kind == "reranker"
            else list(zip(texts, queries))
            if kind == "nli"
            else [(d["title"], " ".join(d["abstract"])) for d in docs[:32]]
            if kind == "article"
            else queries
            if kind == "query"
            else texts
        )
        features = model.tokenize(inputs[:2])
        device_out = model.encode(inputs[:2])
        model.model.to("cpu")
        model.device = "cpu"
        cpu_out = model.encode(inputs[:2])
        model.model.to(args.device)
        model.device = args.device
        difference = float(np.max(np.abs(device_out - cpu_out)))
        tolerance = 0.01
        assert difference < tolerance, (kind, difference)
        model.encode(inputs[:2])
        sync(args.device)
        start = time.perf_counter()
        out = model.encode(inputs, 8 if kind in {"reranker", "nli"} else 16)
        sync(args.device)
        elapsed = time.perf_counter() - start
        result = {
            **model.metadata,
            "finite_forward": bool(np.isfinite(out).all()),
            "cpu_mps_max_abs_difference": difference,
            "comparison_absolute_tolerance": tolerance,
            "benchmark_items": len(inputs),
            "inference_sec": elapsed,
            "items_per_sec": len(inputs) / elapsed,
            "peak_rss_mb": peak_rss_mb(),
        }
        if kind == "general":
            model.model.train()
            opt = torch.optim.AdamW(model.model.parameters(), lr=2e-5)
            q = model.forward(model.tokenize(queries[:2]))
            pos = model.forward(model.tokenize(texts[:2]))
            neg = model.forward(model.tokenize(texts[2:4]))
            loss = torch.nn.functional.cross_entropy(
                torch.stack([(q * pos).sum(1), (q * neg).sum(1)], 1) / 0.05,
                torch.zeros(2, dtype=torch.long, device=args.device),
            )
            loss.backward()
            assert torch.isfinite(loss) and all(
                torch.isfinite(p.grad).all()
                for p in model.model.parameters()
                if p.grad is not None
            )
            opt.step()
            result["finite_backward_optimizer_step"] = True
            result["smoke_loss"] = loss.item()
            del opt
        if args.device == "mps":
            result.update(
                mps_allocated_bytes=torch.mps.current_allocated_memory(),
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
            )
        save_json(ART / f"benchmark_{kind}.json", result)
        print(result, flush=True)
        model.close()


if __name__ == "__main__":
    main()
