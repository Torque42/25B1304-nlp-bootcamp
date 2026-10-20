"""Fixed-shape inference with small batches and bounded MPS cache residency."""

import gc, os, time
import numpy as np
import torch
from neural import Encoder, sync


class SafeEncoder(Encoder):
    def __init__(self, *args, batch_size=2, **kwargs):
        device = kwargs.get(
            "device", args[1] if len(args) > 1 else os.environ.get("NLP_DEVICE", "mps")
        )
        if device == "mps":
            # Strengthen the memory limit; never disable the watermark.
            torch.mps.set_per_process_memory_fraction(0.4)
        super().__init__(*args, **kwargs)
        self.safe_batch = batch_size
        self.metadata.update(
            inference_batch_size=batch_size,
            inference_padding="fixed max length",
            mps_memory_fraction=0.4 if device == "mps" else None,
            cache_cleanup="after every inference batch",
        )

    def encode(self, inputs, batch_size=None, progress=False):
        outputs = []
        start = time.perf_counter()
        for i in range(0, len(inputs), self.safe_batch):
            batch = inputs[i : i + self.safe_batch]
            features = self.tokenize(batch)
            # Pad CPU tensors to the same total length; attention mask stays zero in padding.
            length = features["input_ids"].shape[1]
            for key, value in list(features.items()):
                pad_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                if length < self.max_length:
                    features[key] = torch.nn.functional.pad(
                        value, (0, self.max_length - length), value=pad_value
                    )
            with torch.inference_mode():
                output = self.forward(features).cpu().numpy().copy()
            assert np.isfinite(output).all()
            outputs.append(output)
            del features, output, batch
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
                self.metadata["post_cleanup_allocated_bytes"] = (
                    torch.mps.current_allocated_memory()
                )
                self.metadata["post_cleanup_driver_bytes"] = (
                    torch.mps.driver_allocated_memory()
                )
                if self.metadata["post_cleanup_driver_bytes"] > 6 * 1024**3:
                    raise RuntimeError(
                        "MPS driver memory exceeded6GiB guard; prior score chunks remain durable"
                    )
            if progress and (
                (i + self.safe_batch) % 128 == 0 or i + self.safe_batch >= len(inputs)
            ):
                memory = (
                    {
                        "allocated_gib": torch.mps.current_allocated_memory() / 1024**3,
                        "driver_gib": torch.mps.driver_allocated_memory() / 1024**3,
                    }
                    if self.device == "mps"
                    else {}
                )
                print(
                    self.kind,
                    min(i + self.safe_batch, len(inputs)),
                    "/",
                    len(inputs),
                    "sec",
                    time.perf_counter() - start,
                    memory,
                    flush=True,
                )
        return np.concatenate(outputs)
