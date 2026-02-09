from __future__ import annotations

import time
import uuid
from typing import List

import numpy as np

from backend.analysis.models import RunConfig
from backend.benchmarks.base import BenchResult, generate_image_batch, generate_text_batch, make_step_metrics


def _get_device_type_torch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        return "cpu"
    return "cpu"


def _build_text_model_torch(vocab_size: int, seq_length: int):
    import torch
    import torch.nn as nn

    class TinyText(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 64)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = self.embed(x)
            x = x.mean(dim=1)
            return self.fc(x)

    return TinyText()


def _load_model(model_id: str, seq_length: int):
    import torch
    if model_id == "torch_resnet18":
        from torchvision import models

        return models.resnet18(weights=None)
    if model_id == "torch_tiny_text":
        return _build_text_model_torch(30522, seq_length)
    raise ValueError(f"Unknown Torch model_id: {model_id}")


def run_torch_benchmark(config: RunConfig, step_callback=None) -> BenchResult:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is not installed") from exc

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    notes: List[str] = []

    device_type = _get_device_type_torch()
    config.device_type = device_type
    device = torch.device("cuda" if device_type == "gpu" else "cpu")

    model = _load_model(config.model_id, config.seq_length).to(device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = []

    total_steps = config.warmup_steps + config.steps
    for step in range(total_steps):
        warmup = step < config.warmup_steps

        host_start = time.perf_counter()
        if config.model_id == "torch_resnet18":
            batch = generate_image_batch(config.batch_size)
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2)
        else:
            batch = generate_text_batch(config.batch_size, config.seq_length)
            batch = torch.from_numpy(batch)
        host_end = time.perf_counter()

        batch = batch.to(device)

        start = time.perf_counter()
        if config.training_micro_steps > 0:
            optimizer.zero_grad()
            output = model(batch)
            loss = output.mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _ = model(batch)
        if device_type == "gpu":
            torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        host_ms = (host_end - host_start) * 1000.0
        throughput = config.batch_size / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

        metrics.append(
            make_step_metrics(
                step=step,
                batch_size=config.batch_size,
                precision=config.precision,
                device_type=device_type,
                warmup=warmup,
                latency_ms=latency_ms,
                throughput=throughput,
                host_input_ms=host_ms,
                compile_ms=None,
                memory_mb=None,
                notes="",
            )
        )
        if step_callback:
            step_callback(step + 1, total_steps)

    return BenchResult(run_id=run_id, config=config, metrics=metrics, notes=notes)
