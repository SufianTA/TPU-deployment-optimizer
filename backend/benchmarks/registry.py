from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np

from backend.analysis.models import RunConfig
from backend.analysis.utils import utc_now_iso
from backend.benchmarks.base import BenchResult, make_step_metrics
from backend.benchmarks.jax_bench import run_jax_benchmark
from backend.benchmarks.tf_bench import run_tf_benchmark
from backend.benchmarks.torch_bench import run_torch_benchmark
import os


MODEL_CARDS: Dict[str, Dict[str, str]] = {
    "tf_mobilenet_v2": {
        "framework": "tensorflow",
        "title": "MobileNetV2 (Image)",
        "input_shape": "[batch, 224, 224, 3]",
        "throughput_notes": "Lightweight CNN; typically input-bound if preprocessing is heavy.",
        "bottlenecks": "Input pipeline, small batch sizes, precision selection",
    },
    "tf_text_classifier": {
        "framework": "tensorflow",
        "title": "Simple Text Classifier",
        "input_shape": "[batch, seq_length]",
        "throughput_notes": "Embedding + dense layers; good for latency demos.",
        "bottlenecks": "Embedding table access, batch size, compilation with dynamic shapes",
    },
    "jax_flax_mlp": {
        "framework": "jax",
        "title": "Flax MLP",
        "input_shape": "[batch, 512]",
        "throughput_notes": "Compute-heavy with small input; good for JIT behavior.",
        "bottlenecks": "Compilation spikes, batch sizing",
    },
    "torch_resnet18": {
        "framework": "pytorch",
        "title": "ResNet18 (Image)",
        "input_shape": "[batch, 3, 224, 224]",
        "throughput_notes": "Balanced compute; often benefits from mixed precision.",
        "bottlenecks": "Compute bound, batch sizing",
    },
    "torch_tiny_text": {
        "framework": "pytorch",
        "title": "Tiny Text Classifier",
        "input_shape": "[batch, seq_length]",
        "throughput_notes": "Simple embedding + linear; good for latency baseline.",
        "bottlenecks": "Input pipeline and small batches",
    },
}


def list_models() -> Dict[str, Dict[str, str]]:
    return MODEL_CARDS


def _simulate_benchmark(config: RunConfig, reason: str) -> BenchResult:
    import uuid

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    rng = np.random.default_rng(7)
    metrics = []
    notes = [f"Simulated run: {reason}"]

    for step in range(config.warmup_steps + config.steps):
        warmup = step < config.warmup_steps
        latency_ms = float(rng.normal(120, 8))
        host_ms = float(rng.normal(35, 5))
        compile_ms = float(rng.normal(12, 3)) if step == 0 else 0.0
        throughput = config.batch_size / (latency_ms / 1000.0)

        metrics.append(
            make_step_metrics(
                step=step,
                batch_size=config.batch_size,
                precision=config.precision,
                device_type=config.device_type or "cpu",
                warmup=warmup,
                latency_ms=latency_ms,
                throughput=throughput,
                host_input_ms=host_ms,
                compile_ms=compile_ms,
                memory_mb=None,
                notes="simulated",
            )
        )

    return BenchResult(run_id=run_id, config=config, metrics=metrics, notes=notes)


def run_benchmark(config: RunConfig, step_callback=None, force_simulate: bool = False) -> BenchResult:
    if force_simulate or os.getenv("TPUOPT_SIMULATE", "").lower() in {"1", "true", "yes"}:
        return _simulate_benchmark(config, "Simulation enabled for this environment")
    if config.framework == "tensorflow":
        try:
            return run_tf_benchmark(config, step_callback=step_callback)
        except Exception as exc:
            return _simulate_benchmark(config, f"TensorFlow unavailable: {exc}")
    if config.framework == "jax":
        try:
            return run_jax_benchmark(config, step_callback=step_callback)
        except Exception as exc:
            return _simulate_benchmark(config, f"JAX unavailable: {exc}")
    if config.framework == "pytorch":
        try:
            return run_torch_benchmark(config, step_callback=step_callback)
        except Exception as exc:
            return _simulate_benchmark(config, f"PyTorch unavailable: {exc}")
    raise ValueError(f"Unsupported framework: {config.framework}")
