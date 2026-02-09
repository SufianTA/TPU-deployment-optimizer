from __future__ import annotations

import time
import uuid
from typing import List

import numpy as np

from backend.analysis.models import RunConfig
from backend.benchmarks.base import BenchResult, generate_image_batch, generate_text_batch, make_step_metrics


def _get_device_type_jax() -> str:
    try:
        import jax

        platform = jax.devices()[0].platform
        if platform == "tpu":
            return "tpu"
        if platform == "gpu":
            return "gpu"
    except Exception:
        return "cpu"
    return "cpu"


def run_jax_benchmark(config: RunConfig, step_callback=None) -> BenchResult:
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
    except Exception as exc:
        raise RuntimeError("JAX/Flax is not installed") from exc

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    notes: List[str] = []

    device_type = _get_device_type_jax()
    config.device_type = device_type

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(10)(x)
            return x

    model = MLP()

    def _build_inputs():
        if config.model_id == "jax_flax_mlp":
            return jnp.asarray(np.random.rand(config.batch_size, 512).astype(np.float32))
        return jnp.asarray(generate_image_batch(config.batch_size))

    sample = _build_inputs()
    params = model.init(jax.random.PRNGKey(0), sample)

    @jax.jit
    def infer(p, x):
        return model.apply(p, x)

    metrics = []
    compiled = False

    total_steps = config.warmup_steps + config.steps
    for step in range(total_steps):
        warmup = step < config.warmup_steps

        host_start = time.perf_counter()
        batch = _build_inputs()
        host_end = time.perf_counter()

        start = time.perf_counter()
        _ = infer(params, batch)
        if device_type in {"tpu", "gpu"}:
            _.block_until_ready()
        end = time.perf_counter()

        compile_ms = None
        if not compiled:
            compile_ms = (end - start) * 1000.0
            compiled = True

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
                compile_ms=compile_ms,
                memory_mb=None,
                notes="",
            )
        )
        if step_callback:
            step_callback(step + 1, total_steps)

    return BenchResult(run_id=run_id, config=config, metrics=metrics, notes=notes)
