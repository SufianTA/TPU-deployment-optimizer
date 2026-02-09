from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from backend.analysis.models import RunConfig, StepMetrics
from backend.analysis.utils import utc_now_iso


@dataclass
class BenchResult:
    run_id: str
    config: RunConfig
    metrics: List[StepMetrics]
    notes: List[str]


def generate_image_batch(batch_size: int, height: int = 224, width: int = 224, channels: int = 3) -> np.ndarray:
    return np.random.rand(batch_size, height, width, channels).astype(np.float32)


def generate_text_batch(batch_size: int, seq_length: int, vocab_size: int = 30522) -> np.ndarray:
    return np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int32)


def estimate_host_input_time(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def make_step_metrics(
    step: int,
    batch_size: int,
    precision: str,
    device_type: str,
    warmup: bool,
    latency_ms: float,
    throughput: float,
    host_input_ms: float,
    compile_ms: Optional[float],
    memory_mb: Optional[float],
    notes: str,
) -> StepMetrics:
    compute_ms = max(0.0, latency_ms - host_input_ms - (compile_ms or 0.0))
    idle_ms = max(0.0, latency_ms - host_input_ms - compute_ms - (compile_ms or 0.0))
    return StepMetrics(
        timestamp=utc_now_iso(),
        step=step,
        batch_size=batch_size,
        precision=precision,
        device_type=device_type,
        warmup=warmup,
        latency_ms=latency_ms,
        throughput_items_per_sec=throughput,
        host_input_time_ms=host_input_ms,
        compute_time_ms=compute_ms,
        idle_time_ms=idle_ms,
        compile_time_ms=compile_ms,
        memory_mb=memory_mb,
        notes=notes,
    )
