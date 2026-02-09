from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    steps = 120
    latency = rng.normal(120, 10, size=steps).clip(min=80)
    host_time = rng.normal(40, 6, size=steps).clip(min=10)
    compute_time = rng.normal(70, 8, size=steps).clip(min=30)
    idle_time = rng.normal(10, 3, size=steps).clip(min=1)
    tokens = rng.normal(5200, 350, size=steps).clip(min=1000)
    batch = rng.integers(32, 64, size=steps)
    compile_time = np.zeros(steps)
    compile_time[0] = 900
    memory_mb = rng.normal(520, 20, size=steps).clip(min=400)

    df = pd.DataFrame(
        {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "step": np.arange(steps),
            "batch_size": batch,
            "precision": "fp32",
            "device_type": "cpu",
            "warmup": [True if i < 3 else False for i in range(steps)],
            "latency_ms": latency,
            "throughput_items_per_sec": tokens,
            "host_input_time_ms": host_time,
            "compute_time_ms": compute_time,
            "idle_time_ms": idle_time,
            "compile_time_ms": compile_time,
            "memory_mb": memory_mb,
            "notes": "sample",
        }
    )
    metrics_path = out_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)

    xla_log = out_dir / "xla_compile.log"
    xla_log.write_text(
        """
XLA compilation started
compile time: 1123 ms
compile time: 987 ms
recompile triggered for new shape
compile time: 1044 ms
""".strip(),
        encoding="utf-8",
    )

    trace = {
        "traceEvents": [
            {"cat": "tpu", "dur": 1234},
            {"cat": "host", "dur": 456},
            {"cat": "tpu", "dur": 987},
        ]
    }
    trace_path = out_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    return out_dir
