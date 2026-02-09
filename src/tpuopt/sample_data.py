from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd


def generate_sample_data(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    steps = 120
    step_time = rng.normal(120, 10, size=steps).clip(min=80)
    host_time = rng.normal(45, 8, size=steps).clip(min=10)
    compute_time = rng.normal(60, 6, size=steps).clip(min=30)
    idle_time = rng.normal(15, 4, size=steps).clip(min=1)
    tokens = rng.normal(5200, 350, size=steps).clip(min=1000)
    batch = rng.integers(32, 64, size=steps)
    hbm = rng.normal(0.78, 0.05, size=steps).clip(min=0.5, max=0.95)

    df = pd.DataFrame(
        {
            "step_time_ms": step_time,
            "host_input_time_ms": host_time,
            "tpu_compute_time_ms": compute_time,
            "idle_time_ms": idle_time,
            "tokens_per_sec": tokens,
            "batch_size": batch,
            "hbm_utilization": hbm,
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
