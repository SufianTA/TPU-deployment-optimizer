from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .models import AnalysisSummary, RunConfig
from .scoring import build_summary


REQUIRED_COLUMNS = [
    "timestamp",
    "step",
    "batch_size",
    "precision",
    "device_type",
    "warmup",
    "latency_ms",
    "throughput_items_per_sec",
    "host_input_time_ms",
    "compute_time_ms",
    "idle_time_ms",
    "compile_time_ms",
    "memory_mb",
    "notes",
]


def load_metrics_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def analyze_metrics(path: Path, config: RunConfig, run_id: str) -> AnalysisSummary:
    df = load_metrics_df(path)
    return build_summary(df, config, run_id)
