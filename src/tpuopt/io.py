from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .utils import safe_float


METRICS_COLUMNS = [
    "latency_ms",
    "host_input_time_ms",
    "compute_time_ms",
    "idle_time_ms",
    "throughput_items_per_sec",
    "batch_size",
    "memory_mb",
]


def discover_inputs(profile_dir: Path) -> Dict[str, Optional[Path]]:
    metrics_csv = None
    xla_log = None
    trace_file = None

    if profile_dir.exists():
        for path in profile_dir.rglob("*"):
            if path.is_file():
                if path.name.endswith("metrics.csv"):
                    metrics_csv = metrics_csv or path
                if path.name.endswith("xla_compile.log") or path.name.endswith("xla.log"):
                    xla_log = xla_log or path
                if path.name.endswith("trace.json"):
                    trace_file = trace_file or path

    return {
        "metrics_csv": metrics_csv,
        "xla_log": xla_log,
        "trace_file": trace_file,
    }


def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    # Backward compat: map old step_time_ms to latency_ms
    if "latency_ms" not in df.columns and "step_time_ms" in df.columns:
        df["latency_ms"] = df["step_time_ms"]
    missing = [c for c in METRICS_COLUMNS if c not in df.columns]
    for col in missing:
        df[col] = float("nan")
    return df


def parse_xla_log(xla_log: Path) -> Dict[str, float]:
    text = xla_log.read_text(encoding="utf-8", errors="ignore")
    recompiles = len(re.findall(r"recompile", text, flags=re.IGNORECASE))
    compile_times = [
        safe_float(m)
        for m in re.findall(r"compile time[:=]\s*(\d+\.?\d*)\s*ms", text, flags=re.IGNORECASE)
    ]
    return {
        "recompiles": recompiles,
        "compile_time_ms_mean": float(pd.Series(compile_times).mean()) if compile_times else None,
        "compile_time_ms_max": float(pd.Series(compile_times).max()) if compile_times else None,
    }


def parse_trace(trace_file: Path) -> Dict[str, float]:
    # Best-effort: summarize event categories if present.
    try:
        data = json.loads(trace_file.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}

    events = data.get("traceEvents", []) if isinstance(data, dict) else []
    categories = {}
    for ev in events:
        cat = ev.get("cat")
        dur = ev.get("dur")
        if cat and isinstance(dur, (int, float)):
            categories.setdefault(cat, []).append(float(dur))

    return {
        "trace_event_categories": len(categories),
        "trace_total_events": len(events),
    }


def maybe_pull_gcp_metrics() -> Tuple[bool, Dict[str, float]]:
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        return False, {}

    try:
        from google.cloud import monitoring_v3
    except Exception:
        return False, {}

    _ = monitoring_v3
    return True, {}
