from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Iterable[float], q: float) -> float:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return float("nan")
    return float(np.percentile(data, q))


def summary_stats(values: Iterable[float]) -> dict:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {}
    return {
        "mean": float(np.mean(data)),
        "p50": percentile(data, 50),
        "p90": percentile(data, 90),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
    }


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def compact_list(values: List[str]) -> List[str]:
    return [v for v in values if v]
