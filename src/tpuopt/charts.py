from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px


def render_charts(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if "latency_ms" in df.columns:
        fig = px.line(df, y="latency_ms", title="Latency (ms)")
        path = out_dir / "latency.html"
        fig.write_html(path)
        outputs["latency"] = str(path)

    if "compute_time_ms" in df.columns and "host_input_time_ms" in df.columns:
        fig = px.line(
            df,
            y=["compute_time_ms", "host_input_time_ms"],
            title="Compute vs Input Time (ms)",
        )
        path = out_dir / "compute_vs_input.html"
        fig.write_html(path)
        outputs["compute_vs_input"] = str(path)

    if "throughput_items_per_sec" in df.columns:
        fig = px.line(df, y="throughput_items_per_sec", title="Throughput (items/sec)")
        path = out_dir / "throughput.html"
        fig.write_html(path)
        outputs["throughput"] = str(path)

    return outputs
