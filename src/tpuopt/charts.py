from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px



def render_charts(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if "step_time_ms" in df.columns:
        fig = px.line(df, y="step_time_ms", title="Step Time (ms)")
        path = out_dir / "step_time.html"
        fig.write_html(path)
        outputs["step_time"] = str(path)

    if "tpu_compute_time_ms" in df.columns and "host_input_time_ms" in df.columns:
        fig = px.line(
            df,
            y=["tpu_compute_time_ms", "host_input_time_ms"],
            title="Compute vs Input Time (ms)",
        )
        path = out_dir / "compute_vs_input.html"
        fig.write_html(path)
        outputs["compute_vs_input"] = str(path)

    if "tokens_per_sec" in df.columns:
        fig = px.line(df, y="tokens_per_sec", title="Throughput (tokens/sec)")
        path = out_dir / "throughput.html"
        fig.write_html(path)
        outputs["throughput"] = str(path)

    return outputs
