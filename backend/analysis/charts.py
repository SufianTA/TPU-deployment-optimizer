from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_charts(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    # Latency distribution
    fig = px.histogram(df, x="latency_ms", nbins=30, title="Latency Distribution (ms)")
    path = out_dir / "latency_distribution.html"
    fig.write_html(path)
    outputs["latency_distribution"] = str(path)

    # Throughput over time
    fig = px.line(df, x="step", y="throughput_items_per_sec", title="Throughput Over Time")
    path = out_dir / "throughput_over_time.html"
    fig.write_html(path)
    outputs["throughput_over_time"] = str(path)

    # Breakdown stacked chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name="input", x=df["step"], y=df["host_input_time_ms"]))
    fig.add_trace(go.Bar(name="compute", x=df["step"], y=df["compute_time_ms"]))
    fig.add_trace(go.Bar(name="idle", x=df["step"], y=df["idle_time_ms"]))
    if "compile_time_ms" in df.columns:
        fig.add_trace(go.Bar(name="compile", x=df["step"], y=df["compile_time_ms"].fillna(0)))
    fig.update_layout(barmode="stack", title="Latency Breakdown (ms)")
    path = out_dir / "latency_breakdown.html"
    fig.write_html(path)
    outputs["latency_breakdown"] = str(path)

    # Utilization proxy
    util_proxy = 1 - (df["idle_time_ms"] / df["latency_ms"]).replace([float("inf")], 0).fillna(0)
    fig = px.line(x=df["step"], y=util_proxy, title="Utilization Proxy (1 - idle ratio)")
    path = out_dir / "utilization_proxy.html"
    fig.write_html(path)
    outputs["utilization_proxy"] = str(path)

    # Comparison placeholder (run A vs run B)
    comp = out_dir / "comparison_placeholder.html"
    comp.write_text(
        "<html><body><h3>Comparison View</h3><p>Upload a second run to compare A vs B.</p></body></html>",
        encoding="utf-8",
    )
    outputs["comparison"] = str(comp)

    # Combined charts index
    index = out_dir / "charts.html"
    index.write_text(
        "<html><body>"
        "<h2>TPU Optimization Lab Charts</h2>"
        "<ul>"
        + "".join(f"<li><a href='{Path(p).name}'>{k}</a></li>" for k, p in outputs.items())
        + "</ul></body></html>",
        encoding="utf-8",
    )
    outputs["index"] = str(index)

    return outputs
