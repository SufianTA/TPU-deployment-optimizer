from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .charts import render_charts
from .io import discover_inputs, load_metrics, maybe_pull_gcp_metrics, parse_trace, parse_xla_log
from .models import InputSources, MetricsRollup, Summary, Workload
from .recommendations import build_recommendations
from .utils import compact_list, summary_stats, utc_now_iso


def _metrics_rollup(df: pd.DataFrame) -> MetricsRollup:
    rollup = {}
    for col in df.columns:
        stats = summary_stats(df[col].dropna().tolist())
        if stats:
            rollup[col] = stats
    return MetricsRollup(**rollup)


def analyze_profile(
    profile_dir: Path,
    model_name: str,
    workload: Workload,
    output_dir: Path,
) -> Summary:
    inputs = discover_inputs(profile_dir)
    notes: List[str] = []

    metrics_df = None
    if inputs.get("metrics_csv"):
        metrics_df = load_metrics(inputs["metrics_csv"])
    else:
        notes.append("metrics.csv not found; limited utilization signals available.")

    compilation = parse_xla_log(inputs["xla_log"]) if inputs.get("xla_log") else {}
    trace_artifacts = parse_trace(inputs["trace_file"]) if inputs.get("trace_file") else {}

    gcp_enabled, gcp_metrics = maybe_pull_gcp_metrics()
    if gcp_enabled:
        notes.append("GCP Monitoring integration is enabled but requires project-specific filters.")

    metrics_rollup = _metrics_rollup(metrics_df) if metrics_df is not None else MetricsRollup()

    findings = build_recommendations(
        workload=workload,
        metrics=metrics_rollup.model_dump(exclude_none=True),
        compilation=compilation,
        profile_artifacts=trace_artifacts,
    )

    summary = Summary(
        model_name=model_name,
        workload=workload,
        generated_at=utc_now_iso(),
        inputs=InputSources(
            profile_dir=str(profile_dir),
            metrics_csv=str(inputs.get("metrics_csv")) if inputs.get("metrics_csv") else None,
            xla_log=str(inputs.get("xla_log")) if inputs.get("xla_log") else None,
            trace_file=str(inputs.get("trace_file")) if inputs.get("trace_file") else None,
            gcp_monitoring=gcp_enabled,
        ),
        notes=compact_list(notes),
        metrics=metrics_rollup,
        compilation=compilation,
        profile_artifacts={**trace_artifacts, **gcp_metrics},
        findings=findings,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_df is not None:
        charts_dir = output_dir / "charts"
        render_charts(metrics_df, charts_dir)

    return summary
