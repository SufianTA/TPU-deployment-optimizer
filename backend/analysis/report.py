from __future__ import annotations

from pathlib import Path
from typing import List

from .models import AnalysisSummary


def build_exec_summary(summary: AnalysisSummary) -> List[str]:
    bullets = []
    kpis = summary.kpis
    if kpis.get("throughput_mean"):
        bullets.append(f"Mean throughput: {kpis['throughput_mean']:.2f} items/sec")
    if kpis.get("latency_p50_ms"):
        bullets.append(f"p50 latency: {kpis['latency_p50_ms']:.2f} ms")
    if summary.recommendations:
        top = summary.recommendations[0]
        bullets.append(f"Top recommendation: {top.title}")
    return bullets[:6]


def render_recommendations_md(summary: AnalysisSummary) -> str:
    lines = ["# Recommendations", ""]
    for idx, rec in enumerate(summary.recommendations, start=1):
        lines.append(f"{idx}. **{rec.title}**")
        lines.append(f"   Symptom: {rec.symptom}")
        lines.append(f"   Likely root cause: {rec.likely_root_cause}")
        lines.append(f"   Evidence: {rec.evidence}")
        lines.append(f"   Confidence: {rec.confidence}")
        lines.append(f"   Expected impact: {rec.expected_impact}")
        lines.append("   Action steps:")
        for step in rec.action_steps:
            lines.append(f"   - {step}")
        lines.append("")
    return "\n".join(lines)


def render_report_html(summary: AnalysisSummary) -> str:
    exec_summary = build_exec_summary(summary)
    recs_md = render_recommendations_md(summary)
    recs_html = recs_md.replace("\n", "<br>")

    return f"""
<html>
  <head>
    <title>TPU Deployment Optimization Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      h2 {{ margin-top: 24px; }}
      .kpi {{ display: inline-block; margin-right: 24px; }}
      .card {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; }}
    </style>
  </head>
  <body>
    <h1>TPU Deployment Optimization Report</h1>
    <div class="card">
      <div class="kpi">Model: {summary.model_id}</div>
      <div class="kpi">Framework: {summary.framework}</div>
      <div class="kpi">Workload: {summary.workload}</div>
      <div class="kpi">Device: {summary.device_type}</div>
    </div>

    <h2>Executive Summary</h2>
    <ul>
      {''.join(f'<li>{b}</li>' for b in exec_summary)}
    </ul>

    <h2>Attribution</h2>
    <pre>{summary.attribution}</pre>

    <h2>Recommendations</h2>
    <div>{recs_html}</div>

    <h2>Raw KPIs</h2>
    <pre>{summary.kpis}</pre>

    <h2>Notes / assumptions</h2>
    <ul>
      <li>Recommendations are inferred from metrics and not kernel-level profiling.</li>
      <li>If profiler artifacts are missing, confidence is lower.</li>
    </ul>
  </body>
</html>
"""


def write_reports(out_dir: Path, summary: AnalysisSummary) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    recs_path = out_dir / "recommendations.md"
    report_path = out_dir / "report.html"

    recs_path.write_text(render_recommendations_md(summary), encoding="utf-8")
    report_path.write_text(render_report_html(summary), encoding="utf-8")

    return {"recommendations": str(recs_path), "report_html": str(report_path)}
