from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .models import Summary


def _format_findings(summary: Summary) -> List[str]:
    lines = []
    for finding in summary.findings:
        lines.append(f"{finding.rank}. **{finding.symptom}**")
        lines.append(f"   Likely root cause: {finding.likely_root_cause}")
        lines.append(f"   What to check: {finding.check}")
        lines.append(f"   Remediation: {finding.remediation}")
        if finding.evidence:
            lines.append(f"   Evidence: {finding.evidence}")
        lines.append("")
    return lines


def render_markdown(summary: Summary) -> str:
    lines = [
        f"# TPU Deployment Optimization Report",
        "",
        f"Model: `{summary.model_name}`",
        f"Workload: `{summary.workload}`",
        f"Generated: `{summary.generated_at}`",
        "",
        "## Notes",
    ]
    if summary.notes:
        lines.extend([f"- {note}" for note in summary.notes])
    else:
        lines.append("- None")

    lines.extend(["", "## Metrics Summary"])
    metrics = summary.metrics.model_dump(exclude_none=True)
    if metrics:
        for name, stats in metrics.items():
            lines.append(f"- `{name}`: {stats}")
    else:
        lines.append("- No metrics available")

    lines.extend(["", "## Findings & Recommendations"])
    lines.extend(_format_findings(summary))

    lines.extend(["", "## Inputs"])
    lines.append(f"- Profile dir: `{summary.inputs.profile_dir}`")
    lines.append(f"- Metrics CSV: `{summary.inputs.metrics_csv}`")
    lines.append(f"- XLA log: `{summary.inputs.xla_log}`")
    lines.append(f"- Trace file: `{summary.inputs.trace_file}`")
    lines.append(f"- GCP Monitoring: `{summary.inputs.gcp_monitoring}`")

    lines.extend(["", "## Notes / assumptions"])
    lines.append(
        "- Recommendations are inferred from available metrics and may be incomplete "
        "if profiling artifacts are missing. Validate with full TensorBoard profiling."
    )

    return "\n".join(lines)


def render_html(summary: Summary) -> str:
    md = render_markdown(summary)
    return f"""<html><body><pre>{md}</pre></body></html>"""


def write_report(summary: Summary, out_dir: Path, html: bool = False) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    outputs = {"markdown": str(md_path)}
    if html:
        html_path = out_dir / "report.html"
        html_path.write_text(render_html(summary), encoding="utf-8")
        outputs["html"] = str(html_path)
    return outputs
