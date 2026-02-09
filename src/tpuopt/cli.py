from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .analyzer import analyze_profile
from .models import Summary, Workload
from .report import render_markdown, write_report
from .sample_data import generate_sample_data

app = typer.Typer(add_completion=False)


def _write_recommendations(summary: Summary, out_path: Path) -> None:
    lines = ["# Recommendations", ""]
    for finding in summary.findings:
        lines.append(f"{finding.rank}. {finding.symptom}")
        lines.append(f"   Root cause: {finding.likely_root_cause}")
        lines.append(f"   Check: {finding.check}")
        lines.append(f"   Remediation: {finding.remediation}")
        if finding.evidence:
            lines.append(f"   Evidence: {finding.evidence}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


@app.command()
def analyze(
    profile_dir: Path = typer.Option(..., "--profile_dir", help="Path to profiler output"),
    model_name: str = typer.Option(..., "--model_name", help="Model name"),
    workload: Workload = typer.Option(..., "--workload", help="inference or training"),
    out_dir: Path = typer.Option(Path("./outputs"), help="Output directory"),
) -> None:
    summary = analyze_profile(profile_dir, model_name, workload, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    recs_path = out_dir / "recommendations.md"
    _write_recommendations(summary, recs_path)

    # Also emit a full report for convenience.
    write_report(summary, out_dir, html=False)

    typer.echo(f"Wrote {summary_path}")
    typer.echo(f"Wrote {recs_path}")
    typer.echo(f"Charts in {out_dir / 'charts'}")


@app.command(name="sample-data")
def sample_data(
    out_dir: Path = typer.Option(Path("./sample_data"), help="Output directory"),
) -> None:
    generate_sample_data(out_dir)
    typer.echo(f"Sample data generated at {out_dir}")


@app.command()
def report(
    input: Path = typer.Option(..., "--input", help="Path to summary.json"),
    out_dir: Path = typer.Option(Path("./report"), help="Output directory"),
    html: bool = typer.Option(False, help="Also render HTML"),
) -> None:
    data = json.loads(input.read_text(encoding="utf-8"))
    summary = Summary.model_validate(data)
    write_report(summary, out_dir, html=html)
    typer.echo(f"Report written to {out_dir}")


@app.command(name="dump-markdown")
def dump_markdown(
    input: Path = typer.Option(..., "--input", help="Path to summary.json"),
) -> None:
    data = json.loads(input.read_text(encoding="utf-8"))
    summary = Summary.model_validate(data)
    typer.echo(render_markdown(summary))


if __name__ == "__main__":
    app()
