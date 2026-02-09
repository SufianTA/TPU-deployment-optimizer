from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .models import AnalysisSummary, StepMetrics


def write_metrics_csv(path: Path, rows: Iterable[StepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary_json(path: Path, summary: AnalysisSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(summary)
    data["recommendations"] = [asdict(r) for r in summary.recommendations]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
