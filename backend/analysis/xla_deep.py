from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


HLO_OP_RE = re.compile(r"\b(%?\w+)\s*=\s*(\w+)")


def parse_hlo_stats(hlo_path: Path) -> Dict[str, int]:
    try:
        text = hlo_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    counts: Dict[str, int] = {}
    for line in text.splitlines():
        match = HLO_OP_RE.search(line)
        if match:
            op = match.group(2)
            counts[op] = counts.get(op, 0) + 1
    return counts


def parse_xla_compile_log(log_path: Path) -> Dict[str, float]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    recompiles = len(re.findall(r"recompile", text, flags=re.IGNORECASE))
    compile_times = [
        float(m)
        for m in re.findall(r"compile time[:=]\s*(\d+\.?\d*)\s*ms", text, flags=re.IGNORECASE)
    ]
    return {
        "recompiles": recompiles,
        "compile_time_ms_mean": sum(compile_times) / len(compile_times) if compile_times else 0.0,
        "compile_time_ms_max": max(compile_times) if compile_times else 0.0,
    }
