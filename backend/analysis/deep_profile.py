from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def parse_trace_events(trace_path: Path, top_n: int = 10) -> Dict[str, List[Dict[str, float]]]:
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {"top_ops": [], "categories": []}

    events = data.get("traceEvents", []) if isinstance(data, dict) else []
    name_durations: Counter = Counter()
    cat_durations: Counter = Counter()

    for ev in events:
        name = ev.get("name")
        cat = ev.get("cat")
        dur = ev.get("dur")
        if isinstance(dur, (int, float)):
            if name:
                name_durations[name] += float(dur)
            if cat:
                cat_durations[cat] += float(dur)

    top_ops = [
        {"name": name, "total_us": total}
        for name, total in name_durations.most_common(top_n)
    ]
    categories = [
        {"category": cat, "total_us": total}
        for cat, total in cat_durations.most_common(top_n)
    ]

    return {"top_ops": top_ops, "categories": categories}
