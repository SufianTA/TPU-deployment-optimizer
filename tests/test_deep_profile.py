from backend.analysis.deep_profile import parse_trace_events

def test_parse_trace_events():
    trace = {
        "traceEvents": [
            {"name": "conv2d", "cat": "tpu", "dur": 100},
            {"name": "conv2d", "cat": "tpu", "dur": 200},
            {"name": "input", "cat": "host", "dur": 50},
        ]
    }
    import json
    from pathlib import Path

    path = Path(".test_trace.json")
    path.write_text(json.dumps(trace), encoding="utf-8")
    parsed = parse_trace_events(path)
    path.unlink(missing_ok=True)

    assert parsed["top_ops"][0]["name"] == "conv2d"
