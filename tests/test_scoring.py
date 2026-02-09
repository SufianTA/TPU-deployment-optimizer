import pandas as pd

from backend.analysis.scoring import compute_attribution, generate_recommendations
from backend.analysis.models import RunConfig


def _make_df():
    return pd.DataFrame(
        {
            "latency_ms": [100, 110, 90],
            "host_input_time_ms": [40, 38, 42],
            "compute_time_ms": [50, 55, 45],
            "idle_time_ms": [10, 12, 8],
            "compile_time_ms": [0, 0, 0],
            "batch_size": [16, 16, 16],
            "throughput_items_per_sec": [160, 150, 170],
        }
    )


def test_attribution_sums_to_one():
    df = _make_df()
    attrib = compute_attribution(df)
    total = sum(attrib.values())
    assert 0.99 <= total <= 1.01


def test_recommendations_include_input_pipeline():
    df = _make_df()
    config = RunConfig(
        model_id="tf_mobilenet_v2",
        framework="tensorflow",
        workload="inference",
        batch_size=16,
        seq_length=128,
        warmup_steps=2,
        steps=5,
        precision="fp32",
        device_type="cpu",
    )
    recs = generate_recommendations(df, config)
    assert any("Input pipeline" in r.title for r in recs)
