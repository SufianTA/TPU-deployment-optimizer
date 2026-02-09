from tpuopt.models import Workload
from tpuopt.recommendations import build_recommendations


def test_input_pipeline_bottleneck_detected():
    metrics = {
        "step_time_ms": {"mean": 100.0},
        "host_input_time_ms": {"mean": 40.0},
        "tpu_compute_time_ms": {"mean": 50.0},
        "idle_time_ms": {"mean": 5.0},
        "batch_size": {"mean": 8.0},
    }
    findings = build_recommendations(
        workload=Workload.inference,
        metrics=metrics,
        compilation={},
        profile_artifacts={},
    )
    assert any("Input pipeline" in f.likely_root_cause for f in findings)


def test_recompile_detected():
    findings = build_recommendations(
        workload=Workload.training,
        metrics={"step_time_ms": {"mean": 100.0}},
        compilation={"recompiles": 5},
        profile_artifacts={},
    )
    assert any("recompilations" in f.symptom.lower() for f in findings)
