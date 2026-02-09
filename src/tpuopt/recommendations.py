from __future__ import annotations

from typing import Dict, List

import numpy as np

from .models import Finding, Workload


def _ratio(numer: float, denom: float) -> float:
    if denom == 0 or np.isnan(denom) or np.isnan(numer):
        return float("nan")
    return float(numer / denom)


def build_recommendations(
    workload: Workload,
    metrics: Dict[str, Dict[str, float]],
    compilation: Dict[str, float],
    profile_artifacts: Dict[str, float],
) -> List[Finding]:
    findings: List[Finding] = []

    step = metrics.get("step_time_ms", {})
    host = metrics.get("host_input_time_ms", {})
    compute = metrics.get("tpu_compute_time_ms", {})
    idle = metrics.get("idle_time_ms", {})
    batch = metrics.get("batch_size", {})
    hbm = metrics.get("hbm_utilization", {})

    step_mean = step.get("mean", float("nan"))
    host_mean = host.get("mean", float("nan"))
    compute_mean = compute.get("mean", float("nan"))
    idle_mean = idle.get("mean", float("nan"))
    batch_mean = batch.get("mean", float("nan"))
    hbm_mean = hbm.get("mean", float("nan"))

    host_ratio = _ratio(host_mean, step_mean)
    idle_ratio = _ratio(idle_mean, step_mean)
    compute_ratio = _ratio(compute_mean, step_mean)

    rank = 1

    if not np.isnan(host_ratio) and host_ratio > 0.3:
        findings.append(
            Finding(
                rank=rank,
                symptom="High host input time relative to step time",
                likely_root_cause="Input pipeline bottleneck (host-bound preprocessing or IO)",
                check="host_input_time_ms / step_time_ms",
                remediation="Increase input pipeline parallelism, enable dataset caching, and prefetch."
                " Check CPU utilization and data source throughput.",
                evidence={"host_ratio": round(host_ratio, 3)},
            )
        )
        rank += 1

    if not np.isnan(idle_ratio) and idle_ratio > 0.2:
        findings.append(
            Finding(
                rank=rank,
                symptom="High TPU idle time during steps",
                likely_root_cause="Underutilization from small batches or host stalls",
                check="idle_time_ms / step_time_ms",
                remediation="Increase batch size (within memory limits) and overlap input pipeline"
                " with compute. Consider asynchronous input feeding.",
                evidence={"idle_ratio": round(idle_ratio, 3)},
            )
        )
        rank += 1

    if not np.isnan(compute_ratio) and compute_ratio < 0.6:
        findings.append(
            Finding(
                rank=rank,
                symptom="Low TPU compute time fraction",
                likely_root_cause="Excessive host overheads or frequent re-compilation",
                check="tpu_compute_time_ms / step_time_ms",
                remediation="Stabilize shapes, reduce Python overhead, and avoid dynamic control flow"
                " that triggers recompilation.",
                evidence={"compute_ratio": round(compute_ratio, 3)},
            )
        )
        rank += 1

    recompiles = compilation.get("recompiles")
    if recompiles and recompiles > 2:
        findings.append(
            Finding(
                rank=rank,
                symptom="Repeated XLA recompilations",
                likely_root_cause="Dynamic shapes or changing input signatures",
                check="XLA compile logs (recompile count)",
                remediation="Pad/pack to fixed shapes, cache compiled executables, or"
                " separate shape buckets.",
                evidence={"recompiles": recompiles},
            )
        )
        rank += 1

    if not np.isnan(batch_mean) and batch_mean < (16 if workload == Workload.training else 64):
        findings.append(
            Finding(
                rank=rank,
                symptom="Batch size appears small for TPU throughput",
                likely_root_cause="Conservative batch sizing or memory constraints",
                check="batch_size (mean)",
                remediation="Gradually increase batch size while monitoring HBM usage and"
                " step time. Use gradient accumulation for training if needed.",
                evidence={"batch_size_mean": round(batch_mean, 2)},
            )
        )
        rank += 1

    if not np.isnan(hbm_mean) and hbm_mean > 0.85:
        findings.append(
            Finding(
                rank=rank,
                symptom="High HBM utilization",
                likely_root_cause="Memory pressure causing stalls or limited batch growth",
                check="hbm_utilization",
                remediation="Use mixed precision (bfloat16), reduce activation size,"
                " or increase sharding to spread memory.",
                evidence={"hbm_utilization_mean": round(hbm_mean, 3)},
            )
        )
        rank += 1

    if workload == Workload.training:
        findings.append(
            Finding(
                rank=rank,
                symptom="Training workload may be communication-bound",
                likely_root_cause="All-reduce or cross-replica overhead",
                check="Profiler step breakdown and collective ops time",
                remediation="Tune data/model parallelism, adjust global batch size, and"
                " enable fused collectives where available.",
                evidence={},
            )
        )
        rank += 1

    if not findings:
        findings.append(
            Finding(
                rank=1,
                symptom="No strong bottlenecks detected from available signals",
                likely_root_cause="Limited or incomplete profiling data",
                check="Verify metrics CSV, XLA logs, and trace artifacts",
                remediation="Collect a TPU profile with TensorBoard and include metrics"
                " such as step_time_ms and tpu_compute_time_ms.",
                evidence=profile_artifacts,
            )
        )

    return findings
