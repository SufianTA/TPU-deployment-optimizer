from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .models import AnalysisSummary, Recommendation, RunConfig
from .utils import utc_now_iso


def _safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if not series.dropna().empty else float("nan")


def _percentile(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean, q))


def compute_attribution(df: pd.DataFrame) -> Dict[str, float]:
    latency = _safe_mean(df["latency_ms"])
    if not latency or np.isnan(latency) or latency <= 0:
        return {"input_bound": 0.0, "compute_bound": 0.0, "compile_bound": 0.0, "idle_bound": 0.0}

    input_bound = _safe_mean(df["host_input_time_ms"]) / latency
    compute_bound = _safe_mean(df["compute_time_ms"]) / latency
    idle_bound = _safe_mean(df["idle_time_ms"]) / latency
    compile_bound = 0.0
    if "compile_time_ms" in df.columns:
        compile_bound = _safe_mean(df["compile_time_ms"]) / latency

    raw = np.array([input_bound, compute_bound, compile_bound, idle_bound])
    raw = np.where(np.isnan(raw), 0.0, raw)
    total = raw.sum()
    if total <= 0:
        return {"input_bound": 0.0, "compute_bound": 0.0, "compile_bound": 0.0, "idle_bound": 0.0}
    norm = raw / total
    return {
        "input_bound": float(norm[0]),
        "compute_bound": float(norm[1]),
        "compile_bound": float(norm[2]),
        "idle_bound": float(norm[3]),
    }


def _confidence(value: float, threshold: float, cap: float = 0.95) -> float:
    if np.isnan(value):
        return 0.0
    score = min(cap, max(0.1, value / threshold))
    return float(round(score, 2))


def _compile_event_ratio(df: pd.DataFrame) -> float:
    if "compile_time_ms" not in df.columns:
        return 0.0
    series = df["compile_time_ms"].dropna()
    if series.empty:
        return 0.0
    return float((series > 0).sum() / len(df))


def generate_recommendations(df: pd.DataFrame, config: RunConfig) -> List[Recommendation]:
    recs: List[Recommendation] = []

    latency = _safe_mean(df["latency_ms"])
    p50 = _percentile(df["latency_ms"], 50)
    p90 = _percentile(df["latency_ms"], 90)
    p99 = _percentile(df["latency_ms"], 99)
    throughput = _safe_mean(df["throughput_items_per_sec"])
    host_ratio = _safe_mean(df["host_input_time_ms"]) / latency if latency else 0.0
    compute_ratio = _safe_mean(df["compute_time_ms"]) / latency if latency else 0.0
    idle_ratio = _safe_mean(df["idle_time_ms"]) / latency if latency else 0.0
    compile_ratio = _safe_mean(df.get("compile_time_ms", pd.Series(dtype=float))) / latency if latency else 0.0
    compile_event_ratio = _compile_event_ratio(df)

    batch_mean = _safe_mean(df["batch_size"]) if "batch_size" in df.columns else float("nan")
    memory_p95 = _percentile(df.get("memory_mb", pd.Series(dtype=float)), 95)

    if host_ratio > 0.3:
        recs.append(
            Recommendation(
                title="Input pipeline is host-bound",
                symptom="High host input time relative to total latency",
                likely_root_cause="Input preprocessing or IO is not keeping up with device execution",
                evidence={"host_ratio": round(host_ratio, 3)},
                confidence=_confidence(host_ratio, 0.3),
                expected_impact="Medium to high (10–35%)",
                action_steps=[
                    "Increase input pipeline parallelism and prefetch depth",
                    "Enable dataset caching or pre-shard input files",
                    "Move heavy preprocessing out of the hot path",
                ],
            )
        )

    if compile_ratio > 0.1 or compile_event_ratio > 0.2:
        recs.append(
            Recommendation(
                title="Compilation overhead is visible",
                symptom="Compile time is a noticeable portion of total latency",
                likely_root_cause="Dynamic shapes or frequent recompilations",
                evidence={
                    "compile_ratio": round(compile_ratio, 3),
                    "compile_event_ratio": round(compile_event_ratio, 3),
                },
                confidence=_confidence(max(compile_ratio, compile_event_ratio), 0.1),
                expected_impact="Medium (8–25%)",
                action_steps=[
                    "Bucket inputs to fixed shapes",
                    "Enable compilation caching or warmup",
                    "Avoid shape-changing control flow in hot path",
                ],
            )
        )

    if idle_ratio > 0.2:
        recs.append(
            Recommendation(
                title="Device idle time is high",
                symptom="Device spends significant time idle between steps",
                likely_root_cause="Small batches or host stalls",
                evidence={"idle_ratio": round(idle_ratio, 3)},
                confidence=_confidence(idle_ratio, 0.2),
                expected_impact="Medium (10–30%)",
                action_steps=[
                    "Increase batch size within memory limits",
                    "Overlap input pipeline with compute",
                    "Use request aggregation or micro-batching for serving",
                ],
            )
        )

    if not np.isnan(batch_mean) and batch_mean < (16 if config.workload == "training" else 64):
        recs.append(
            Recommendation(
                title="Batch size is likely too small",
                symptom="Throughput is limited by small effective batch size",
                likely_root_cause="Conservative batch sizing or memory headroom concerns",
                evidence={"batch_size_mean": round(batch_mean, 2)},
                confidence=_confidence(64 - batch_mean, 32, cap=0.85),
                expected_impact="Medium to high (10–40%)",
                action_steps=[
                    "Increase batch size gradually while watching latency and memory",
                    "Use gradient accumulation for training if needed",
                    "Enable dynamic batching in serving",
                ],
            )
        )

    if compute_ratio > 0.6 and config.precision == "fp32":
        recs.append(
            Recommendation(
                title="Use TPU-friendly precision",
                symptom="Compute dominates latency and precision is fp32",
                likely_root_cause="FP32 compute is slower on accelerators",
                evidence={"compute_ratio": round(compute_ratio, 3)},
                confidence=_confidence(compute_ratio, 0.6),
                expected_impact="Medium (8–20%)",
                action_steps=[
                    "Switch to bfloat16 where numerically safe",
                    "Validate accuracy with a small eval set",
                ],
            )
        )

    if p99 > 1.5 * p50:
        recs.append(
            Recommendation(
                title="Tail latency is elevated",
                symptom="p99 is much higher than p50",
                likely_root_cause="Queueing or uneven input/compute overlap",
                evidence={"p50_ms": round(p50, 2), "p99_ms": round(p99, 2)},
                confidence=_confidence(p99 / p50, 1.5, cap=0.9),
                expected_impact="Low to medium (5–15%)",
                action_steps=[
                    "Limit max in-flight requests",
                    "Use batching with a small max wait window",
                    "Smooth input pipeline spikes",
                ],
            )
        )

    if not np.isnan(memory_p95) and memory_p95 > 16000:
        recs.append(
            Recommendation(
                title="Memory pressure likely",
                symptom="Memory usage is high",
                likely_root_cause="Large activations or batch sizes near limit",
                evidence={"memory_p95_mb": round(memory_p95, 1)},
                confidence=_confidence(memory_p95, 16000, cap=0.85),
                expected_impact="Low to medium (5–15%)",
                action_steps=[
                    "Reduce sequence length or activation size",
                    "Use activation checkpointing",
                    "Increase sharding or reduce batch size",
                ],
            )
        )

    if compute_ratio > 0.6:
        recs.append(
            Recommendation(
                title="Check for graph-level optimizations",
                symptom="Compute dominates with limited throughput gains",
                likely_root_cause="Lack of operator fusion or inefficient kernels",
                evidence={"compute_ratio": round(compute_ratio, 3)},
                confidence=_confidence(compute_ratio, 0.6, cap=0.8),
                expected_impact="Low to medium (5–12%)",
                action_steps=[
                    "Review profiler op breakdown for unfused hot spots",
                    "Enable XLA or torch.compile (where applicable)",
                ],
            )
        )

    if throughput and throughput < 0.7 * df["throughput_items_per_sec"].max():
        recs.append(
            Recommendation(
                title="Serving concurrency tuning",
                symptom="Throughput varies significantly across steps",
                likely_root_cause="Inconsistent request concurrency or batch assembly",
                evidence={"throughput_mean": round(throughput, 2)},
                confidence=0.6,
                expected_impact="Low to medium (5–12%)",
                action_steps=[
                    "Tune max in-flight requests",
                    "Use dynamic batching with a max queue delay",
                ],
            )
        )

    if not recs:
        recs.append(
            Recommendation(
                title="Collect more signals",
                symptom="Metrics are insufficient to identify a dominant bottleneck",
                likely_root_cause="Incomplete profiling artifacts",
                evidence={},
                confidence=0.4,
                expected_impact="N/A",
                action_steps=[
                    "Capture a TensorBoard profile window",
                    "Include compile_time_ms and memory_mb if possible",
                ],
            )
        )

    return recs


def build_summary(df: pd.DataFrame, config: RunConfig, run_id: str) -> AnalysisSummary:
    attribution = compute_attribution(df)

    kpis = {
        "latency_p50_ms": _percentile(df["latency_ms"], 50),
        "latency_p90_ms": _percentile(df["latency_ms"], 90),
        "latency_p99_ms": _percentile(df["latency_ms"], 99),
        "throughput_mean": _safe_mean(df["throughput_items_per_sec"]),
        "utilization_proxy": 1.0 - float(attribution.get("idle_bound", 0.0)),
    }

    summary = AnalysisSummary(
        run_id=run_id,
        model_id=config.model_id,
        framework=config.framework,
        workload=config.workload,
        device_type=config.device_type,
        generated_at=utc_now_iso(),
        attribution=attribution,
        kpis=kpis,
        recommendations=generate_recommendations(df, config),
        notes=[],
        artifacts={},
    )
    return summary
