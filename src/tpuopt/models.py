from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Workload(str, Enum):
    inference = "inference"
    training = "training"


class InputSources(BaseModel):
    profile_dir: Optional[str] = None
    metrics_csv: Optional[str] = None
    xla_log: Optional[str] = None
    trace_file: Optional[str] = None
    gcp_monitoring: bool = False


class MetricSummary(BaseModel):
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float


class MetricsRollup(BaseModel):
    step_time_ms: Optional[MetricSummary] = None
    host_input_time_ms: Optional[MetricSummary] = None
    tpu_compute_time_ms: Optional[MetricSummary] = None
    idle_time_ms: Optional[MetricSummary] = None
    tokens_per_sec: Optional[MetricSummary] = None
    batch_size: Optional[MetricSummary] = None
    hbm_utilization: Optional[MetricSummary] = None


class Finding(BaseModel):
    rank: int
    symptom: str
    likely_root_cause: str
    check: str
    remediation: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class Summary(BaseModel):
    model_name: str
    workload: Workload
    generated_at: str
    inputs: InputSources
    notes: List[str]
    metrics: MetricsRollup
    compilation: Dict[str, Any]
    profile_artifacts: Dict[str, Any]
    findings: List[Finding]
