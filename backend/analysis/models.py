"""Core data structures for TPU Optimization Lab."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RunConfig:
    model_id: str
    framework: str
    workload: str
    batch_size: int
    seq_length: int
    warmup_steps: int
    steps: int
    precision: str
    device_type: str
    training_micro_steps: int = 0
    notes: str = ""


@dataclass
class StepMetrics:
    timestamp: str
    step: int
    batch_size: int
    precision: str
    device_type: str
    warmup: bool
    latency_ms: float
    throughput_items_per_sec: float
    host_input_time_ms: float
    compute_time_ms: float
    idle_time_ms: float
    compile_time_ms: Optional[float]
    memory_mb: Optional[float]
    notes: str = ""


@dataclass
class Recommendation:
    title: str
    symptom: str
    likely_root_cause: str
    evidence: Dict[str, float]
    confidence: float
    expected_impact: str
    action_steps: List[str]


@dataclass
class AnalysisSummary:
    run_id: str
    model_id: str
    framework: str
    workload: str
    device_type: str
    generated_at: str
    attribution: Dict[str, float]
    kpis: Dict[str, float]
    recommendations: List[Recommendation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
