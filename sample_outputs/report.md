# TPU Deployment Optimization Report

Model: `demo-model`
Workload: `inference`
Generated: `2026-02-09T16:00:00+00:00`

## Notes
- Sample data generated for demo purposes.

## Metrics Summary
- `step_time_ms`: {'mean': 121.6, 'p50': 120, 'p90': 128, 'p95': 129, 'p99': 129.8}
- `host_input_time_ms`: {'mean': 49.4, 'p50': 50, 'p90': 54, 'p95': 54.5, 'p99': 54.9}
- `tpu_compute_time_ms`: {'mean': 60.6, 'p50': 60, 'p90': 63.2, 'p95': 63.6, 'p99': 63.9}
- `idle_time_ms`: {'mean': 11.6, 'p50': 10, 'p90': 15.6, 'p95': 16.3, 'p99': 16.9}
- `tokens_per_sec`: {'mean': 5100, 'p50': 5100, 'p90': 5260, 'p95': 5280, 'p99': 5296}
- `batch_size`: {'mean': 48, 'p50': 48, 'p90': 48, 'p95': 48, 'p99': 48}
- `hbm_utilization`: {'mean': 0.8, 'p50': 0.8, 'p90': 0.82, 'p95': 0.82, 'p99': 0.82}

## Findings & Recommendations
1. **High host input time relative to step time**
   Likely root cause: Input pipeline bottleneck (host-bound preprocessing or IO)
   What to check: host_input_time_ms / step_time_ms
   Remediation: Increase input pipeline parallelism, enable dataset caching, and prefetch. Check CPU utilization and data source throughput.
   Evidence: {'host_ratio': 0.406}

2. **Batch size appears small for TPU throughput**
   Likely root cause: Conservative batch sizing or memory constraints
   What to check: batch_size (mean)
   Remediation: Gradually increase batch size while monitoring HBM usage and step time. Use gradient accumulation for training if needed.
   Evidence: {'batch_size_mean': 48}

## Inputs
- Profile dir: `./sample_data`
- Metrics CSV: `./sample_data/metrics.csv`
- XLA log: `./sample_data/xla_compile.log`
- Trace file: `./sample_data/trace.json`
- GCP Monitoring: `False`

## Notes / assumptions
- Recommendations are inferred from available metrics and may be incomplete if profiling artifacts are missing. Validate with full TensorBoard profiling.
