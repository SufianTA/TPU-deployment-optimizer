# Recommendations

1. High host input time relative to step time
   Root cause: Input pipeline bottleneck (host-bound preprocessing or IO)
   Check: host_input_time_ms / step_time_ms
   Remediation: Increase input pipeline parallelism, enable dataset caching, and prefetch. Check CPU utilization and data source throughput.
   Evidence: {'host_ratio': 0.406}

2. Batch size appears small for TPU throughput
   Root cause: Conservative batch sizing or memory constraints
   Check: batch_size (mean)
   Remediation: Gradually increase batch size while monitoring HBM usage and step time. Use gradient accumulation for training if needed.
   Evidence: {'batch_size_mean': 48}
