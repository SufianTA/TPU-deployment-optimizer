# Recommendations

1. **Input pipeline is host-bound**
   Symptom: High host input time relative to total latency
   Likely root cause: Input preprocessing or IO is not keeping up with device execution
   Evidence: {'host_ratio': 0.31}
   Confidence: 0.84
   Expected impact: Medium to high (10–35%)
   Action steps:
   - Increase input pipeline parallelism and prefetch depth
   - Enable dataset caching or pre-shard input files
   - Move heavy preprocessing out of the hot path
