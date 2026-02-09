# TPU Optimization Cheat Sheet

This is a practical, non-exhaustive list of knobs that often matter for TPU deployment performance. Use it alongside profiler data.

## Batch sizing
- Increase batch size until HBM utilization approaches the safe limit for your model.
- For training, consider gradient accumulation if global batch size needs to stay fixed.
- Track throughput (`tokens_per_sec`) and step time as you scale.

## Shapes and compilation
- Favor static or bucketed shapes to avoid recompilations.
- Pad/pack sequences to fixed shapes where feasible.
- Cache compiled executables across replicas if your stack supports it.

## Input pipeline
- Use dataset caching for repeated epochs.
- Increase `num_parallel_calls` and prefetch depth.
- Keep preprocessing lightweight or move it upstream to data preparation.

## XLA / compiler
- Reduce dynamic control flow in the critical path.
- Use fused operations where available.
- Keep model function signatures stable across steps.

## Memory (HBM)
- Use bfloat16 or mixed precision where safe.
- Reduce activation size or checkpoint activations.
- Consider parameter sharding to reduce per-core memory usage.

## Parallelism strategy
- For training, tune data/model/pipeline parallelism to balance communication and compute.
- Watch all-reduce and cross-replica overhead in profiler traces.

## Serving tradeoffs
- Lower latency often competes with throughput. Decide the target SLA first.
- Micro-batching can improve throughput if you can tolerate latency.
