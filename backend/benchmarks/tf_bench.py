from __future__ import annotations

import time
import uuid
from typing import List, Tuple

import numpy as np

from backend.analysis.models import RunConfig
from backend.analysis.utils import utc_now_iso
from backend.benchmarks.base import BenchResult, generate_image_batch, generate_text_batch, make_step_metrics


def _get_device_type_tf() -> str:
    try:
        import tensorflow as tf

        if tf.config.list_logical_devices("TPU"):
            return "tpu"
        if tf.config.list_logical_devices("GPU"):
            return "gpu"
    except Exception:
        return "cpu"
    return "cpu"


def _build_text_model_tf(vocab_size: int, seq_length: int, num_classes: int = 2):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(seq_length,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, 64)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def _load_model(model_id: str, seq_length: int):
    import tensorflow as tf

    if model_id == "tf_mobilenet_v2":
        return tf.keras.applications.MobileNetV2(weights=None, include_top=True)
    if model_id == "tf_text_classifier":
        return _build_text_model_tf(vocab_size=30522, seq_length=seq_length)
    raise ValueError(f"Unknown TF model_id: {model_id}")


def run_tf_benchmark(config: RunConfig, step_callback=None) -> BenchResult:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is not installed") from exc

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    notes: List[str] = []

    device_type = _get_device_type_tf()
    config.device_type = device_type

    if device_type == "tpu":
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            notes.append("TPU detected and initialized via TPUStrategy")
        except Exception:
            strategy = tf.distribute.get_strategy()
            notes.append("TPU not initialized; using default strategy")
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = _load_model(config.model_id, config.seq_length)
        optimizer = tf.keras.optimizers.Adam()

    metrics = []

    def _run_step(inputs):
        if config.training_micro_steps > 0:
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                loss = tf.reduce_mean(outputs)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss
        return model(inputs, training=False)

    total_steps = config.warmup_steps + config.steps
    for step in range(total_steps):
        warmup = step < config.warmup_steps

        host_start = time.perf_counter()
        if config.model_id == "tf_mobilenet_v2":
            batch = generate_image_batch(config.batch_size)
        else:
            batch = generate_text_batch(config.batch_size, config.seq_length)
        host_end = time.perf_counter()

        start = time.perf_counter()
        _run_step(batch)
        if device_type == "tpu":
            tf.experimental.async_wait()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        host_ms = (host_end - host_start) * 1000.0
        throughput = config.batch_size / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

        metrics.append(
            make_step_metrics(
                step=step,
                batch_size=config.batch_size,
                precision=config.precision,
                device_type=device_type,
                warmup=warmup,
                latency_ms=latency_ms,
                throughput=throughput,
                host_input_ms=host_ms,
                compile_ms=None,
                memory_mb=None,
                notes="",
            )
        )
        if step_callback:
            step_callback(step + 1, total_steps)

    return BenchResult(run_id=run_id, config=config, metrics=metrics, notes=notes)
