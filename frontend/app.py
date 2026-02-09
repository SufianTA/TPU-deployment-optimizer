from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import os

from backend.analysis.charts import render_charts
from backend.analysis.deep_profile import parse_trace_events
from backend.analysis.xla_deep import parse_hlo_stats, parse_xla_compile_log
from backend.analysis.io import analyze_metrics, load_metrics_df
from backend.analysis.metrics import write_metrics_csv, write_summary_json
from backend.analysis.report import build_exec_summary, write_reports
from backend.analysis.scoring import build_summary
from backend.analysis.models import RunConfig
from backend.benchmarks.registry import list_models, run_benchmark


st.set_page_config(page_title="TPU Deployment Optimization Lab", layout="wide")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_CARDS = list_models()


st.title("TPU Deployment Optimization Lab")

st.markdown(
    "This lab runs a small benchmark, collects metrics, and generates a structured optimization report. "
    "It does not tune kernels or rewrite your model, but it does surface likely bottlenecks and "
    "actionable configuration levers."
)

with st.expander("What the tool actually does"):
    st.markdown(
        "- Runs a short, repeatable benchmark and captures latency and throughput metrics.\n"
        "- Estimates input vs compute vs idle vs compile time.\n"
        "- Produces an evidence-backed recommendation list with confidence and expected impact."
    )

with st.expander("Limitations"):
    st.markdown(
        "- This is not kernel-level profiling. It relies on end-to-end metrics and available traces.\n"
        "- TPU-specific signals are only available when running on TPU with proper profiling hooks.\n"
        "- Recommendations are heuristic and should be validated with full profiler data."
    )

# Sidebar workflow
st.sidebar.header("Workflow")

st.sidebar.markdown("**1) Select Model**")
model_ids = list(MODEL_CARDS.keys())
selected_model = st.sidebar.selectbox("Model", model_ids)
model_card = MODEL_CARDS[selected_model]

st.sidebar.markdown("**2) Configure Run**")
framework = model_card["framework"]
workload = st.sidebar.selectbox("Workload", ["inference", "training"])
batch_size = st.sidebar.slider("Batch size", 1, 256, 32, step=1)
seq_length = st.sidebar.slider("Seq length", 16, 512, 128, step=16)
warmup_steps = st.sidebar.slider("Warmup steps", 1, 10, 3)
steps = st.sidebar.slider("Benchmark steps", 5, 50, 15)
precision = st.sidebar.selectbox("Precision", ["fp32", "bf16", "fp16"])
training_micro_steps = st.sidebar.slider("Training micro-steps", 0, 5, 0)

st.sidebar.markdown("**3) Run Benchmark**")
run_button = st.sidebar.button("Run Benchmark")

st.sidebar.markdown("**4) Analyze & Optimize**")

with st.sidebar.expander("Explain what's happening"):
    st.markdown(
        "1. Instrument: select a model and config.\n"
        "2. Run: execute benchmark steps with warmup.\n"
        "3. Collect: write metrics and optional traces.\n"
        "4. Analyze: compute bottleneck attribution and recommendations."
    )


st.subheader("Model Card")
st.write(f"**{model_card['title']}**")
st.write(f"Input shape: {model_card['input_shape']}")
st.write(f"Expected throughput: {model_card['throughput_notes']}")
st.write(f"Typical bottlenecks: {model_card['bottlenecks']}")

st.subheader("Workflow Explanations")
with st.expander("1) Select Model"):
    st.markdown("Pick a preloaded model. The lab uses lightweight versions to keep runs fast.")
with st.expander("2) Configure Run"):
    st.markdown("Set batch size, sequence length, warmup steps, and precision for repeatable benchmarks.")
with st.expander("3) Run Benchmark"):
    st.markdown("Runs inference (and optional micro training steps) to generate metrics and timings.")
with st.expander("4) Analyze & Optimize"):
    st.markdown("Computes bottleneck attribution and generates evidence-backed recommendations.")


st.subheader("Upload Artifacts")
col1, col2, col3 = st.columns(3)
with col1:
    uploaded_metrics = st.file_uploader("Upload metrics.csv", type=["csv"], key="metrics")
with col2:
    uploaded_profile = st.file_uploader("Upload profile zip (optional)", type=["zip"], key="profile")
with col3:
    uploaded_compare = st.file_uploader("Upload metrics.csv (Run B)", type=["csv"], key="metrics_b")

model_upload = st.file_uploader("Upload model (TF SavedModel zip or ONNX) - experimental", type=["zip", "onnx"])
st.caption("Model upload is experimental. Metrics/profile upload is recommended for now.")

if uploaded_profile:
    st.info("Profile zip uploaded. It will be stored alongside this run if you analyze.")
if model_upload:
    st.warning("Model upload is experimental and not yet executed. Use metrics/profile upload for analysis.")


status_box = st.empty()


def _save_profile_zip(run_dir: Path, uploaded) -> None:
    if not uploaded:
        return
    profile_dir = run_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    data = uploaded.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(profile_dir)


def _run_and_analyze() -> None:
    config = RunConfig(
        model_id=selected_model,
        framework=framework,
        workload=workload,
        batch_size=batch_size,
        seq_length=seq_length,
        warmup_steps=warmup_steps,
        steps=steps,
        precision=precision,
        device_type="",
        training_micro_steps=training_micro_steps,
    )

    status_box.info("Running benchmark...")
    progress = st.progress(0)
    log_box = st.empty()

    def _on_step(step, total):
        progress.progress(min(1.0, step / total))
        log_box.text(f"Completed step {step}/{total}")

    force_sim = False
    if os.getenv("K_SERVICE") or os.getenv("TPUOPT_SIMULATE", "").lower() in {"1", "true", "yes"}:
        force_sim = True
        st.info("Running in Cloud Run or simulation mode. Benchmark will use synthetic timing.")

    result = run_benchmark(config, step_callback=_on_step, force_simulate=force_sim)

    run_dir = ARTIFACTS_DIR / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    write_metrics_csv(metrics_path, result.metrics)

    summary = build_summary(load_metrics_df(metrics_path), config, result.run_id)
    summary.notes.extend(result.notes)

    summary_path = run_dir / "summary.json"
    write_summary_json(summary_path, summary)

    charts_dir = run_dir / "charts"
    chart_outputs = render_charts(load_metrics_df(metrics_path), charts_dir)

    report_outputs = write_reports(run_dir, summary)

    _save_profile_zip(run_dir, uploaded_profile)

    st.session_state["run_id"] = result.run_id
    st.session_state["summary"] = summary
    st.session_state["metrics_path"] = str(metrics_path)
    st.session_state["charts_dir"] = str(charts_dir)
    st.session_state["report_outputs"] = report_outputs
    st.session_state["chart_outputs"] = chart_outputs

    status_box.success(f"Run complete. Artifacts saved under {run_dir}")


if run_button:
    _run_and_analyze()

if uploaded_metrics and not run_button:
    run_id = "uploaded"
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    metrics_path.write_bytes(uploaded_metrics.read())

    config = RunConfig(
        model_id=selected_model,
        framework=framework,
        workload=workload,
        batch_size=batch_size,
        seq_length=seq_length,
        warmup_steps=warmup_steps,
        steps=steps,
        precision=precision,
        device_type="uploaded",
    )

    summary = analyze_metrics(metrics_path, config, run_id)
    summary_path = run_dir / "summary.json"
    write_summary_json(summary_path, summary)
    charts_dir = run_dir / "charts"
    chart_outputs = render_charts(load_metrics_df(metrics_path), charts_dir)
    report_outputs = write_reports(run_dir, summary)

    _save_profile_zip(run_dir, uploaded_profile)

    st.session_state["run_id"] = run_id
    st.session_state["summary"] = summary
    st.session_state["metrics_path"] = str(metrics_path)
    st.session_state["charts_dir"] = str(charts_dir)
    st.session_state["report_outputs"] = report_outputs
    st.session_state["chart_outputs"] = chart_outputs


summary = st.session_state.get("summary")

if summary:
    kpis = summary.kpis
    st.subheader("Run Summary")
    if summary.device_type != "tpu":
        st.info(f"TPU not detected. Running on {summary.device_type.upper()} baseline.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("p50 latency (ms)", f"{kpis.get('latency_p50_ms', 0):.2f}")
    k2.metric("p90 latency (ms)", f"{kpis.get('latency_p90_ms', 0):.2f}")
    k3.metric("p99 latency (ms)", f"{kpis.get('latency_p99_ms', 0):.2f}")
    k4.metric("Throughput", f"{kpis.get('throughput_mean', 0):.2f} items/sec")

    tabs = st.tabs(["Overview", "Bottlenecks", "Recommendations", "Deep Analysis", "Raw Metrics", "Export"])

    with tabs[0]:
        st.markdown("**Copy-ready executive summary**")
        for bullet in build_exec_summary(summary):
            st.markdown(f"- {bullet}")
        st.markdown("**Attribution**")
        st.json(summary.attribution)

        if uploaded_compare:
            compare_df = pd.read_csv(uploaded_compare)
            base_df = load_metrics_df(Path(st.session_state.get("metrics_path")))
            st.markdown("**Comparison (Run A vs Run B)**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "metric": ["p50 latency", "p90 latency", "p99 latency", "throughput mean"],
                        "run_a": [
                            base_df["latency_ms"].median(),
                            base_df["latency_ms"].quantile(0.9),
                            base_df["latency_ms"].quantile(0.99),
                            base_df["throughput_items_per_sec"].mean(),
                        ],
                        "run_b": [
                            compare_df["latency_ms"].median(),
                            compare_df["latency_ms"].quantile(0.9),
                            compare_df["latency_ms"].quantile(0.99),
                            compare_df["throughput_items_per_sec"].mean(),
                        ],
                    }
                )
            )

    with tabs[1]:
        st.markdown("**Bottleneck attribution (%):**")
        st.json(summary.attribution)
        charts_dir = Path(st.session_state.get("charts_dir", ""))
        if charts_dir.exists():
            for chart in charts_dir.glob("*.html"):
                st.components.v1.html(chart.read_text(encoding="utf-8"), height=400, scrolling=True)

    with tabs[2]:
        for rec in summary.recommendations:
            st.markdown(f"### {rec.title}")
            st.markdown(f"**Symptom:** {rec.symptom}")
            st.markdown(f"**Root cause:** {rec.likely_root_cause}")
            st.markdown(f"**Evidence:** {rec.evidence}")
            st.markdown(f"**Confidence:** {rec.confidence}")
            st.markdown(f"**Expected impact:** {rec.expected_impact}")
            st.markdown("**Action steps:**")
            for step in rec.action_steps:
                st.markdown(f"- {step}")

    with tabs[3]:
        st.markdown("**Kernel & compiler signals (best effort)**")
        run_id = st.session_state.get("run_id")
        run_dir = ARTIFACTS_DIR / run_id
        profile_dir = run_dir / "profile"
        trace_path = profile_dir / "trace.json"
        if trace_path.exists():
            parsed = parse_trace_events(trace_path)
            st.markdown("Top ops by total time (us)")
            st.dataframe(parsed["top_ops"])
            st.markdown("Categories")
            st.dataframe(parsed["categories"])
        else:
            st.info("No trace.json found. Upload a profile zip to populate kernel signals.")

        hlo_path = profile_dir / "hlo.txt"
        if hlo_path.exists():
            st.markdown("HLO op counts (best effort)")
            st.dataframe(parse_hlo_stats(hlo_path))
        else:
            st.caption("HLO/MLIR dump not found. If available, add it to the profile zip.")

        xla_log = profile_dir / "xla_compile.log"
        if xla_log.exists():
            st.markdown("XLA compile log summary")
            st.json(parse_xla_compile_log(xla_log))
        else:
            st.caption("XLA compile log not found.")

    with tabs[4]:
        df = load_metrics_df(Path(st.session_state.get("metrics_path")))
        st.dataframe(df)

    with tabs[5]:
        run_id = st.session_state.get("run_id")
        run_dir = ARTIFACTS_DIR / run_id
        if run_dir.exists():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in run_dir.rglob("*"):
                    if file.is_file():
                        zf.write(file, arcname=file.relative_to(run_dir))
            st.download_button(
                "Download artifacts ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"{run_id}_artifacts.zip",
                mime="application/zip",
            )

    st.subheader("Engineer Details")
    run_id = st.session_state.get("run_id")
    run_dir = ARTIFACTS_DIR / run_id
    profile_dir = run_dir / "profile"
    st.json(
        {
            "metrics_path": st.session_state.get("metrics_path"),
            "charts_dir": st.session_state.get("charts_dir"),
            "report_outputs": st.session_state.get("report_outputs"),
            "profile_dir": str(profile_dir) if profile_dir.exists() else None,
        }
    )
else:
    st.info("Run a benchmark or upload metrics.csv to see analysis.")
