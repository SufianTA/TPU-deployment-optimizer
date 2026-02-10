from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.analysis.charts import render_charts
from backend.analysis.deep_profile import parse_trace_events
from backend.analysis.io import analyze_metrics, load_metrics_df
from backend.analysis.metrics import write_metrics_csv, write_summary_json
from backend.analysis.report import build_exec_summary, write_reports
from backend.analysis.scoring import build_summary
from backend.analysis.models import RunConfig
from backend.analysis.xla_deep import parse_hlo_stats, parse_xla_compile_log
from backend.benchmarks.registry import list_models, run_benchmark


st.set_page_config(page_title="TPU Deployment Optimization Lab", layout="wide")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_CARDS = list_models()


def _style() -> None:
    st.markdown(
        """
<style>
  .hero {
    background: linear-gradient(135deg, #0b1220 0%, #1e293b 50%, #0f4c81 100%);
    color: #f8fafc;
    padding: 24px 28px;
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(2, 6, 23, 0.35);
    margin-bottom: 16px;
  }
  .hero-title { font-size: 32px; font-weight: 700; letter-spacing: 0.2px; }
  .hero-sub { font-size: 14px; opacity: 0.85; margin-top: 6px; }
  .kpi-card {
    background: #0f172a;
    color: #e2e8f0;
    padding: 14px 16px;
    border-radius: 12px;
    box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.2);
    min-height: 86px;
  }
  .kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; }
  .kpi-value { font-size: 22px; font-weight: 700; margin-top: 6px; }
  .section-title { font-size: 18px; font-weight: 600; margin: 10px 0 6px; }
  .panel {
    border: 1px solid #e2e8f0;
    background: #ffffff;
    padding: 14px 16px;
    border-radius: 12px;
  }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
  .badge-high { background: #dcfce7; color: #166534; }
  .badge-med { background: #fef3c7; color: #92400e; }
  .badge-low { background: #fee2e2; color: #991b1b; }
  .muted { color: #64748b; font-size: 12px; }
</style>
""",
        unsafe_allow_html=True,
    )


_style()

st.markdown(
    """
<div class="hero">
  <div class="hero-title">TPU Deployment Optimization Lab</div>
  <div class="hero-sub">Run a benchmark → collect metrics → diagnose bottlenecks → recommend config levers.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "This tool runs a short, repeatable benchmark and derives utilization proxies from end‑to‑end metrics. "
    "It does not tune kernels or rewrite models; it surfaces likely bottlenecks and configuration levers."
)

with st.expander("Limitations"):
    st.markdown(
        "- Metrics are end‑to‑end and best‑effort; kernel‑level signals require profiler artifacts.\n"
        "- TPU‑specific behavior only appears when running on TPU.\n"
        "- Recommendations are heuristic; validate with full profiling."
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


def _list_runs() -> List[str]:
    if not ARTIFACTS_DIR.exists():
        return []
    runs = [p.name for p in ARTIFACTS_DIR.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    return sorted(runs)


run_history = _list_runs()
if run_history:
    st.sidebar.markdown("**Run ID**")
    selected_run = st.sidebar.selectbox("Select Run", run_history, index=len(run_history) - 1)
else:
    selected_run = None

with st.sidebar.expander("Explain what's happening"):
    st.markdown(
        "1. Instrument: select a model and config.\n"
        "2. Run: execute benchmark steps with warmup.\n"
        "3. Collect: write metrics and optional traces.\n"
        "4. Analyze: compute bottleneck attribution and recommendations."
    )


st.markdown("<div class='section-title'>Model Card</div>", unsafe_allow_html=True)
st.write(f"**{model_card['title']}**")
st.write(f"Input shape: {model_card['input_shape']}")
st.write(f"Expected throughput: {model_card['throughput_notes']}")
st.write(f"Typical bottlenecks: {model_card['bottlenecks']}")


st.markdown("<div class='section-title'>Upload Artifacts</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    uploaded_metrics = st.file_uploader("Upload metrics.csv", type=["csv"], key="metrics")
with col2:
    uploaded_profile = st.file_uploader("Upload profile zip (optional)", type=["zip"], key="profile")
with col3:
    uploaded_compare = st.file_uploader("Upload metrics.csv (Run B)", type=["csv"], key="metrics_b")

model_upload = st.file_uploader("Upload model (TF SavedModel zip or ONNX) - experimental", type=["zip", "onnx"])
st.caption("Model upload is experimental. Metrics/profile upload is recommended for now.")
if model_upload:
    st.warning("Model upload is experimental and not executed. Use metrics/profile upload for analysis.")


status_box = st.empty()


def _save_profile_zip(run_dir: Path, uploaded) -> None:
    if not uploaded:
        return
    profile_dir = run_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    data = uploaded.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(profile_dir)


def _compute_kpis(df: pd.DataFrame, attribution: Dict[str, float]) -> Dict[str, float | str]:
    p50 = float(df["latency_ms"].median())
    p90 = float(df["latency_ms"].quantile(0.9))
    p99 = float(df["latency_ms"].quantile(0.99))
    throughput = float(df["throughput_items_per_sec"].mean())
    util_proxy = 1.0 - float(attribution.get("idle_bound", 0.0))
    dominant = max(attribution, key=attribution.get) if attribution else "n/a"
    dominant = dominant.replace("_bound", "").title()
    return {
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "throughput": throughput,
        "util": util_proxy * 100.0,
        "dominant": dominant,
    }


def _landing_preview() -> None:
    sample_path = Path("sample_outputs/run_sample/metrics.csv")
    if not sample_path.exists():
        return
    df = pd.read_csv(sample_path)
    st.markdown("<div class='section-title'>Landing Preview</div>", unsafe_allow_html=True)
    st.caption("Preview uses sample data. Run a benchmark for live results.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(px.line(df, x="step", y="latency_ms", title="Latency (ms)"), use_container_width=True)
    with c2:
        st.plotly_chart(
            px.line(df, x="step", y="throughput_items_per_sec", title="Throughput"),
            use_container_width=True,
        )
    with c3:
        util = 1 - (df["idle_time_ms"] / df["latency_ms"]).fillna(0)
        st.plotly_chart(px.line(x=df["step"], y=util, title="Utilization Proxy"), use_container_width=True)


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
    st.toast("Initializing benchmark run…", icon="⚙️")

    def _on_step(step, total):
        progress.progress(min(1.0, step / total))
        log_box.text(f"Step {step}/{total} complete")
        if step == 1:
            st.toast("Warmup complete, collecting metrics…", icon="📈")

    force_sim = False
    if os.getenv("K_SERVICE") or os.getenv("TPUOPT_SIMULATE", "").lower() in {"1", "true", "yes"}:
        force_sim = True
        st.info("Running in Cloud Run or simulation mode. Benchmark uses synthetic timing.")

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
    st.toast("Analysis complete. Charts and report are ready.", icon="✅")


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

# Load selected run if no active session
if selected_run and "summary" not in st.session_state:
    run_dir = ARTIFACTS_DIR / selected_run
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    if summary_path.exists() and metrics_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        st.session_state["run_id"] = selected_run
        st.session_state["summary"] = summary
        st.session_state["metrics_path"] = str(metrics_path)
        st.session_state["charts_dir"] = str(run_dir / "charts")

summary = st.session_state.get("summary")

if not summary:
    _landing_preview()
    st.warning("No metrics available. Run a benchmark or upload metrics.csv to continue.")
    st.stop()

# Summary may be dict (from disk) or model instance
if hasattr(summary, "kpis"):
    summary_data = summary
else:
    summary_data = summary

# Load metrics
metrics_path = Path(st.session_state.get("metrics_path"))
if not metrics_path.exists():
    st.error("Metrics file not found. Run a benchmark to generate metrics.")
    st.stop()

metrics_df = load_metrics_df(metrics_path)

# Attribution
if hasattr(summary_data, "attribution"):
    attribution = summary_data.attribution
else:
    attribution = summary_data.get("attribution", {})

kpis = _compute_kpis(metrics_df, attribution)

# KPI row
st.markdown("<div class='section-title'>Run KPIs</div>", unsafe_allow_html=True)
cols = st.columns(4)
cols[0].markdown(
    f"""<div class="kpi-card"><div class="kpi-label">Throughput</div>
    <div class="kpi-value">{kpis['throughput']:.2f} items/s</div></div>""",
    unsafe_allow_html=True,
)
cols[1].markdown(
    f"""<div class="kpi-card"><div class="kpi-label">Avg Latency</div>
    <div class="kpi-value">{metrics_df['latency_ms'].mean():.2f} ms</div></div>""",
    unsafe_allow_html=True,
)
cols[2].markdown(
    f"""<div class="kpi-card"><div class="kpi-label">Utilization Proxy</div>
    <div class="kpi-value">{kpis['util']:.1f}%</div></div>""",
    unsafe_allow_html=True,
)
cols[3].markdown(
    f"""<div class="kpi-card"><div class="kpi-label">Dominant Bottleneck</div>
    <div class="kpi-value">{kpis['dominant']}</div></div>""",
    unsafe_allow_html=True,
)

st.markdown("---")

# Tabs
st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
tabs = st.tabs(["Overview", "Bottlenecks", "Recommendations", "Compare Runs", "Export"])

with tabs[0]:
    st.markdown("**Run Timeline**")
    st.plotly_chart(
        px.line(metrics_df, x="step", y="throughput_items_per_sec", title="Throughput over steps"),
        use_container_width=True,
    )

    st.markdown("**Breakdown (stacked)**")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="input", x=metrics_df["step"], y=metrics_df["host_input_time_ms"]))
    fig.add_trace(go.Bar(name="compute", x=metrics_df["step"], y=metrics_df["compute_time_ms"]))
    fig.add_trace(go.Bar(name="idle", x=metrics_df["step"], y=metrics_df["idle_time_ms"]))
    if "compile_time_ms" in metrics_df.columns:
        fig.add_trace(go.Bar(name="compile", x=metrics_df["step"], y=metrics_df["compile_time_ms"].fillna(0)))
    fig.update_layout(barmode="stack", title="Latency breakdown (ms)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Latency distribution**")
    hist = px.histogram(metrics_df, x="latency_ms", nbins=30, title="Latency distribution")
    st.plotly_chart(hist, use_container_width=True)

    st.markdown("**Latency percentiles**")
    st.write({"p50": kpis["p50"], "p90": kpis["p90"], "p99": kpis["p99"]})

with tabs[1]:
    st.markdown("**Bottleneck attribution (%)**")
    if attribution:
        fig = px.pie(
            names=list(attribution.keys()),
            values=[v * 100 for v in attribution.values()],
            hole=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Attribution not available. Check metrics input.")

    st.markdown("**How we infer bottlenecks**")
    st.markdown(
        "- Input/compute/idle/compile ratios derived from per‑step timings.\n"
        "- Dominant ratio determines primary bottleneck.\n"
        "- Confidence increases with stable metrics and available traces."
    )

    st.markdown("**What data we used**")
    missing = []
    for col in ["latency_ms", "host_input_time_ms", "compute_time_ms", "idle_time_ms"]:
        if col not in metrics_df.columns:
            missing.append(col)
    if missing:
        st.warning(f"Missing metrics: {', '.join(missing)}")
    else:
        st.success("All required metrics present")

    st.markdown("**Evidence table**")
    latency_mean = metrics_df["latency_ms"].mean()
    evidence = {
        "input_ratio": float(metrics_df["host_input_time_ms"].mean() / latency_mean),
        "compute_ratio": float(metrics_df["compute_time_ms"].mean() / latency_mean),
        "idle_ratio": float(metrics_df["idle_time_ms"].mean() / latency_mean),
        "compile_spike_score": float((metrics_df["compile_time_ms"].fillna(0) > 0).mean())
        if "compile_time_ms" in metrics_df.columns
        else 0.0,
    }
    st.dataframe(pd.DataFrame([evidence]))

with tabs[2]:
    recs = summary_data.recommendations if hasattr(summary_data, "recommendations") else summary_data.get("recommendations", [])
    if not recs:
        st.warning("No recommendations available. Collect more metrics or profiles.")
    for rec in recs:
        confidence = rec.confidence if hasattr(rec, "confidence") else rec.get("confidence", 0.0)
        badge = "badge-low"
        label = "Low"
        if confidence >= 0.75:
            badge = "badge-high"
            label = "High"
        elif confidence >= 0.5:
            badge = "badge-med"
            label = "Medium"

        title = rec.title if hasattr(rec, "title") else rec.get("title", "Recommendation")
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(f"**{title}** <span class='badge {badge}'>Confidence: {label}</span>", unsafe_allow_html=True)
        symptom = rec.symptom if hasattr(rec, "symptom") else rec.get("symptom", "")
        cause = rec.likely_root_cause if hasattr(rec, "likely_root_cause") else rec.get("likely_root_cause", "")
        st.markdown(f"**Symptom:** {symptom}")
        st.markdown(f"**Root cause:** {cause}")
        evidence = rec.evidence if hasattr(rec, "evidence") else rec.get("evidence", {})
        st.markdown("**Evidence**")
        st.write(evidence)
        impact = rec.expected_impact if hasattr(rec, "expected_impact") else rec.get("expected_impact", "")
        st.markdown(f"**Expected impact:** {impact}")
        actions = rec.action_steps if hasattr(rec, "action_steps") else rec.get("action_steps", [])
        with st.expander("Details"):
            for step in actions:
                st.markdown(f"- {step}")
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    runs = _list_runs()
    if len(runs) < 2:
        st.info("Run history is limited. Execute another run to enable comparisons.")
    else:
        run_a = st.selectbox("Run A", runs, index=max(0, len(runs) - 2))
        run_b = st.selectbox("Run B", runs, index=len(runs) - 1)

        def _load_run(run_id: str) -> Tuple[pd.DataFrame, Dict]:
            run_dir = ARTIFACTS_DIR / run_id
            df = load_metrics_df(run_dir / "metrics.csv")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            return df, summary

        df_a, sum_a = _load_run(run_a)
        df_b, sum_b = _load_run(run_b)

        attr_a = sum_a.get("attribution", {})
        attr_b = sum_b.get("attribution", {})
        kpi_a = _compute_kpis(df_a, attr_a)
        kpi_b = _compute_kpis(df_b, attr_b)

        st.markdown("**Side‑by‑side KPIs**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Run A: {run_a}**")
            st.write(kpi_a)
        with c2:
            st.markdown(f"**Run B: {run_b}**")
            st.write(kpi_b)

        st.markdown("**Overlay charts**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_a["step"], y=df_a["throughput_items_per_sec"], name=f"{run_a} throughput"))
        fig.add_trace(go.Scatter(x=df_b["step"], y=df_b["throughput_items_per_sec"], name=f"{run_b} throughput"))
        fig.update_layout(title="Throughput comparison")
        st.plotly_chart(fig, use_container_width=True)

        util_a = 1 - (df_a["idle_time_ms"] / df_a["latency_ms"]).fillna(0)
        util_b = 1 - (df_b["idle_time_ms"] / df_b["latency_ms"]).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_a["step"], y=util_a, name=f"{run_a} utilization"))
        fig.add_trace(go.Scatter(x=df_b["step"], y=util_b, name=f"{run_b} utilization"))
        fig.update_layout(title="Utilization proxy comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**What changed**")
        st.write(
            {
                "Run A": {
                    "batch_size_mean": float(df_a["batch_size"].mean()) if "batch_size" in df_a else "n/a",
                    "precision": df_a["precision"].iloc[0] if "precision" in df_a else "n/a",
                    "device": sum_a.get("device_type", "n/a"),
                },
                "Run B": {
                    "batch_size_mean": float(df_b["batch_size"].mean()) if "batch_size" in df_b else "n/a",
                    "precision": df_b["precision"].iloc[0] if "precision" in df_b else "n/a",
                    "device": sum_b.get("device_type", "n/a"),
                },
            }
        )

with tabs[4]:
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
    else:
        st.warning("No run artifacts available.")

    st.markdown("**Copy Executive Summary**")
    exec_summary: List[str] = []
    if hasattr(summary_data, "recommendations"):
        exec_summary = build_exec_summary(summary_data)
    else:
        kpi_dict = summary_data.get("kpis", {}) if isinstance(summary_data, dict) else {}
        if kpi_dict.get("throughput_mean"):
            exec_summary.append(f"Mean throughput: {kpi_dict['throughput_mean']:.2f} items/sec")
        if kpi_dict.get("latency_p50_ms"):
            exec_summary.append(f"p50 latency: {kpi_dict['latency_p50_ms']:.2f} ms")
        recs = summary_data.get("recommendations", []) if isinstance(summary_data, dict) else []
        if recs:
            exec_summary.append(f"Top recommendation: {recs[0].get('title', 'Review recommendations')}")
    if not exec_summary:
        exec_summary = ["No summary available yet."]
    st.text_area("Executive Summary", value="\n".join(f"- {b}" for b in exec_summary), height=140)
