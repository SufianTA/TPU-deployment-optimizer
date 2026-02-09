from __future__ import annotations

from pathlib import Path

import streamlit as st

from tpuopt.analyzer import analyze_profile
from tpuopt.models import Workload
from tpuopt.report import render_markdown

st.set_page_config(page_title="TPU Deployment Optimizer", layout="wide")

st.title("TPU Deployment Utilization & Optimization Toolkit")

uploaded_dir = st.file_uploader(
    "Upload a metrics.csv or a trace directory zip", type=["csv", "zip"]
)

profile_dir = st.text_input("Or enter a profile directory", value="./sample_data")
model_name = st.text_input("Model name", value="demo-model")
workload = st.selectbox("Workload", options=["inference", "training"], index=0)

if st.button("Analyze"):
    out_dir = Path("./streamlit_outputs")
    summary = analyze_profile(Path(profile_dir), model_name, Workload(workload), out_dir)

    st.subheader("Recommendations")
    st.markdown(render_markdown(summary))

    charts_dir = out_dir / "charts"
    if charts_dir.exists():
        for chart in charts_dir.glob("*.html"):
            st.components.v1.html(chart.read_text(encoding="utf-8"), height=450, scrolling=True)

st.caption("This dashboard expects local profile directories; uploads are not yet wired.")
