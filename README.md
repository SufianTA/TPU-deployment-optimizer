# TPU Deployment Optimization Lab

A compact, production‑quality lab that runs model benchmarks, captures metrics, and generates a structured optimization report. It is designed to be honest and practical: it does not tune kernels, but it does measure end‑to‑end behavior and surface likely bottlenecks with actionable levers.

## What this is
- A Streamlit web app with a four‑step workflow: select model, configure run, benchmark, analyze.
- A small analysis engine that computes bottleneck attribution and ranks recommendations with evidence.
- Artifact output under `artifacts/<run_id>/` including `metrics.csv`, `summary.json`, charts, and reports.

## What this is not
- It does not claim kernel‑level access.
- It does not replace full TPU profiler analysis.
- It does not auto‑tune model graphs.

## Quick start (local)
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install .[streamlit,models]

streamlit run frontend/app.py
```

## Cloud Run (deployed)
This repo is configured for Cloud Run via the root `Dockerfile`.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud run deploy tpuopt-dashboard \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## How the workflow works
1. **Select Model**: Choose a preloaded TF/JAX/PyTorch model.
2. **Configure Run**: Batch size, sequence length, warmup, steps, precision.
3. **Run Benchmark**: Inference benchmark (training micro‑step optional).
4. **Analyze & Optimize**: Attribution, recommendations, charts, and export.

## Upload support
- Upload a `metrics.csv` and optional profile zip to analyze existing runs.
- Model upload (TF SavedModel zip or ONNX) is experimental and not required.

## Artifacts
Each run produces:
- `artifacts/<run_id>/metrics.csv`
- `artifacts/<run_id>/summary.json`
- `artifacts/<run_id>/charts/` (HTML)
- `artifacts/<run_id>/recommendations.md`
- `artifacts/<run_id>/report.html`

## Notes / assumptions
- If TPU is not detected, the lab runs on CPU/GPU and labels results accordingly.
- Profiling artifacts are best‑effort; the app proceeds even if profiles are missing.
- Recommendations are heuristic and should be validated with deeper profiling.

## Repo layout
- `backend/analysis`: ingestion, scoring, reporting, charts
- `backend/benchmarks`: TF/JAX/PyTorch runners
- `frontend/app.py`: Streamlit app
- `artifacts/`: generated per run

## License
Apache‑2.0
