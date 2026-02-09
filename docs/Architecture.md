# Architecture

The lab is structured as a benchmark pipeline with a deterministic output directory per run.

```mermaid
graph TD
  A[Select Model + Config] --> B[Benchmark Runner]
  B --> C[metrics.csv]
  B --> D[summary.json]
  B --> E[profile/ (optional)]
  C --> F[Analysis Engine]
  F --> G[Attribution + Recommendations]
  G --> H[Charts + Report]
  H --> I[Export ZIP]
```

## Modules
- `backend/benchmarks`: framework runners and model registry
- `backend/analysis`: scoring, charts, reports
- `frontend/app.py`: Streamlit workflow UI
