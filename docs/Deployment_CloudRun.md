# Deploy to Cloud Run

This deploys the Streamlit dashboard. It uses the root `Dockerfile`, which runs Streamlit on port `8080` as required by Cloud Run.

## Prereqs
- A GCP project with billing enabled
- `gcloud` installed and authenticated

## Deploy
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud run deploy tpuopt-dashboard \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Notes
- If you want a CLI-only container, use `Dockerfile.cli` and build with:
```bash
docker build -f Dockerfile.cli -t tpuopt-cli .
```
