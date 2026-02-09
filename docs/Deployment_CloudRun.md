# Deploy to Cloud Run

This deploys the Streamlit dashboard using the root `Dockerfile`.

## Prereqs
- A GCP project with billing enabled
- `gcloud` installed and authenticated

## Deploy
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy tpuopt-dashboard \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Notes
- Cloud Run requires the container to listen on port `8080`.
- The app will run on CPU/GPU by default and label TPU detection in the UI.
