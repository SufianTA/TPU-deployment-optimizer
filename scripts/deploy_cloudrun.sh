#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-}"
REGION="${2:-us-central1}"
SERVICE="${3:-tpuopt-dashboard}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Usage: ./deploy_cloudrun.sh <project-id> [region] [service]" >&2
  exit 1
fi

gcloud auth login

gcloud config set project "$PROJECT_ID"

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated
