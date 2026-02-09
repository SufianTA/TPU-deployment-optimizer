param(
  [string]$ProjectId,
  [string]$Region = "us-central1",
  [string]$Service = "tpuopt-dashboard"
)

if (-not $ProjectId) {
  Write-Error "ProjectId is required. Example: .\deploy_cloudrun.ps1 -ProjectId my-gcp-project"
  exit 1
}

gcloud auth login

gcloud config set project $ProjectId

gcloud run deploy $Service `
  --source . `
  --region $Region `
  --allow-unauthenticated
