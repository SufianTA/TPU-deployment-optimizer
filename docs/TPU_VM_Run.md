# Run on a TPU VM

This lab can run on a TPU VM if the environment provides a TPU runtime. The app detects TPU availability and labels results accordingly.

## Steps
1. Create a TPU VM in your GCP project.
2. SSH into the VM and clone the repo.
3. Install dependencies:
```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install .[streamlit,models]
```
4. Start the app:
```bash
streamlit run frontend/app.py --server.port 8080
```

## Notes
- TPU profiling artifacts are optional. If TensorBoard profiles are not available, the lab still produces metrics and recommendations.
- The app does not claim kernel‑level access; it uses profiler‑visible events when possible.
