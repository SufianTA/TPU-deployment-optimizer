# Example Commands

```bash
# local run
pip install .[streamlit,models]
streamlit run frontend/app.py

# build and run container locally
docker build -t tpuopt-lab .
docker run -p 8080:8080 tpuopt-lab
```

Sample outputs live under `sample_outputs/run_sample/`.
