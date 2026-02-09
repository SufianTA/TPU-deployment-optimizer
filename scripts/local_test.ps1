param(
  [string]$VenvPath = ".venv"
)

python -m venv $VenvPath
& "$VenvPath\Scripts\activate"
python -m pip install -U pip
pip install .[streamlit,models]
pytest
