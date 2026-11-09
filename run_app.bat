@echo off
cd /d "%~dp0"
python -m streamlit run app.py --server.address localhost
if errorlevel 1 (
  echo Install first: python -m pip install ".[ui]"
  pause
)
