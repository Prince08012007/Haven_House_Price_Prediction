@echo off
cd /d "%~dp0"
if not exist .venv (
  where py >nul 2>nul
  if %errorlevel%==0 (py -m venv .venv) else (python -m venv .venv)
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist model\model.joblib (
  echo Training the house price model for first run...
  python train.py
  if errorlevel 1 pause & exit /b 1
)
start "Haven House Price Prediction" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
