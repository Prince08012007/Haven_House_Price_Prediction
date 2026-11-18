@echo off
title House Price Prediction ML - Web GUI
cd /d "%~dp0"

echo ================================================================
echo       🏡 Ames House Price Prediction - Machine Learning GUI
echo ================================================================
echo Starting Streamlit Web Application...
echo.

python -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while launching Streamlit.
    echo Press any key to exit...
    pause >nul
)
