@echo off
title House Price Prediction ML - Desktop App
cd /d "%~dp0"

echo ================================================================
echo       🏡 Ames House Price Prediction - Desktop Application
echo ================================================================
echo Starting native desktop window...
echo.

python desktop_gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while launching desktop GUI.
    echo Press any key to exit...
    pause >nul
)
