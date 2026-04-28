@echo off
title PhysicsNODE - Local Server
echo.
echo  PhysicsNODE - Austenite Reversion Kinetics
echo  -------------------------------------------
echo.

cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: python not found in PATH
    pause
    exit /b 1
)

python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing dependencies...
    pip install -r requirements.txt
    pip install streamlit
    echo.
)

echo  Starting server at http://localhost:8501
echo  Press Ctrl+C to stop
echo.
streamlit run src/streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
