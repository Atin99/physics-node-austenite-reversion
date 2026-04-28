# PhysicsNODE - Local Server Launch Script

Write-Host ""
Write-Host "  PhysicsNODE - Austenite Reversion Kinetics" -ForegroundColor Cyan
Write-Host "  -------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

Set-Location $PSScriptRoot

# check python
try {
    $null = Get-Command python -ErrorAction Stop
} catch {
    Write-Host "  ERROR: python not found in PATH" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}

# check streamlit
$check = python -c "import streamlit" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
    python -m pip install streamlit
    Write-Host ""
}

Write-Host "  Starting server at http://localhost:8501" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host ""

python -m streamlit run src/streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
