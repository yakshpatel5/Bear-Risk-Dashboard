@echo off
:: ============================================================
:: start_dashboard.bat — Bear Risk Dashboard Launcher (Windows)
:: ============================================================
:: Usage: Double-click OR run from Command Prompt
:: Stop:  Ctrl+C in the terminal window
:: ============================================================

setlocal enabledelayedexpansion
title Bear Risk Dashboard
cd /d "%~dp0"

echo.
echo ============================================================
echo    AI BEAR RISK DASHBOARD  v1.0
echo ============================================================
echo.

:: ── Step 1: Virtual environment ─────────────────────────────
echo [1/5] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    echo   OK: venv activated
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo   OK: .venv activated
) else (
    echo   WARNING: No venv found -- using system Python
    echo   Fix: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
)

:: ── Step 2: FRED API key ────────────────────────────────────
echo.
echo [2/5] Checking environment...
if "%FRED_API_KEY%"=="" (
    if exist ".env" (
        for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
            if /i "%%a"=="FRED_API_KEY" set FRED_API_KEY=%%b
        )
    )
)
if "%FRED_API_KEY%"=="" (
    echo   WARNING: FRED_API_KEY not set!
    echo   Get a free key at: https://fred.stlouisfed.org/
    echo   Add to .env file: FRED_API_KEY=your_key_here
    set /p CONT="  Continue with cached data only? [y/N]: "
    if /i not "!CONT!"=="y" (
        echo Exiting.
        exit /b 1
    )
) else (
    echo   OK: FRED_API_KEY is set
)

:: ── Step 3: Health check ────────────────────────────────────
echo.
echo [3/5] Running startup health check...
python health_check_startup.py
echo.

:: ── Step 4: Start scheduler in background ───────────────────
echo [4/5] Starting background scheduler...
if not exist ".cache" mkdir .cache
start /B python scheduler.py > .cache\scheduler_stdout.log 2>&1
timeout /t 2 /nobreak > nul
echo   OK: Scheduler started
echo   Log: .cache\scheduler.log

:: ── Step 5: Alert check ─────────────────────────────────────
echo.
echo [5/5] Running alert check...
python personal_alerts.py
echo.

:: ── Launch Streamlit ────────────────────────────────────────
echo Launching at http://localhost:8501
echo Press Ctrl+C to stop.
echo.

:: Open browser after 3 second delay
start "" cmd /c "timeout /t 3 /nobreak > nul && start http://localhost:8501"

:: Run Streamlit (foreground)
streamlit run app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --theme.base dark ^
    --theme.backgroundColor "#0a0e1a" ^
    --theme.secondaryBackgroundColor "#0f1525" ^
    --theme.textColor "#e2e8f4"

:: Cleanup
echo.
echo Stopping scheduler...
taskkill /f /im python.exe /fi "WINDOWTITLE eq scheduler*" > nul 2>&1
echo Done.
endlocal
