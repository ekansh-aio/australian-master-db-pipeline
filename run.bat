@echo off
REM Legal Document Processing Pipeline - Quick Start Script (Windows)
REM This script helps you run the pipeline with minimal configuration

echo ======================================================================
echo Legal Document Processing Pipeline
echo ======================================================================
echo.

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: No .env file found!
    echo.
    echo Please create a .env file with your Azure credentials:
    echo   copy .env.example .env
    echo   REM Then edit .env with your credentials
    echo.
    exit /b 1
)

REM Check Python version
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://www.python.org
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python version: %PYTHON_VERSION%
echo.

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import dotenv" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Dependencies not installed!
    echo Installing dependencies...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        exit /b 1
    )
    echo.
)

echo [OK] Dependencies installed
echo.

REM Run the pipeline
echo Starting pipeline...
echo ======================================================================
echo.

python pipeline.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo [FAILED] Pipeline failed!
    echo ======================================================================
    echo.
    echo Check pipeline.log for details
    exit /b 1
) else (
    echo.
    echo ======================================================================
    echo [SUCCESS] Pipeline completed successfully!
    echo ======================================================================
    echo.
    echo Check output files:
    echo   - output\intermediate\all_chunks.json
    echo   - output\intermediate\top_k_chunks.json
    echo   - output\final\processing_stats.json
    echo   - pipeline.log
    echo.
)
