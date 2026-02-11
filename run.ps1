# Legal Document Processing Pipeline - Quick Start Script (PowerShell)
# This script helps you run the pipeline with minimal configuration

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Legal Document Processing Pipeline" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: No .env file found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please create a .env file with your Azure credentials:" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.example .env" -ForegroundColor White
    Write-Host "  # Then edit .env with your credentials" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Gray
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or later from https://www.python.org" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Gray
$depsInstalled = python -c "import dotenv" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Dependencies not installed!" -ForegroundColor Yellow
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    Write-Host ""
    
    pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

Write-Host "[OK] Dependencies installed" -ForegroundColor Green
Write-Host ""

# Run the pipeline
Write-Host "Starting pipeline..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

python pipeline.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Check output files:" -ForegroundColor Cyan
    Write-Host "  - output\intermediate\all_chunks.json" -ForegroundColor White
    Write-Host "  - output\intermediate\top_k_chunks.json" -ForegroundColor White
    Write-Host "  - output\final\processing_stats.json" -ForegroundColor White
    Write-Host "  - pipeline.log" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "[FAILED] Pipeline failed!" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check pipeline.log for details" -ForegroundColor Yellow
    exit 1
}
