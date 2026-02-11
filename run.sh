#!/bin/bash

# Legal Document Processing Pipeline - Quick Start Script
# This script helps you run the pipeline with minimal configuration

echo "======================================================================"
echo "Legal Document Processing Pipeline"
echo "======================================================================"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo ""
    echo "Please create a .env file with your Azure credentials:"
    echo "  cp .env.example .env"
    echo "  # Then edit .env with your credentials"
    echo ""
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if dependencies are installed
if ! python3 -c "import dotenv" 2>/dev/null; then
    echo ""
    echo "⚠️  Dependencies not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "✓ Dependencies installed"
echo ""

# Run the pipeline
echo "Starting pipeline..."
echo "======================================================================"
echo ""

python3 pipeline.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Pipeline completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Check output files:"
    echo "  - output/intermediate/all_chunks.json"
    echo "  - output/intermediate/top_k_chunks.json"
    echo "  - output/final/processing_stats.json"
    echo "  - pipeline.log"
else
    echo ""
    echo "======================================================================"
    echo "✗ Pipeline failed!"
    echo "======================================================================"
    echo ""
    echo "Check pipeline.log for details"
    exit 1
fi
