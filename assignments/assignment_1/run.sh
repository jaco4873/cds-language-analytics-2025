#!/bin/bash

echo "
╭──────────────────────────────────╮
│        ANALYSIS PIPELINE         │
╰──────────────────────────────────╯"

echo "Starting analysis..."

# Check for virtual environment
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "⚠️  Error: No virtual environment detected in the project root."
    echo "   Please run setup.sh first to set up the environment."
    read -p "   Would you like to run setup.sh now? (y/n): " run_setup
    if [[ $run_setup == "y" || $run_setup == "Y" ]]; then
        echo "🔧 Running setup.sh..."
        bash setup.sh
        exit 0
    else
        echo "🛑 Exiting. Please run setup.sh before running this script."
        exit 1
    fi
else
    echo "✅ Virtual environment detected."
fi

# Get command line arguments
LOG_LEVEL=${1:-INFO}  # Default to INFO if not provided

echo "🔍 Running with log level: $LOG_LEVEL"

# Run the main script with error handling
echo "⏳ Executing analysis..."
if uv run src/main.py --log-level "$LOG_LEVEL"; then
    echo "
╭──────────────────────────────────╮
│     ANALYSIS COMPLETED! 🎉       │
╰──────────────────────────────────╯"
else
    echo "
╭──────────────────────────────────╮
│     ANALYSIS FAILED! ❌          │
╰──────────────────────────────────╯"
    echo "Error code: $?. Please check the logs."
    exit 1
fi

echo "ℹ️  To run with different log levels, use: ./run.sh [DEBUG|INFO|WARNING|ERROR|CRITICAL]"
echo "   Example: ./run.sh DEBUG"