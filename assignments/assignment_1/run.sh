#!/bin/bash
# Run the main script

echo "Starting analysis..."

# Check for virtual environment
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "Error: No virtual environment detected in the project root."
    echo "Please run setup.sh first to set up the environment."
    read -p "Would you like to run setup.sh now? (y/n): " run_setup
    if [[ $run_setup == "y" || $run_setup == "Y" ]]; then
        echo "Running setup.sh..."
        bash setup.sh
        if [ $? -ne 0 ]; then
            echo "Setup failed. Please check the errors and try again."
            exit 1
        fi
    else
        echo "Exiting. Please run setup.sh before running this script."
        exit 1
    fi
else
    echo "Virtual environment detected."
fi

# Run the main script with error handling
if uv run src/main.py; then
    echo "Analysis completed successfully!"
else
    echo "Analysis failed with error code $?. Please check the logs."
    exit 1
fi