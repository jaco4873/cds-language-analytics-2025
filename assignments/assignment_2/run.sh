#!/bin/bash

# Text Classification Benchmark Run Script
# This script runs the main orchestration script for text classification benchmarks

echo "
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                ┃
┃   🔍 Fake News Text Classification Benchmark Suite 🔍          ┃
┃                                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"

# Function to print section headers with emojis
print_section() {
    local emoji=$2
    if [ -z "$emoji" ]; then
        emoji="🔹"  # Default emoji
    fi
    
    echo ""
    echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
    echo "┃ $emoji  $1"
    echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
}

# Check for virtual environment
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    print_section "Environment Check" "🔍"
    echo "❌ No virtual environment detected in the project root."
    echo "💡 Please run setup.sh first to set up the environment."
    
    echo ""
    echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
    echo "┃                                                                ┃"
    echo "┃   🔴 🔴 🔴  SETUP REQUIRED  🔴 🔴 🔴                           ┃"
    echo "┃                                                                ┃"
    echo "┃   🚀 Would you like to run setup.sh now?                       ┃"
    echo "┃                                                                ┃"
    echo "┃   ✅ Press [ENTER] to run setup.sh                            ┃"
    echo "┃   ❌ Press any other key to exit                              ┃"
    echo "┃                                                                ┃"
    echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
    echo ""
    
    read -p "➤ " run_setup
    if [[ -z "$run_setup" || "$run_setup" == "y" || "$run_setup" == "Y" ]]; then
        echo "🚀 Running setup.sh..."
        bash setup.sh
        exit 0
    else
        echo "❌ Exiting. Please run setup.sh before running this script."
        exit 1
    fi
else
    print_section "Environment Check" "✅"
    echo "🌟 Virtual environment detected."
    
    # Activate the environment if not already activated
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -d ".venv" ]; then
            echo "🔌 Activating virtual environment..."
            source .venv/bin/activate
        elif [ -d "venv" ]; then
            echo "🔌 Activating virtual environment..."
            source venv/bin/activate
        fi
    else
        echo "✨ Virtual environment is already activated."
    fi
fi

# Check if fake_or_real_news.csv exists
print_section "Data Check" "📊"
if [ ! -f "data/fake_or_real_news.csv" ]; then
    echo "⚠️  Warning: fake_or_real_news.csv not found in data directory."
    echo "📝 Please place the Fake News dataset in data/ directory to proceed."
    echo "❌ Exiting. Cannot continue without dataset."
    exit 1
else
    echo "✅ Fake News dataset found."
fi

# Ensure required directories exist
print_section "Directory Setup" "📁"
mkdir -p data/vectorized models output results
echo "✅ Required directories have been created."

# Run the main script
print_section "Starting Analysis" "🚀"
echo "🔥 Running the complete text classification benchmark pipeline..."
echo "⏳ This may take several minutes depending on your hardware."
echo ""

# Execute main.py and wait for it to complete
uv run src/main.py
exit_code=$?

# Check if the command succeeded
if [ $exit_code -eq 0 ]; then
    print_section "Analysis Complete" "🎉"
    echo "🏆 The text classification benchmark has completed successfully! 🏆"
    echo ""
    echo "📊 Results can be found in the following directories:"
    echo "   📝 Classification reports: ./output/reports"
    echo "   💾 Trained models: ./output/models/"
    echo "   📈 Visualizations: ./output/figures/"
    echo ""
    echo "📋 To view the comparison results, check: ./output/reports/model_comparison.csv"
    
    # List visualization files
    echo ""
    echo "🖼️  Generated visualizations:"
    ls -1 output/figures/*.png | sed 's/^/   🔹 /'
else
    print_section "Analysis Failed" "❌"
    echo "❌ The analysis failed with error code $exit_code."
    echo "🔍 Please check the error messages above for more information."
    exit 1
fi

echo "
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                ┃
┃                        ✨ FINISHED ✨                          ┃
┃                                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"