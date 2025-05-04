#!/bin/bash

# IMDb Sentiment Analysis Run Script
echo "
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                ┃
┃   🎭 IMDb Sentiment Analysis 🎭                                ┃
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

# Display current settings
print_section "Configuration" "⚙️"
echo "📋 Current settings:"
python -c "
from src.settings import settings
print(f'   • Sample Size: {settings.SAMPLE_SIZE} reviews')
print(f'   • Model Type: {settings.DEFAULT_MODEL}')
print('')
print('   🤖 Transformer (DistilBERT) Settings:')
print(f'   • Training Epochs: {settings.NUM_EPOCHS}')
print(f'   • Batch Size: {settings.BATCH_SIZE}')
print(f'   • Model Architecture: {settings.TRANSFORMER_MODEL}')
print('')
print('   📊 Shared Settings:')
print(f'   • Validation Split: {settings.VALIDATION_SPLIT}')
"
if [ $? -ne 0 ]; then
    print_section "Process Failed" "❌"
    echo "❌ Failed to load settings"
    echo "🔍 Please check if the settings are properly configured."
    exit 1
fi
echo ""

# Run the model
print_section "Model Execution" "🧠"
echo "🔥 Running IMDb sentiment analysis..."
echo "⏳ This may take some time for data download and model training."

uv run -m src.main run
if [ $? -ne 0 ]; then
    print_section "Process Failed" "❌"
    echo "❌ Model execution failed with error code $?."
    echo "🔍 Please check the error messages above for more information."
    exit 1
fi

# Check if model execution succeeded
if [ $? -eq 0 ]; then
    print_section "Process Complete" "🎉"
    echo "🏆 The IMDb sentiment analysis has been completed successfully!"
    echo ""
    echo "📊 You can find the results in the output directory:"
    echo "   💾 Model files in output/model"
    echo "   📊 Visualizations in output/figures:"
    echo "      - metrics_comparison.png: Baseline vs. Transformer model comparison"
    echo "      - sentiment_distribution.png: Distribution of positive and negative reviews"
    echo "      - review_length_performance.png: Performance by review length"
    echo ""
else
    print_section "Process Failed" "❌"
    echo "❌ The model execution failed with error code $?."
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
