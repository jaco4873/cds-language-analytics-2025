#!/bin/bash

# N-gram Language Model Run Script
echo "
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                ┃
┃   🔮 N-gram Language Model Generator 🔮                        ┃
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

# Check for training data
print_section "Data Check" "📊"
if [ -z "$(ls -A data/gutenberg/*.txt 2>/dev/null)" ]; then
    echo "⚠️  Warning: No text files found in data/gutenberg directory."
    echo "📝 Please add some .txt files to data/gutenberg before proceeding."
    echo "❌ Exiting. Cannot continue without training data."
    exit 1
else
    echo "✅ Training data found in data/gutenberg."
fi

# Ensure required directories exist
print_section "Directory Setup" "📁"
mkdir -p data/gutenberg models output
echo "✅ Required directories have been created."

# Run the training
print_section "Training Model" "🚀"
echo "🔥 Training trigram model on Gutenberg data..."
echo "⏳ This may take a few moments depending on the data size."

# Display current settings
echo ""
echo "📋 Current settings from settings.py:"
python -c "
from src.config.settings import settings
print(f'   • N-gram Size: {settings.DEFAULT_NGRAM_SIZE}')
print(f'   • Default Tokens: {settings.DEFAULT_TOKENS}')
print(f'   • Top-K: {settings.DEFAULT_TOP_K}')
print(f'   • Top-P: {settings.DEFAULT_TOP_P}')
print(f'   • Temperature: {settings.DEFAULT_TEMPERATURE}')
print(f'   • Log Level: {settings.LOG_LEVEL}')
print(f'   • Smoothing: {settings.USE_SMOOTHING}')
print(f'   • Stupid Backoff: {settings.USE_STUPID_BACKOFF}')
"
if [ $? -ne 0 ]; then
    print_section "Process Failed" "❌"
    echo "❌ Failed to load settings from settings.py"
    echo "🔍 Please check if the settings file exists and is properly formatted."
    exit 1
fi
echo ""

# Train the model
python -m src.scripts.train gutenberg-model data/gutenberg
if [ $? -ne 0 ]; then
    print_section "Process Failed" "❌"
    echo "❌ Model training failed with error code $?."
    echo "🔍 Please check the error messages above for more information."
    exit 1
fi

# Check if training succeeded
if [ $? -eq 0 ]; then
    print_section "Text Generation" "✨"
    echo "🎨 Generating sample text using the trained model..."
    
    # Generate text with different settings
    echo "📝 Generating with default settings..."
    python -m src.scripts.generate gutenberg-model --tokens 50
    if [ $? -ne 0 ]; then
        print_section "Process Failed" "❌"
        echo "❌ Text generation failed with error code $?."
        echo "🔍 Please check the error messages above for more information."
        exit 1
    fi
    
    echo ""
    echo "📝 Generating with custom seed..."
    python -m src.scripts.generate gutenberg-model --tokens 50 --seed "The quick brown fox" --top-k 25
    if [ $? -ne 0 ]; then
        print_section "Process Failed" "❌"
        echo "❌ Text generation with custom seed failed with error code $?."
        echo "🔍 Please check the error messages above for more information."
        exit 1
    fi
    
    print_section "Process Complete" "🎉"
    echo "🏆 The n-gram language model has been trained and tested successfully!"
    echo ""
    echo "📊 You can find:"
    echo "   💾 Trained model in: ./models/gutenberg-model.json"
    echo "   📝 Generated text samples above"
    echo ""
    
    print_section "Customization Guide" "⚙️"
    echo "There are two ways to customize the model behavior (command line arguments take precedence over settings):"
    echo ""
    echo "1️⃣  Default Settings in src/config/settings.py:"
    echo "   Model Configuration:"
    echo "   • DEFAULT_NGRAM_SIZE = 2      # Size of n-grams (2 for bigrams, 3 for trigrams, etc.)"
    echo "   • USE_SMOOTHING = True        # Enable Laplace smoothing for better handling of rare events"
    echo "   • USE_STUPID_BACKOFF = True   # Enable stupid backoff for unseen n-grams"
    echo "   • BACKOFF_ALPHA = 0.1         # Stupid backoff penalty factor"
    echo ""
    echo "   Generation Parameters:"
    echo "   • DEFAULT_TOKENS = 100        # Number of tokens to generate"
    echo "   • DEFAULT_TOP_K = 25          # Limit sampling to top K most likely tokens"
    echo "   • DEFAULT_TOP_P = None        # Nucleus sampling threshold (disabled by default)"
    echo "   • DEFAULT_TEMPERATURE = 1.0    # Temperature for controlling randomness"
    echo "                                  # < 1.0: more focused, conservative"
    echo "                                  # > 1.0: more diverse, creative"
    echo ""
    echo "   System Settings:"
    echo "   • LOG_LEVEL = 'INFO'          # Logging detail (DEBUG/INFO/WARNING/ERROR)"
    echo ""
    echo "2️⃣  Command Line Arguments (these override settings.py):"
    echo ""
    echo "🔸 Training Arguments:"
    echo "   --n-gram-size INT    Size of n-grams"
    echo "   --smoothing          Enable Laplace smoothing"
    echo "   --no-smoothing       Disable Laplace smoothing"
    echo "   --stupid-backoff     Enable stupid backoff"
    echo "   --no-stupid-backoff  Disable stupid backoff"
    echo ""
    echo "🔸 Generation Arguments:"
    echo "   --tokens INT         Number of tokens to generate"
    echo "   --seed TEXT         Starting text for generation"
    echo "   --top-k INT         Use top-k sampling"
    echo "   --top-p FLOAT       Use nucleus (top-p) sampling (default: disabled)"
    echo "   --temperature FLOAT Temperature for sampling"
    echo ""
    echo "🔸 Training Examples:"
    echo "   • Train a 4-gram model with smoothing and stupid backoff:"
    echo "     uv run python -m src.scripts.train my-model data/gutenberg --n-gram-size 4 --smoothing --stupid-backoff"
    echo ""
    echo "   • Train without smoothing but with backoff:"
    echo "     uv run python -m src.scripts.train custom-model data/gutenberg --n-gram-size 3 --no-smoothing --stupid-backoff"
    echo ""
    echo "🔸 Generation Examples:"
    echo "   • Basic generation with defaults:"
    echo "     uv run python -m src.scripts.generate gutenberg-model"
    echo ""
    echo "   • Generate longer text with seed:"
    echo "     uv run python -m src.scripts.generate gutenberg-model --tokens 200 --seed 'Once upon a time'"
    echo ""
    echo "   • Conservative generation (less random):"
    echo "     uv run python -m src.scripts.generate gutenberg-model --top-k 10 --temperature 0.7"
    echo ""
    echo "   • Creative generation (more random):"
    echo "     uv run python -m src.scripts.generate gutenberg-model --top-k 50 --temperature 1.2"
    echo ""
    echo "   • Advanced generation with all options:"
    echo "     uv run python -m src.scripts.generate gutenberg-model \\"
    echo "       --tokens 150 \\"
    echo "       --seed 'In the beginning' \\"
    echo "       --top-k 15 \\"
    echo "       --top-p 0.9 \\"
    echo "       --temperature 0.8"
else
    print_section "Process Failed" "❌"
    echo "❌ The model training failed with error code $?."
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