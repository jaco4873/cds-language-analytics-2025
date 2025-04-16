#!/bin/bash

# BERTopic News Analysis Run Script
echo "
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                ┃
┃   🔮 BERTopic News Headline Analysis 🔮                       ┃
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

# Check for data
print_section "Data Check" "📊"
if [ ! -f "data/News_Category_Dataset_v3_subset.jsonl" ] || [ ! -f "data/embeddings_headlines.npy" ]; then
    echo "⚠️  Warning: Required data files not found in data directory."
    echo "📝 Please ensure the following files exist in the data directory:"
    echo "   • News_Category_Dataset_v3_subset.jsonl"
    echo "   • embeddings_headlines.npy"
    echo "❌ Exiting. Cannot continue without required data."
    exit 1
else
    echo "✅ Data files found."
fi

# Ensure required directories exist
print_section "Directory Setup" "📁"
mkdir -p output
echo "✅ Required directories have been created."

# Display current settings
print_section "Configuration" "⚙️"
echo "📋 Current settings from settings.py:"
python -c "
from src.config.settings import settings
print(f'   • Minimum Topic Size: {settings.MIN_TOPIC_SIZE}')
print(f'   • Remove Stopwords: {settings.REMOVE_STOPWORDS}')
print(f'   • Reduce Frequent Words: {settings.REDUCE_FREQUENT_WORDS}')
print(f'   • UMAP n_neighbors: {settings.UMAP_N_NEIGHBORS}')
print(f'   • UMAP n_components: {settings.UMAP_N_COMPONENTS}')
print(f'   • UMAP min_dist: {settings.UMAP_MIN_DIST}')
"
if [ $? -ne 0 ]; then
    print_section "Process Failed" "❌"
    echo "❌ Failed to load settings from settings.py"
    echo "🔍 Please check if the settings file exists and is properly formatted."
    exit 1
fi
echo ""

# Run the topic modeling
print_section "Topic Modeling" "🚀"
echo "🔥 Running BERTopic analysis on news headlines..."
echo "⏳ This may take a few moments depending on the data size."

python -m src.main
if [ $? -ne 0 ]; then
    print_section "Process Failed" "❌"
    echo "❌ Topic modeling failed with error code $?."
    echo "🔍 Please check the error messages above for more information."
    exit 1
fi

# Check if topic modeling succeeded
if [ $? -eq 0 ]; then
    print_section "Process Complete" "🎉"
    echo "🏆 The BERTopic analysis has been completed successfully!"
    echo ""
    echo "📊 You can find the results in the output directory:"
    echo "   💾 Visualization files (.html and .png)"
    echo "   📝 Analysis summary"
    echo ""
else
    print_section "Process Failed" "❌"
    echo "❌ The topic modeling failed with error code $?."
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