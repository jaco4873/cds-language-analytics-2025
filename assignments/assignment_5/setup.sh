#!/usr/bin/env bash

echo "🚀 Starting IMDb Sentiment Analysis setup..."

# Detect the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📱 macOS detected"
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux"* ]]; then
    echo "🐧 Linux detected"
    OS_TYPE="linux"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "This script only supports macOS and Linux."
    exit 1  
fi

# Find Python executable (try python3 first, then python)
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Found Python: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Found Python: $(python --version)"
else
    echo "❌ Python not found. Please install Python 3.8+ before continuing."
    echo "   Visit https://www.python.org/downloads/ for installation instructions."
    exit 1
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to the current PATH
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Failed to install uv. Please install it manually:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1  
fi

echo "✅ uv is installed: $(uv --version)"

# Create virtual environment with the detected Python
echo "🔨 Creating virtual environment..."
$PYTHON_CMD -m venv .venv

# Check if venv creation was successful
if [ ! -d ".venv" ]; then
    echo "❌ Failed to create virtual environment."
    echo "   Please try creating it manually with: $PYTHON_CMD -m venv .venv"
    exit 1
fi

# Activate the virtual environment
echo "🔌 Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # For Windows Git Bash
    source .venv/Scripts/activate
else
    echo "❌ Couldn't find activation script in .venv directory."
    echo "   Please try activating it manually and then run 'uv sync'."
    exit 1
fi

# Verify activation worked
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment activation failed."
    echo "   Please try activating it manually with: source .venv/bin/activate"
    exit 1
fi

# Install dependencies with uv sync
echo "📦 Installing dependencies from pyproject.toml..."
uv sync

# Note about directories
echo "📁 Directories will be created automatically when the app runs"
echo "   All necessary folders are handled by the configuration system"

# Data preparation notice
echo "📊 Note about data:"
echo "   • The app loads IMDb data directly from Hugging Face"
echo "   • On first run, data will be cached locally (transparent download)"
echo "   • This may take a few minutes depending on your connection"
echo "   • Subsequent runs will use the cached data"

echo "✅ Setup completed successfully!"
echo "🎉 Virtual environment has been activated and everything is ready!"

# Prompt user to run the model
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃                                                                ┃"
echo "┃   🌟 🌟 🌟  READY TO LAUNCH  🌟 🌟 🌟                          ┃"
echo "┃                                                                ┃"
echo "┃   🚀 Ready to analyze IMDb sentiment with ML & transformers?   ┃"
echo "┃                                                                ┃"
echo "┃   ✅ Press [ENTER] to start now                                ┃"
echo "┃   ⏱️  Press any other key to start later                        ┃"
echo "┃                                                                ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""

read -p "➤ " run_model
if [[ -z "$run_model" || "$run_model" == "y" || "$run_model" == "Y" ]]; then
    echo "🔥 Launching IMDb sentiment analysis..."
    bash run.sh
else
    echo "📝 You can run the model later with: bash run.sh or ./run.sh"
fi
