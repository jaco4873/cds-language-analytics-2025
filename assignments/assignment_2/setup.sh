#!/usr/bin/env bash

echo "🚀 Starting Text Classification Benchmark setup..."

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

# Create virtual environment with uv
echo "🔨 Creating virtual environmentand installing dependencies ..."
uv sync

# Determine activation script path
if [[ -f ".venv/bin/activate" ]]; then
    ACTIVATE_PATH=".venv/bin/activate"
else
    echo "❌ Virtual environment activation script not found." 
    echo "Please open the workspace in your IDE and activate the virtual environment manually."
    echo "After activating the virtual environment, please run 'uv sync' to install the dependencies."
    exit 1 
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$ACTIVATE_PATH"

echo "✅ Setup completed successfully!"
echo "🎉 Virtual environment has been activated automatically!"

# Check if data file exists
if [[ ! -f "data/fake_or_real_news.csv" ]]; then
    echo "⚠️ Fake News dataset not found in data directory."
    echo "Please make sure to place the dataset file in the data directory before running the analysis."
fi

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p data/vectorized models output results

# Prompt user to run the analysis
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃                                                                ┃"
echo "┃   🌟 🌟 🌟  READY TO LAUNCH  🌟 🌟 🌟                          ┃"
echo "┃                                                                ┃"
echo "┃   🚀 Ready to launch the text classification analysis?         ┃"
echo "┃                                                                ┃"
echo "┃   ✅ Press [ENTER] to start now                               ┃"
echo "┃   ⏱️  Press any other key to start later                      ┃"
echo "┃                                                                ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""

read -p "➤ " run_analysis
if [[ -z "$run_analysis" || "$run_analysis" == "y" || "$run_analysis" == "Y" ]]; then
    echo "🔥 Launching analysis..."
    bash run.sh
else
    echo "📝 You can run the analysis later with: bash run.sh or ./run.sh"
fi 