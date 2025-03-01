#!/usr/bin/env bash

echo "🚀 Starting project setup..."

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
    return 1  
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
    return 1  
fi

echo "✅ uv is installed: $(uv --version)"
# Create virtual environment with uv
echo "🔨 Creating virtual environment with Python 3.12..."
uv venv --python=3.12

# Determine activation script path
if [[ -f ".venv/bin/activate" ]]; then
    ACTIVATE_PATH=".venv/bin/activate"
else
    echo "❌ Virtual environment activation script not found." 
    echo "Please open the workspace in your IDE and activate the virtual environment manually."
    echo "If you are using VSCode or Cursor, you can use the 'Python: Select Interpreter' command."
    echo "If you are using PyCharm, you can use the 'Python Interpreter' settings."
    echo "If you are using another IDE, please refer to the documentation for activating the virtual environment."
    echo "After activating the virtual environment, please run "uv sync" to install the dependencies, and you're all set!"
    return 1 
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$ACTIVATE_PATH"

# Install dependencies with uv
echo "📚 Installing project dependencies..."
uv sync

# Download spaCy English language model
echo "Downloading spaCy English language model ("en_core_web_md")..."
uv run spacy download en_core_web_md
echo "✅ spaCy English language model installed"

echo "✅ Setup completed successfully!"
echo "🎉 Virtual environment has been activated automatically!"

# Prompt user to run the analysis
echo ""
echo "┌───────────────────────────────────────────────┐"
echo "│                                               │"
echo "│  🚀 Ready to launch the analysis?             │"
echo "│                                               │"
echo "│  Press [ENTER] to start now                   │"
echo "│  Press any other key to start later           │"
echo "│                                               │"
echo "└───────────────────────────────────────────────┘"
echo ""

read -p "➤ " run_analysis
if [[ -z "$run_analysis" || "$run_analysis" == "y" || "$run_analysis" == "Y" ]]; then
    echo "🔥 Launching analysis..."
    bash run.sh
else
    echo "📝 You can run the analysis later with: bash run.sh or ./run.sh"
fi
