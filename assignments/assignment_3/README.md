# 🔮 N-gram Language Model Generator

This project implements a generative n-gram language model that can be trained on text data and generate new text based on the learned patterns.

## ⚡ Quickstart

The easiest way to get started is to simply run:
```bash
./run.sh
```
This script will:
1. Check if setup is needed and run `setup.sh` automatically if required
2. Download and prepare training data
3. Train a model with default settings
4. Generate sample texts with different parameters
5. Show a comprehensive guide for customization

For more control, you can run the steps manually:
```bash
# 1. Setup environment and download data
./setup.sh

# 2. Train with custom settings
uv run python -m src.scripts.train gutenberg-model data/gutenberg \
    --n-gram-size 3 \
    --smoothing \
    --stupid-backoff

# 3. Generate text with various parameters
uv run python -m src.scripts.generate gutenberg-model \
    --tokens 150 \
    --seed "In the beginning" \
    --top-k 15 \
    --temperature 0.9
```

## 🏗️ Project Structure

```
assignments/assignment_3/
├── data/               # Training data
│   └── gutenberg/     # Gutenberg corpus texts
├── models/            # Saved model files
├── output/            # Generated text output
├── src/
│   ├── core/          # Core model implementation
│   │   └── ngram.py   # N-gram model class
│   ├── utils/         # Utility modules
│   │   └── logger.py  # Logging configuration
│   ├── config/        # Configuration
│   │   └── settings.py # Project settings
│   └── scripts/       # Command-line interfaces
│       ├── train.py   # Model training script
│       ├── generate.py # Text generation script
│       └── download_data.py # Dataset downloader
├── pyproject.toml    # Project dependencies
├── setup.sh          # Environment setup
└── run.sh           # Training/testing script
```
## 📚 Detailed Usage Guide

The project uses Click as the command-line interface (CLI) tool, providing a more intuitive and user-friendly command structure compared to argparse. Click handles command parsing, option validation, and help documentation automatically.

### Training Models

```bash
uv run python -m src.scripts.train MODEL_NAME DATA_PATH [OPTIONS]

Options:
  --n-gram-size INT     Size of n-grams (default: 2)
  --smoothing           Enable Laplace smoothing
  --stupid-backoff      Enable stupid backoff
  --help               Show this message and exit.
```

Example:
```bash
# Train a trigram model with smoothing and stupid backoff
uv run python -m src.scripts.train gutenberg-model data/gutenberg \
    --n-gram-size 3 \
    --smoothing \
    --stupid-backoff
```

### Generating Text

```bash
uv run python -m src.scripts.generate MODEL_NAME [OPTIONS]

Options:
  --tokens INT          Number of tokens to generate
  --seed TEXT          Starting text for generation
  --top-k INT          Limit to top K most likely tokens
  --top-p FLOAT        Nucleus sampling threshold
  --temperature FLOAT  Sampling temperature (default: 1.0)
  --help               Show this message and exit.
```

Examples:
```bash
# Basic generation
uv run python -m src.scripts.generate gutenberg-model --tokens 100

# Creative generation with seed
uv run python -m src.scripts.generate gutenberg-model \
    --tokens 200 \
    --seed "Once upon a time" \
    --top-k 50 \
    --temperature 1.2

# Conservative generation
uv run python -m src.scripts.generate gutenberg-model \
    --tokens 150 \
    --top-k 10 \
    --temperature 0.7
```

### ⚙️ Configuration

All default settings can be found in `src/config/settings.py`:
Command line arguments always take precedence over default settings in `settings.py`, allowing for easy experimentation without modifying code.

```python
# Model Configuration
DEFAULT_NGRAM_SIZE = 2      # N-gram size
USE_SMOOTHING = True        # Laplace smoothing
USE_STUPID_BACKOFF = True   # Stupid backoff
BACKOFF_ALPHA = 0.2        # Stupid backoff penalty factor

# Generation Parameters
DEFAULT_TOKENS = 100        # Output length
DEFAULT_TOP_K = 25         # Top-k sampling
DEFAULT_TOP_P = None       # Nucleus sampling (disabled)
DEFAULT_TEMPERATURE = 1.0   # Sampling temperature
```

## 🎯 Generation Strategies

### Temperature Control
- **Low (0.1-0.7)**: More focused, predictable text
- **Medium (0.7-1.0)**: Balanced creativity
- **High (1.0-2.0)**: More diverse, experimental text

### Sampling Methods
1. **Top-k Sampling**
   - Restricts to k most likely tokens
   - Example: `--top-k 25`

2. **Nucleus (Top-p) Sampling**
   - Dynamically selects vocabulary
   - Better for maintaining coherence
   - Example: `--top-p 0.9`


## 🔍 Debugging & Error Handling

### Logging
Enable debug logging for more detailed insights:
```bash
LOG_LEVEL=DEBUG uv run python -m src.scripts.generate gutenberg-model
```
This shows:
- Smoothing calculations
- Backoff decisions
- Probability distributions
- Token selection process

### Automated Safety Features 🔧
The system includes:
1. **Environment Management**
   - OS detection (macOS/Linux)
   - Virtual environment handling
   - Dependency management
   - NLTK data downloads

2. **Data & Training Safeguards**
   - Training data validation
   - Directory structure verification
   - Configuration validation
   - Detailed error reporting

### Troubleshooting Steps
If you encounter issues:
1. Check console error messages
2. Run `./setup.sh` to reset environment
3. Delete .venv and rerun setup if needed
4. Verify training data in data/gutenberg