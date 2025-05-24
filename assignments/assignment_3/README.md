# Assignment 3: N-gram Language Model

## Introduction
This project implements a generative n-gram language model that can be trained on text data and generate new text based on the learned patterns. The model supports various configurations such as n-gram size, smoothing techniques, and different text generation strategies.

## Data
The project uses the Gutenberg corpus texts located in the `data/gutenberg` directory. These texts serve as the training data for the n-gram model.

## Project Structure

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

## Getting Started
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

### Configuration
All default settings can be found in `src/config/settings.py`:

```python
# Model Configuration
DEFAULT_NGRAM_SIZE = 3      # N-gram size
USE_SMOOTHING = True        # Laplace smoothing
USE_STUPID_BACKOFF = True   # Stupid backoff
BACKOFF_ALPHA = 0.1        # Stupid backoff penalty factor

# Generation Parameters
DEFAULT_TOKENS = 100        # Output length
DEFAULT_TOP_K = 25         # Top-k sampling
DEFAULT_TOP_P = None       # Nucleus sampling (disabled)
DEFAULT_TEMPERATURE = 1.0   # Sampling temperature
```

Command line arguments always take precedence over default settings in `settings.py`, allowing for easy experimentation without modifying code.

## Methods
The project employs n-gram language modeling, which analyzes sequences of n consecutive tokens to predict the next token in a sequence. The implementation includes:

### Training Process
The training script processes text files and builds an n-gram model with the following features:
- Configurable n-gram size (default: 3)
- Optional Laplace smoothing for handling unseen n-grams
- Optional Stupid Backoff strategy for more robust predictions

### Generation Strategies
Text generation implements several techniques to control the output quality:

1. **Temperature Control**
   - **Low (0.1-0.7)**: More focused, predictable text
   - **Medium (0.7-1.0)**: Balanced creativity
   - **High (1.0-2.0)**: More diverse, experimental text

2. **Sampling Methods**
   - **Top-k Sampling**: Restricts to k most likely tokens
   - **Nucleus (Top-p) Sampling**: Dynamically selects vocabulary based on probability threshold

Generate text using the following command:

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

## Result Samples and Discussion

The default 3-gram model trained on the Gutenberg corpus demonstrates the effects of different generation parameters. Below are sample outputs with varying configurations:

### Default Settings (top-k=25, temperature=1.0)
```
particularly poetical grandpapas pell-mell Jattir kindles simile recklesse what Cumberland Vane, jailer never-broken
Cries pell-mell vicarage Calues Jattir, Maides Cumberland implicit gauntleted jailer Jattir implicit jailer, pell-mell
recklesse longevity simile Calues tousled vnkindest himself.
```
The default settings produce text with a mix of archaic terms and proper nouns, showing no coherence. The model appears to be generating disconnected words from the training corpus without meaningful structure.


### High Temperature (top-k=50, temperature=1.2)
```
laid help Ashurites cupbearers Beyond the Pronounce Pronounce Partners twigging deserting alienated from Lucy prophecies 
anthropologists adioyn duels bearing in bearing Even so ye Free-thought dissention sciential conditioning line-knife
bolled Beyond THAT Free-thought bearing, who is the very time 17:5 wedding Pronounce boades bolled...
```
In theory, higher temperature should increase randomness and result in more diverse word combinations. As we can see, the text remains  incoherent. The repetition of words like "Pronounce" and "bearing" suggests the model is still heavily influenced by frequency patterns in the training data.


### Guided Generation with Seed Text
```
Once upon a time, or on the LORD from the land of the gate With STILL advantaged STILL Pertains to Baalah jackets
thatch Dowrie Weak 12:35 A flailings 27:19 yeelding 12:35 Baalah Pertains Recompence today...
```
Despite starting with a familiar narrative seed phrase, the model immediately loses coherence and diverts to biblical-style text with random numbers (likely verse references) and archaic terms. This demonstrates the model's inability to maintain thematic consistency even with explicit guidance.

### Conservative Settings (top-k=10, temperature=0.7)
```
the temporary PUNISHMENT confirmed to nick 105:38 echoes still confirmed, no one could fix PUNISHMENT confirmed. 
As to the king had at first she thought it was a great deal of flour mingled with oil, and the LORD, and the other, 
in his hand. "I do not think it is written in the house of their hands..."
```

With more conservative settings (lower temperature and fewer top-k options), we would expect more predictable text. While there are occasional glimpses of sentence-like structures and some biblical phrasings that appear more complete, the output remains fundamentally incoherent. 

In general, while the model captures vocabulary and occasional grammatical patterns from the corpus, it fails to produce coherent text under any parameter configuration. The biblical nature of much of the output reflects the prominence of such texts in the Gutenberg corpus.

## Troubleshooting Steps
If you encounter issues:
1. Check console error messages
2. Run `./setup.sh` to reset environment
3. Delete .venv and rerun setup if needed
4. Verify training data in data/gutenberg


