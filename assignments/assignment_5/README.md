# IMDb Movie Review Sentiment Analysis

## Introduction
This project implements and compares transformer models with traditional machine learning approaches for sentiment analysis on the IMDb movie review dataset.

This repository contains the code and implementation details for comparing DistilBERT (a transformer model) against TF-IDF + Logistic Regression (a traditional ML approach) for IMDb movie review sentiment classification.

For the full research report, methodology, results, and discussion, please see [REPORT.md](./REPORT.md).

## Data
The project uses the IMDb movie review dataset from HuggingFace and provides:

- Binary sentiment classification (positive/negative reviews)
- Dataset loaded directly using HuggingFace Datasets library
- Balanced dataset creation through stratified sampling to maintain equal positive/negative distributions
- Train/validation/test splitting functionality

## Methods

### Models

This project implements two different approaches:

**Baseline Model:**
- TF-IDF vectorization with n-gram range: 1-3
- Bayesian hyperparameter optimization
- Logistic Regression classifier

**Transformer Model:**
- DistilBERT (distilbert-base-uncased) for sequence classification
- Text truncation/padding to 512 tokens maximum length
- Fine-tuning approach with early stopping

## Getting Started

### Quick Start

Simply run the provided script to start the analysis:

```bash
./run.sh
```

This script handles everything automatically:
- Checks if the environment is set up, and prompts to run setup if needed
- Displays current configuration settings
- Runs the IMDb sentiment analysis pipeline

On first run, the system will automatically download the IMDb dataset from Hugging Face and cache it locally for future use.

### CLI Options

If you prefer more control, you can run the analysis with custom parameters:

```bash
python src/main.py run --sample-size 10000 --model transformer --num-epochs 3
```

Available parameters:
- `--sample-size`: Number of samples to use (default: 10000)
- `--model`: Which model to train - "logistic", "transformer", or "both" (default: "both")
- `--num-epochs`: Number of epochs for transformer training (default: 3)
- `--resume-from`: Resume training from a checkpoint (e.g., checkpoint-3000)

### Resuming Training

You can continue training from a saved checkpoint:

```bash
python src/main.py run --model transformer --resume-from checkpoint-3000
```

## Project Structure

```
assignment_5/
├── output/                    # Output directory
│   ├── model/                 # Saved model files
│   └── figures/               # Visualization outputs
├── src/                       # Source code
│   ├── main.py                # Main execution script
│   ├── settings.py            # Centralized configuration
│   ├── models/                # Model implementations
│   │   ├── distil_bert.py     # Transformer model
│   │   └── logistic_regression.py # Baseline model
│   ├── utils/                 # Utility functions
│   │   ├── data.py            # Data loading and preprocessing
│   │   └── visualization.py   # Visualization functions
├── README.md                  # Project documentation
├── REPORT.md                  # Academic research report
├── run.sh                     # Execution script
└── setup.sh                   # Environment setup script
```

## Technology Stack

- **Data Processing**: HuggingFace Datasets
- **Machine Learning**: scikit-learn, HuggingFace Transformers
- **Hyperparameter Optimization**: scikit-optimize (Bayesian optimization)
- **Visualization**: Matplotlib
- **Configuration**: Pydantic
- **CLI Interface**: Click

## Acknowledgments

- This project uses the IMDb dataset created by Maas et al. (2011)
- Transformer models powered by the [HuggingFace Transformers](https://huggingface.co/transformers/) library
- Dataset handling provided by [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
