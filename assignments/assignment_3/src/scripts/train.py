"""Training Script for N-gram Language Model

This script provides a command-line interface for training n-gram language models.
It supports configuration of model parameters and features through both
command-line arguments and settings.py defaults.

Features:
    - Configurable n-gram size
    - Optional Laplace smoothing
    - Optional stupid backoff with lower-order models
    - Progress tracking with tqdm
    - Detailed logging

Usage:
    ```bash
    # Basic usage
    python -m src.scripts.train MODEL_NAME DATA_PATH
    
    # With all features
    python -m src.scripts.train MODEL_NAME DATA_PATH \
        --n-gram-size 3 \
        --smoothing \
        --stupid-backoff
    
    # Custom configuration
    LOG_LEVEL=DEBUG python -m src.scripts.train MODEL_NAME DATA_PATH
    ```

Arguments:
    MODEL_NAME: Name for saving the model
    DATA_PATH: Directory containing training text files
    --n-gram-size: Size of n-grams (default from settings.py)
    --smoothing/--no-smoothing: Enable/disable Laplace smoothing
    --stupid-backoff/--no-stupid-backoff: Enable/disable backoff

The script will:
1. Create necessary directories
2. Initialize model with specified parameters
3. Train on all .txt files in DATA_PATH
4. Save model and any lower-order models
"""

import click
from ..core.ngram import NgramModel
from ..config.settings import settings
from ..utils.logger import logger

@click.command()
@click.argument('model_name', type=str)
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--n-gram-size', type=int, default=settings.DEFAULT_NGRAM_SIZE,
              help=f'Size of n-grams to use (default: {settings.DEFAULT_NGRAM_SIZE})')
@click.option('--smoothing/--no-smoothing', default=settings.USE_SMOOTHING,
              help=f'Enable count smoothing (default: {"enabled" if settings.USE_SMOOTHING else "disabled"})')
@click.option('--stupid-backoff/--no-stupid-backoff', default=settings.USE_STUPID_BACKOFF,
              help=f'Enable stupid backoff with lower-order n-grams (default: {"enabled" if settings.USE_STUPID_BACKOFF else "disabled"})')
def train(model_name: str, data_path: str, n_gram_size: int, 
         smoothing: bool, stupid_backoff: bool):
    """Train an n-gram language model."""
    # Log all settings being used
    logger.info(f"Training {n_gram_size}-gram model '{model_name}' on {data_path}")
    logger.info("Model settings:")
    logger.info(f"  • N-gram size: {n_gram_size}")
    logger.info(f"  • Smoothing: {'enabled' if smoothing else 'disabled'}")
    logger.info(f"  • Stupid Backoff: {'enabled' if stupid_backoff else 'disabled'}")
    
    # Create and train model with explicit settings
    model = NgramModel(
        name=model_name,
        n_gram_size=n_gram_size,
        use_smoothing=smoothing,
        use_stupid_backoff=stupid_backoff
    )
    
    # Train model
    try:
        model.train(data_path)
        model.save()
        logger.info("Training complete!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    train() 