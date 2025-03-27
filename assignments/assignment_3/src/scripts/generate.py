"""Text Generation Script for N-gram Language Model

This script provides a command-line interface for generating text using
trained n-gram language models. It offers multiple generation strategies
and parameters for controlling the output.

Features:
    - Seed text support
    - Multiple sampling strategies:
        - Top-k sampling
        - Nucleus (top-p) sampling
        - Temperature scaling
    - Configurable output length
    - Detailed debug logging

Usage:
    ```bash
    # Basic usage
    python -m src.scripts.generate MODEL_NAME
    
    # With seed text
    python -m src.scripts.generate MODEL_NAME \
        --seed "Once upon a time" \
        --tokens 200
    
    # Advanced sampling
    python -m src.scripts.generate MODEL_NAME \
        --top-k 25 \
        --top-p 0.9 \
        --temperature 0.8
    
    # Debug mode
    LOG_LEVEL=DEBUG python -m src.scripts.generate MODEL_NAME
    ```

Arguments:
    MODEL_NAME: Name of the trained model to use
    --tokens: Number of tokens to generate
    --seed: Starting text for generation
    --top-k: Limit sampling to top K most likely tokens
    --top-p: Use nucleus sampling with threshold P
    --temperature: Sampling temperature (higher = more random)

The script will:
1. Load the specified model and its settings
2. Process any seed text
3. Generate text using specified parameters
4. Print the generated text to stdout

Debug logging will show:
- Probability calculations
- Sampling decisions
- Backoff operations
- Token selections
"""

import click
from nltk import word_tokenize

from ..core.ngram import NgramModel
from ..config.settings import settings
from ..utils.logger import logger

@click.command()
@click.argument("model_name", type=str)
@click.option("--tokens", "-t", type=int, default=settings.DEFAULT_TOKENS,
              help="Number of tokens to generate")
@click.option("--seed", "-s", type=str, help="Starting text for generation")
@click.option("--top-k", "-k", type=int, default=settings.DEFAULT_TOP_K,
              help="Use top-k sampling")
@click.option("--top-p", "-p", type=float, default=settings.DEFAULT_TOP_P,
              help="Use nucleus (top-p) sampling")
@click.option("--temperature", type=float, default=settings.DEFAULT_TEMPERATURE,
              help="Temperature for sampling")
def generate(model_name: str, tokens: int, seed: str | None, 
            top_k: int | None, top_p: float | None, temperature: float):
    """Generate text using a trained n-gram language model."""
    try:
        logger.info(f"Loading model '{model_name}'")
        model = NgramModel.load(model_name)
        logger.info(f"Loaded {model.n_gram_size}-gram model")
        logger.info(f"Model settings:")
        logger.info(f"  • Smoothing: {'enabled' if model.use_smoothing else 'disabled'}")
        logger.info(f"  • Stupid Backoff: {'enabled' if model.use_stupid_backoff else 'disabled'}")
        logger.info(f"  • Vocabulary size: {len(model.vocab)}")

        # Log generation settings
        logger.info("Generation settings:")
        logger.info(f"  • Tokens: {tokens}")
        logger.info(f"  • Top-k: {top_k if top_k else 'disabled'}")
        logger.info(f"  • Top-p: {top_p if top_p else 'disabled'}")
        logger.info(f"  • Temperature: {temperature}")
        
        # Process seed if provided
        seed_tuple: tuple[str, ...] | None = None
        if seed:
            logger.info(f"Using seed text: '{seed}'")
            seed_tuple = tuple(word_tokenize(seed))
        
        # Generate text 
        logger.info("Generating text...")
        generated_text = model.generate(
            seed=seed_tuple,
            tokens=tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
        
        # Output the generated text
        logger.info("Generated text:")
        logger.info("-" * 60)
        logger.info(f"{generated_text}")
        logger.info("-" * 60)
        logger.info("Generation complete!")
        
    except FileNotFoundError:
        logger.error(f"Could not find model '{model_name}'")
        raise
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    generate() 