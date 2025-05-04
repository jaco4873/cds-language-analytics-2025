"""
Configuration settings for IMDb sentiment analysis.

This module centralizes all configurable parameters.

Usage:
    from settings import settings
    
    # Access settings
    sample_size = settings.SAMPLE_SIZE
"""

from pathlib import Path
from pydantic_settings import BaseSettings
import logging


class Settings(BaseSettings):
    """Application settings with type validation.
    
    All settings can be overridden by environment variables with the same name.
    """
    # System Settings
    LOG_LEVEL: int = logging.INFO
    PROJECT_DIR: Path = Path(__file__).parent
    BASE_DIR: Path = PROJECT_DIR.parent 
    OUTPUT_DIR: Path = BASE_DIR / "outputs" 
    MODEL_DIR: Path = OUTPUT_DIR / "model"
    FIGURES_DIR: Path = OUTPUT_DIR / "figures"
    
    # Data Settings
    SAMPLE_SIZE: int = 10000
    VALIDATION_SPLIT: float = 0.1
    RANDOM_SEED: int = 42
    
    # Sentiment Analysis Constants
    SENTIMENT_ID2LABEL: dict[int, str] = {0: "negative", 1: "positive"}
    SENTIMENT_LABEL2ID: dict[str, int] = {"negative": 0, "positive": 1}
    NUM_SENTIMENT_CLASSES: int = 2
    
     # Model Configuration
    DEFAULT_MODEL: str = "both"  # "logistic", "transformer", or "both"
    
    # Logistic Regression Optimization Settings
    BAYES_SEARCH_ITERATIONS: int = 20
    BAYES_CV_FOLDS: int = 3
    
    # Transformer Settings
    TRANSFORMER_MODEL: str = "distilbert-base-uncased"
    NUM_EPOCHS: int = 3
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-5
    LEARNING_RATE_SCHEDULER_TYPE: str = "linear"
    LOGGING_STEPS: int = 50
    WEIGHT_DECAY: float = 0.01
    SAVE_STRATEGY: str = "epoch"
    SAVE_TOTAL_LIMIT: int = 3
    LOAD_BEST_MODEL_AT_END: bool = True
    METRIC_FOR_BEST_MODEL: str = "eval_loss"
    EVAL_STRATEGY: str = "epoch"
    EARLY_STOPPING_PATIENCE: int = 3
    MAX_GRAD_NORM: float = 0.8
    
    # Visualization Settings
    PLOT_DPI: int = 300
    METRICS_FIGSIZE: tuple = (10, 6)
    DIST_FIGSIZE: tuple = (8, 6)
    REVIEW_LENGTH_FIGSIZE: tuple = (12, 6)
    

    # Ensure directories exist
    def model_init(self):
        """Ensure all required directories exist."""
        for directory in [self.OUTPUT_DIR, self.MODEL_DIR, self.FIGURES_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Create global settings instance
settings = Settings()
settings.model_init()
