"""Project Settings and Configuration

This module defines the default configuration for the n-gram language model
using Pydantic for type safety and validation. These settings can be
overridden by command-line arguments in the training and generation scripts.

Settings Categories:
1. Model Configuration
   - N-gram size
   - Smoothing settings
   - Backoff settings

2. Generation Parameters
   - Output length
   - Sampling strategies
   - Temperature control

3. System Settings
   - File paths
   - Logging configuration

Usage:
    ```python
    from .settings import settings
    
    # Access settings
    n = settings.DEFAULT_NGRAM_SIZE
    temp = settings.DEFAULT_TEMPERATURE
    
    # Use in model
    model = NgramModel(
        "my-model",
        n_gram_size=settings.DEFAULT_NGRAM_SIZE,
        use_smoothing=settings.USE_SMOOTHING
    )
    ```

Environment Variables:
    LOG_LEVEL: Set logging detail (DEBUG/INFO/WARNING/ERROR)
    MODELS_DIR: Override default models directory
    DATA_DIR: Override default data directory
"""

from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with type validation.
    
    All settings can be overridden by environment variables with the same name.
    Command-line arguments take precedence over both defaults and environment variables.
    """
    
    # Model Configuration
    DEFAULT_NGRAM_SIZE: int = 3  # Size of n-grams (2=bigrams, 3=trigrams)
    USE_SMOOTHING: bool = True   # Enable Laplace smoothing
    USE_STUPID_BACKOFF: bool = True  # Enable stupid backoff
    BACKOFF_ALPHA: float = 0.1 # Stupid backoff alpha (low default due to little training data)
    
    # Generation Parameters
    DEFAULT_TOKENS: int = 100    # Number of tokens to generate
    DEFAULT_TOP_K: int | None = 25      # Limit to top K tokens (None=disabled)
    # Note: Only one of top_k or top_p should be enabled at a time
    DEFAULT_TOP_P: float | None = None  # Nucleus sampling threshold (None=disabled)
    DEFAULT_TEMPERATURE: float = 1.0  # Sampling temperature
    
    # System Settings
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    PROJECT_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_DIR / "data"
    MODELS_DIR: Path = PROJECT_DIR / "models"
    OUTPUT_DIR: Path = PROJECT_DIR / "output"
    
    # Ensure directories exist
    def model_init(self):
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Create global settings instance
settings = Settings()
settings.model_init() 