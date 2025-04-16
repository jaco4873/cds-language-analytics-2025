"""Settings Configuration

This module defines the default configuration for the BERTopic analysis
using Pydantic for type safety and validation.

Settings Categories:
1. Model Configuration
   - Min topic size 
   - Remove stop words options
   - Other BERTopic parameters

2. Visualization Parameters
   - Plot size
   - Color schemes
   - Font sizes

3. System Settings
   - File paths
   - Data files

Usage:
    ```python
    from src.config.settings import settings
    
    # Access settings
    min_topic_size = settings.MIN_TOPIC_SIZE
    ```
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with type validation.
    
    All settings can be overridden by environment variables with the same name.
    """
    
    # Model Configuration
    MIN_TOPIC_SIZE: int = 25  
    REMOVE_STOPWORDS: bool = True  
    REDUCE_FREQUENT_WORDS: bool = True  
    UMAP_N_NEIGHBORS: int = 15  
    UMAP_N_COMPONENTS: int = 5  
    UMAP_MIN_DIST: float = 0.0 
    
    # Visualization Parameters
    FIGSIZE_WIDTH: int = 12  
    FIGSIZE_HEIGHT: int = 8  
    HEATMAP_CMAP: str = "viridis"
    FONT_SIZE: int = 12  
    
    # System Settings
    PROJECT_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_DIR / "data"
    OUTPUT_DIR: Path = PROJECT_DIR / "output"
    JSONL_FILE: str = "News_Category_Dataset_v3_subset.jsonl"
    HEADLINES_EMBEDDINGS: str = "embeddings_headlines.npy"
    
    # Ensure directories exist
    def model_init(self):
        """Ensure all required directories exist."""
        for directory in [self.DATA_DIR, self.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Create global settings instance
settings = Settings()
settings.model_init() 