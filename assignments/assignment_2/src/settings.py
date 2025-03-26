"""
Centralized configuration using Pydantic Settings.

This module defines the configuration model and loads settings from:
1. config.yaml file (primary source)
2. Environment variables (fallback/override)

All configuration is accessible through the `settings` instance.
"""

from typing import Literal, Any
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
class DataConfig(BaseSettings):
    """Data loading and preprocessing configuration."""

    csv_path: str = "data/fake_or_real_news.csv"
    test_size: float = 0.2
    random_state: int = 42


class VectorizationConfig(BaseSettings):
    """Text vectorization configuration."""

    skip_vectorization: bool = False
    vectorizer_type: Literal["tfidf", "count"] = "tfidf"
    max_features: int = 10000
    min_df: int = 2
    max_df: float = 0.95
    lowercase: bool = True
    save_vectorized: bool = True
    vectorized_dir: str = "data/vectorized"


class ModelConfig(BaseSettings):
    """Base configuration for all models."""

    enabled: bool = True
    name: str = "model"


class LogisticRegressionConfig(ModelConfig):
    """Logistic Regression model configuration."""

    c_value: float = 1.0
    max_iter: int = 1000
    solver: str = "liblinear"
    name: str = "logistic_regression"


class NeuralNetworkConfig(ModelConfig):
    """Neural Network (MLP) model configuration."""

    hidden_layer_sizes: list[int] = [100, 50]
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: str = "auto"
    learning_rate: str = "adaptive"
    learning_rate_init: float = 0.001
    max_iter: int = 200
    early_stopping: bool = True
    validation_fraction: float = 0.1
    name: str = "neural_network"


class NaiveBayesConfig(ModelConfig):
    """Naive Bayes model configuration."""

    alpha: float = 1.0
    fit_prior: bool = True
    name: str = "naive_bayes"


class ModelsConfig(BaseSettings):
    """Configuration for all models."""

    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    neural_network: NeuralNetworkConfig = NeuralNetworkConfig()
    naive_bayes: NaiveBayesConfig = NaiveBayesConfig()
    random_state: int = 42


class OutputConfig(BaseSettings):
    """Output and results configuration."""

    models_dir: str = "models"
    output_dir: str = "output"
    results_dir: str = "results"
    visualize_results: bool = True


class Settings(BaseSettings):
    """Main configuration class that contains all settings."""

    log_level: str = "INFO"
    data: DataConfig = DataConfig()
    vectorization: VectorizationConfig = VectorizationConfig()
    models: ModelsConfig = ModelsConfig()
    output: OutputConfig = OutputConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    @model_validator(mode="before")
    @classmethod
    def load_from_yaml(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if isinstance(values, dict):
            return values

    @staticmethod
    def _update_nested_dict(
        base_dict: dict[str, Any], update_dict: dict[str, Any]
    ) -> None:
        """
        Update a nested dictionary recursively.

        Args:
            base_dict: The dictionary to update
            update_dict: The dictionary with updates
        """
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                Settings._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value


# Create a singleton instance
settings = Settings()

# Export the settings instance as the primary interface
__all__ = ["settings"]
