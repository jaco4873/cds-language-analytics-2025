"""
Centralized configuration using Pydantic Settings.

This module defines configuration models with defaults for all project settings.
Environment variables can be used to override these defaults.

All configuration is accessible through the `settings` instance.
"""

from typing import Literal
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

    output_dir: str = "output"
    models_dir: str = "output/models"
    reports_dir: str = "output/reports"
    figures_dir: str = "output/figures"
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


# Create a singleton instance
settings = Settings()

# Export the settings instance as the primary interface
__all__ = ["settings"]
