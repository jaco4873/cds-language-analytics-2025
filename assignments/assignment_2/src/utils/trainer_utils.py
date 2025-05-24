"""
Utility functions for model training to reduce boilerplate code across trainers.
"""

import time
from typing import Any, Callable
import numpy as np
from sklearn.base import BaseEstimator
from utils.model_utils import evaluate_model
from utils.common import ensure_dir
from utils.logger import logger
from settings import settings


def setup_training_environment(config: Any) -> tuple[str, str, str]:
    """
    Set up and ensure existence of training environment directories.

    Parameters:
    -----------
    config : Any
        Configuration object with model settings

    Returns:
    --------
    tuple[str, str, str]
        Tuple containing (models_dir, reports_dir, model_name)
    """
    # Set up directories from settings
    models_dir = settings.output.models_dir
    reports_dir = settings.output.reports_dir
    model_name = config.name

    # Ensure output directories exist
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

    return models_dir, reports_dir, model_name


def train_with_timing(
    model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray
) -> tuple[BaseEstimator, float]:
    """
    Train model with timing.

    Parameters:
    -----------
    model : BaseEstimator
        Scikit-learn model to train
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels

    Returns:
    --------
    tuple[BaseEstimator, float]
        Trained model and training time in seconds
    """
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    return model, training_time


def train_and_evaluate_model(
    model_factory: Callable[[], BaseEstimator],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: Any,
    model_info: str | None = None,
) -> dict[str, Any]:
    """
    Complete model training pipeline with shared boilerplate code.

    Parameters:
    -----------
    model_factory : Callable[[], BaseEstimator]
        Function that creates and returns a scikit-learn model
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Testing labels
    config : Any
        Configuration object with model settings
    model_info : Optional[str]
        Optional string with model info to log

    Returns:
    --------
    Dict[str, Any]
        Dictionary of model performance metrics
    """
    # Setup directories
    models_dir, reports_dir, model_name = setup_training_environment(config)

    # Log model information if provided
    if model_info:
        logger.info(model_info)

    # Create model
    model = model_factory()

    # Train model with timing
    model, training_time = train_with_timing(model, X_train, y_train)

    # If the model has n_iter_ attribute (convergence info), log it
    if hasattr(model, "n_iter_"):
        if model.n_iter_ < config.max_iter:
            logger.info(f"Convergence achieved after {model.n_iter_} iterations")
        else:
            logger.warning(
                f"Maximum iterations ({config.max_iter}) reached without convergence"
            )

    # Evaluate and save the model and results
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        model_name,
        training_time,
        models_dir,
        reports_dir,
    )

    return metrics
