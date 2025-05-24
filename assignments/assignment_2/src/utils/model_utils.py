"""
Model utility functions for text classification tasks.
"""

import os
import pickle
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report

from utils.common import ensure_dir
from utils.logger import logger
from settings import settings


def save_model(
    model: BaseEstimator, model_name: str, models_dir: str | None = None
) -> None:
    """
    Save a trained model to disk.

    Parameters:
    -----------
    model : BaseEstimator
        The model to save
    model_name : str
        Name to use for the saved model file
    models_dir : Optional[str]
        Directory to save the model
    """
    if models_dir is None:
        models_dir = settings.output.models_dir

    ensure_dir(models_dir)
    model_path = os.path.join(models_dir, f"{model_name}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {model_path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    report_name: str,
    reports_dir: str | None = None,
) -> None:
    """
    Generate and save a classification report to a text file.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    report_name : str
        Name to use for the report file
    reports_dir : str | None
        Directory to save the report
    """
    if reports_dir is None:
        reports_dir = settings.output.reports_dir

    ensure_dir(reports_dir)
    report_path = os.path.join(reports_dir, f"{report_name}.txt")

    # Generate the classification report
    report = classification_report(y_true, y_pred, target_names=["REAL", "FAKE"])

    # Add some additional information
    report_content = f"Classification Report for {report_name}\n"
    report_content += "=" * 50 + "\n\n"
    report_content += report

    # Save to file
    with open(report_path, "w") as f:
        f.write(report_content)

    logger.info(f"Classification report saved to {report_path}")
    logger.info(f"Classification Report: \n{report}")


def load_model(model_name: str, models_dir: str | None = None) -> Any:
    """
    Load a trained model from disk.

    Parameters:
    -----------
    model_name : str
        Name of the model file to load
    models_dir : str | None
        Directory to load the model from

    Returns:
    --------
    model : Any
        The loaded model
    """
    if models_dir is None:
        models_dir = settings.output.models_dir

    model_path = os.path.join(models_dir, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        error_msg = f"Model file not found at {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Model loaded from {model_path}")
    return model


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    training_time: float,
    models_dir: str | None = None,
    reports_dir: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate model, save results, and return metrics.

    Parameters:
    -----------
    model : BaseEstimator
        Trained model to evaluate
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model for saving
    training_time : float
        Time taken for training the model
    models_dir : str | None
        Directory for saving model
    reports_dir : str | None
        Directory for saving results

    Returns:
    --------
    metrics : dict[str, Any]
        Dictionary of model performance metrics
    """
    if models_dir is None:
        models_dir = settings.output.models_dir
    if reports_dir is None:
        reports_dir = settings.output.reports_dir

    # Get predictions
    y_pred = model.predict(X_test)

    # Get classification report as dictionary
    report = classification_report(
        y_test, y_pred, target_names=["REAL", "FAKE"], output_dict=True
    )

    # Extract key metrics
    metrics = {
        "model": model_name,
        "description": model_name.replace("_", " ").title(),
        "real_precision": report["REAL"]["precision"],
        "real_recall": report["REAL"]["recall"],
        "real_f1": report["REAL"]["f1-score"],
        "fake_precision": report["FAKE"]["precision"],
        "fake_recall": report["FAKE"]["recall"],
        "fake_f1": report["FAKE"]["f1-score"],
        "accuracy": report["accuracy"],
        "training_time": training_time,
    }

    logger.info(f"\n{model_name} model training and evaluation complete!")
    logger.info(f"Training time: {metrics['training_time']:.2f} seconds")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

    logger.info(f"Real precision: {metrics['real_precision']:.4f}")
    logger.info(f"Fake precision: {metrics['fake_precision']:.4f}")
    logger.info(f"Real recall: {metrics['real_recall']:.4f}")
    logger.info(f"Fake recall: {metrics['fake_recall']:.4f}")
    logger.info(f"Real F1-Score: {metrics['real_f1']:.4f}")
    logger.info(f"Fake F1-Score: {metrics['fake_f1']:.4f}")
    logger.info(f"Model saved to: {os.path.join(models_dir, model_name + '.pkl')}")
    logger.info(
        f"Classification report saved to: {os.path.join(reports_dir, model_name + '_report.txt')}"
    )

    # Save model
    save_model(model, model_name, models_dir)

    # Save classification report
    report_name = f"{model_name}_report"
    save_classification_report(y_test, y_pred, report_name, reports_dir)

    return metrics
