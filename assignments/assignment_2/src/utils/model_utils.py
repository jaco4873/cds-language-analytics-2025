"""
Model utility functions for text classification tasks.
This module contains functions for saving models and evaluation reports.
"""

import os
import pickle
from typing import Any
from sklearn.metrics import classification_report

from utils.common import ensure_dir
from settings import settings


def save_model(model, model_name, models_dir=None) -> None:
    """
    Save a trained model to disk.

    Parameters:
    -----------
    model : trained model object
        The model to save
    model_name : str
        Name to use for the saved model file
    models_dir : str, optional
        Directory to save the model
    """
    if models_dir is None:
        models_dir = settings.output.models_dir

    ensure_dir(models_dir)
    model_path = os.path.join(models_dir, f"{model_name}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")


def save_classification_report(y_true, y_pred, report_name, output_dir=None) -> None:
    """
    Generate and save a classification report to a text file.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    report_name : str
        Name to use for the report file
    output_dir : str, optional
        Directory to save the report
    """
    if output_dir is None:
        output_dir = settings.output.output_dir

    ensure_dir(output_dir)
    report_path = os.path.join(output_dir, f"{report_name}.txt")

    # Generate the classification report
    report = classification_report(y_true, y_pred, target_names=["REAL", "FAKE"])

    # Add some additional information
    report_content = f"Classification Report for {report_name}\n"
    report_content += "=" * 50 + "\n\n"
    report_content += report

    # Save to file
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"Classification report saved to {report_path}")
    print("\nClassification Report:")
    print(report)


def load_model(model_name, models_dir=None) -> Any:
    """
    Load a trained model from disk.

    Parameters:
    -----------
    model_name : str
        Name of the model file to load
    models_dir : str, optional
        Directory to load the model from

    Returns:
    --------
    model : trained model object
        The loaded model
    """
    if models_dir is None:
        models_dir = settings.output.models_dir

    model_path = os.path.join(models_dir, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(
    model, X_test, y_test, model_name, training_time, models_dir, output_dir
) -> dict:
    """
    Evaluate model, save results, and return metrics.

    Parameters:
    -----------
    model : object
        Trained model to evaluate
    X_test, y_test : testing data
    model_name : str
        Name of the model for saving
    training_time : float
        Time taken for training the model
    models_dir, output_dir : str
        Directories for saving model and results

    Returns:
    --------
    metrics : dict
        Dictionary of model performance metrics
    """
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

    print(f"\n{model_name} model training and evaluation complete!")
    print(f"Training time: {metrics['training_time']:.2f} seconds")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    print(f"Real precision: {metrics['real_precision']:.4f}")
    print(f"Fake precision: {metrics['fake_precision']:.4f}")
    print(f"Real recall: {metrics['real_recall']:.4f}")
    print(f"Fake recall: {metrics['fake_recall']:.4f}")
    print(f"Real F1-Score: {metrics['real_f1']:.4f}")
    print(f"Fake F1-Score: {metrics['fake_f1']:.4f}")
    print(f"Model saved to: {os.path.join(models_dir, model_name + '.pkl')}")
    print(
        f"Classification report saved to: {os.path.join(output_dir, model_name + '_report.txt')}"
    )

    # Save model
    save_model(model, model_name, models_dir)

    # Save classification report
    report_name = f"{model_name}_report"
    save_classification_report(y_test, y_pred, report_name, output_dir)

    return metrics
