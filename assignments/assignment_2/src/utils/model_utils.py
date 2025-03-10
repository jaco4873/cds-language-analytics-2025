"""
Model utility functions for text classification tasks.
This module contains functions for saving models and evaluation reports.
"""

import os
import pickle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

from utils.common import ensure_dir
from settings import settings


def save_model(model, model_name, models_dir=None):
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


def save_classification_report(y_true, y_pred, report_name, output_dir=None):
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


def load_model(model_name, models_dir=None):
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


def train_model(model_type, config, X_train, y_train):
    """
    Create and train a model based on config.

    Parameters:
    -----------
    model_type : str
        Type of model to create ('logistic_regression', 'neural_network', or 'naive_bayes')
    config : object
        Configuration object with model parameters
    X_train, y_train : training data

    Returns:
    --------
    trained_model : object
        The trained model
    """
    if model_type == "logistic_regression":
        print(
            f"Training Logistic Regression (C={config.c_value}, max_iter={config.max_iter}, "
            f"solver={config.solver})..."
        )
        model = LogisticRegression(
            C=config.c_value,
            max_iter=config.max_iter,
            random_state=42,
            solver=config.solver,
            n_jobs=-1,
        )

    elif model_type == "neural_network":
        print(
            f"Training Neural Network (hidden_layers={config.hidden_layer_sizes}, "
            f"activation={config.activation}, solver={config.solver}, "
            f"max_iter={config.max_iter}, early_stopping={config.early_stopping})..."
        )
        model = MLPClassifier(
            hidden_layer_sizes=config.hidden_layer_sizes,
            activation=config.activation,
            solver=config.solver,
            alpha=config.alpha,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            learning_rate_init=config.learning_rate_init,
            max_iter=config.max_iter,
            random_state=42,
            early_stopping=config.early_stopping,
            validation_fraction=config.validation_fraction,
        )

    elif model_type == "naive_bayes":
        print(
            f"Training Naive Bayes (alpha={config.alpha}, fit_prior={config.fit_prior})..."
        )
        model = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model, X_test, y_test, model_name, training_time, models_dir, output_dir
):
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

    # Save model
    save_model(model, model_name, models_dir)

    # Save classification report
    report_name = f"{model_name}_report"
    save_classification_report(y_test, y_pred, report_name, output_dir)

    return metrics
