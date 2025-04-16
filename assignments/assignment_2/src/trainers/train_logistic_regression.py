#!/usr/bin/env python3
"""
Script to train a logistic regression classifier on the Fake News dataset.
This script uses settings from the central configuration.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from settings import settings
from utils.logger import logger
from utils.trainer_utils import train_and_evaluate_model


def train_logistic_regression(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> dict:
    """
    Function to train and evaluate the logistic regression model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
        
    Returns:
    --------
    dict
        Dictionary of model performance metrics
    """
    # Get configuration from settings
    lr_config = settings.models.logistic_regression
    
    # Create model info string for logging
    model_info = f"Training Logistic Regression (C={lr_config.c_value}, max_iter={lr_config.max_iter}, solver={lr_config.solver})..."
    
    # Define model factory function that uses dict expansion
    def create_model():
        return LogisticRegression(
            random_state=settings.models.random_state,
            **lr_config.dict(exclude={"name", "enabled"})
        )
    
    # Use the shared train and evaluate function
    metrics = train_and_evaluate_model(
        model_factory=create_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=lr_config,
        model_info=model_info
    )
    
    return metrics


if __name__ == "__main__":
    train_logistic_regression()
