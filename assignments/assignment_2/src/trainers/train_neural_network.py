#!/usr/bin/env python3
"""
Script to train a neural network classifier on the Fake News dataset.
This script uses settings from the central configuration.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from settings import settings
from utils.trainer_utils import train_and_evaluate_model


def train_neural_network(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> dict:
    """
    Function to train and evaluate the neural network model.
    
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
    nn_config = settings.models.neural_network
    
    # Get the hidden layer sizes directly from settings
    hidden_layer_sizes = tuple(nn_config.hidden_layer_sizes)
    
    # Create model info string for logging
    model_info = (
        f"Training neural network with architecture: {hidden_layer_sizes}\n"
        f"Parameters: alpha={nn_config.alpha}, max_iter={nn_config.max_iter}, "
        f"learning_rate_init={nn_config.learning_rate_init}"
    )
    
    # Define model factory function
    def create_model():
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=settings.models.random_state,
            **nn_config.dict(exclude={"name", "enabled", "hidden_layer_sizes"})
        )
    
    # Use the shared train and evaluate function
    metrics = train_and_evaluate_model(
        model_factory=create_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=nn_config,
        model_info=model_info
    )
    
    return metrics


if __name__ == "__main__":
    train_neural_network()
