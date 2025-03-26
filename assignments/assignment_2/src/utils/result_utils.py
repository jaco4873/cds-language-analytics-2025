import pandas as pd
import os

from utils.logger import logger


def create_comparison_table(results, results_dir) -> pd.DataFrame:
    """
    Create and save a table comparing model performance.

    Parameters:
    -----------
    results : list of dict
        List of dictionaries containing model metrics
    results_dir : str
        Directory to save the comparison table

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the comparison table
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Reorder columns for better display
    display_cols = [
        "model",
        "description",
        "accuracy",
        "training_time",
        "real_precision",
        "real_recall",
        "real_f1",
        "fake_precision",
        "fake_recall",
        "fake_f1",
    ]

    # Rename columns for better display
    rename_map = {
        "model": "Model",
        "description": "Description",
        "real_precision": "REAL Precision",
        "real_recall": "REAL Recall",
        "real_f1": "REAL F1",
        "fake_precision": "FAKE Precision",
        "fake_recall": "FAKE Recall",
        "fake_f1": "FAKE F1",
        "accuracy": "Accuracy",
        "training_time": "Training Time (s)",
    }

    # Create and display comparison table
    comparison_table = df[display_cols].rename(columns=rename_map).set_index("Model")

    # Log the table
    logger.info("Model Performance Comparison")
    logger.info("\n" + comparison_table.round(4).to_string())

    # Save the table to CSV
    output_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_table.to_csv(output_path)
    logger.info(f"Comparison saved to {output_path}")

    return df
