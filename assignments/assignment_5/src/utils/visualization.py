"""
Visualization tools for IMDb sentiment analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset
from typing import Any
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ..settings import settings
from .logger import logger


def plot_metrics_comparison(
    baseline_metrics: dict[str, float],
    transformer_metrics: dict[str, float],
    output_path: Path,
) -> None:
    """
    Create a bar chart comparing baseline and transformer model metrics.

    Args:
        baseline_metrics: Metrics from baseline model
        transformer_metrics: Metrics from transformer model
        output_path: Path to save the figure
    """
    metrics = ["accuracy", "precision", "recall", "f1"]

    plt.figure(figsize=(10, 6))

    # Set up bar positions
    x = np.array(range(len(metrics)))
    width = 0.35

    # Get metric values
    baseline_values = [baseline_metrics.get(m, 0) for m in metrics]
    transformer_values = [transformer_metrics.get(m, 0) for m in metrics]

    # Plot bars
    plt.bar(x - width / 2, baseline_values, width, label="TF-IDF + Logistic Regression")
    plt.bar(x + width / 2, transformer_values, width, label="DistilBERT")

    plt.ylabel("Score")
    plt.title("Performance Comparison: IMDb Sentiment Analysis", fontsize=15)
    plt.xticks(x, metrics)
    plt.ylim(0, 1)

    # Add value labels
    for i, v in enumerate(baseline_values):
        plt.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center")

    for i, v in enumerate(transformer_values):
        plt.text(i + width / 2, v + 0.01, f"{v:.2f}", ha="center")

    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.PLOT_DPI)
    plt.close()

    logger.info(f"Saved metrics comparison: {output_path}")


def plot_review_length_performance(
    test_dataset: Dataset,
    baseline_preds: list[int],
    transformer_preds: list[int],
    output_path: Path,
) -> None:
    """
    Create a plot showing model performance by review length.

    Args:
        test_dataset: Test data
        baseline_preds: Predictions from baseline model
        transformer_preds: Predictions from transformer model
        output_path: Path to save the figure
    """
    logger.info("Analyzing performance by review length...")

    # Calculate review lengths
    reviews = test_dataset["review"]
    sentiments = test_dataset["sentiment"]
    review_lengths = [len(review) for review in reviews]

    # Create correctness lists
    baseline_correct = [pred == true for pred, true in zip(baseline_preds, sentiments)]
    transformer_correct = [
        pred == true for pred, true in zip(transformer_preds, sentiments)
    ]

    # Define length bins
    bins = [0, 100, 200, 300, 500, 1000, float("inf")]
    bin_labels = ["0-100", "101-200", "201-300", "301-500", "501-1000", "1000+"]

    # Create bin assignments
    bin_assignments = []
    for length in review_lengths:
        for i, upper in enumerate(bins[1:]):
            if length < upper:
                bin_assignments.append(bin_labels[i])
                break
        else:
            bin_assignments.append(bin_labels[-1])

    # Group by bins
    bin_data = {}
    for bin_label in bin_labels:
        bin_data[bin_label] = {
            "baseline_correct": [],
            "transformer_correct": [],
            "count": 0,
        }

    for bin_assignment, b_correct, t_correct in zip(
        bin_assignments, baseline_correct, transformer_correct
    ):
        bin_data[bin_assignment]["baseline_correct"].append(b_correct)
        bin_data[bin_assignment]["transformer_correct"].append(t_correct)
        bin_data[bin_assignment]["count"] += 1

    # Calculate accuracies
    bin_categories = []
    baseline_acc = []
    transformer_acc = []
    counts = []

    for bin_label in bin_labels:
        if bin_data[bin_label]["count"] > 0:
            bin_categories.append(bin_label)
            baseline_acc.append(
                sum(bin_data[bin_label]["baseline_correct"])
                / bin_data[bin_label]["count"]
            )
            transformer_acc.append(
                sum(bin_data[bin_label]["transformer_correct"])
                / bin_data[bin_label]["count"]
            )
            counts.append(bin_data[bin_label]["count"])

    x = list(range(len(bin_categories)))
    width = 0.35

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], baseline_acc, width, label="TF-IDF + LogReg")
    plt.bar([i + width / 2 for i in x], transformer_acc, width, label="DistilBERT")

    # Add count labels
    for i, count in enumerate(counts):
        plt.text(i, 0.05, f"n={count}", ha="center", va="bottom", color="black")

    plt.xlabel("Review Length")
    plt.ylabel("Accuracy")
    plt.title("Performance by Review Length", fontsize=15)
    plt.xticks(x, bin_categories)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.PLOT_DPI)
    plt.close()

    logger.info(f"Saved review length performance plot to {output_path}")


def plot_confusion_matrices_side_by_side(
    true_labels: list[int],
    logistic_preds: list[int],
    transformer_preds: list[int],
    figures_dir: Path,
) -> None:
    """
    Create side-by-side confusion matrices for both models in a single figure.

    Args:
        true_labels: True sentiment labels
        logistic_preds: Logistic regression model predictions
        transformer_preds: Transformer model predictions
        figures_dir: Directory to save figures
    """
    # Calculate confusion matrices
    cm_lr = confusion_matrix(true_labels, logistic_preds)
    cm_transformer = confusion_matrix(true_labels, transformer_preds)

    # Normalize
    cm_lr = cm_lr.astype("float") / cm_lr.sum(axis=1)[:, np.newaxis]
    cm_transformer = (
        cm_transformer.astype("float") / cm_transformer.sum(axis=1)[:, np.newaxis]
    )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot logistic regression confusion matrix
    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm_lr, display_labels=["Negative", "Positive"]
    )
    disp1.plot(ax=ax1, cmap="Blues", values_format=".2f")
    ax1.set_title("Logistic Regression Confusion Matrix", fontsize=15)

    # Plot transformer confusion matrix
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm_transformer, display_labels=["Negative", "Positive"]
    )
    disp2.plot(ax=ax2, cmap="Blues", values_format=".2f")
    ax2.set_title("DistilBERT Confusion Matrix", fontsize=15)

    # Save
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=settings.PLOT_DPI)
    plt.close()

    logger.info("Saved side-by-side confusion matrices")


def visualize_results(
    results: dict[str, Any], output_dir: Path, test_dataset: Dataset | None = None
) -> None:
    """
    Orchestrate the creation of all visualization plots.

    Args:
        results: Dictionary with model metrics and predictions
        output_dir: Directory to save output
        test_dataset: Test dataset
    """
    logger.info("Creating visualization plots")

    figures_dir = output_dir / "figures"

    # Compare models if we have both
    if "logistic_regression" in results and "transformer" in results:
        # Metrics comparison
        plot_metrics_comparison(
            baseline_metrics=results["logistic_regression"],
            transformer_metrics=results["transformer"],
            output_path=figures_dir / "metrics_comparison.png",
        )

        # Additional visualizations when predictions are available
        if (
            test_dataset is not None
            and "logistic_regression_preds" in results
            and "transformer_preds" in results
        ):
            # Review length performance
            plot_review_length_performance(
                test_dataset=test_dataset,
                baseline_preds=results["logistic_regression_preds"],
                transformer_preds=results["transformer_preds"],
                output_path=figures_dir / "review_length_performance.png",
            )

            # Side-by-side confusion matrices
            plot_confusion_matrices_side_by_side(
                true_labels=test_dataset["sentiment"],
                logistic_preds=results["logistic_regression_preds"],
                transformer_preds=results["transformer_preds"],
                figures_dir=figures_dir,
            )

        logger.info("Created comparison plots")

    logger.info(f"All plots saved to {figures_dir}")
