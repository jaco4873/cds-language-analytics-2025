"""
Visualization utilities for text classification benchmarks.
This module contains functions to create and save visualizations of model performance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import logger
from settings import settings

# Set style for plots
sns.set_style("whitegrid")


def create_visualizations(df, figures_dir=None) -> None:
    """
    Create and save visualizations comparing model performance.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model metrics
    figures_dir : str
        Directory to save visualizations
    """
    if figures_dir is None:
        figures_dir = settings.output.figures_dir

    logger.info(f"Creating visualizations in {figures_dir}...")

    create_accuracy_comparison(df, figures_dir)
    create_f1_comparison(df, figures_dir)
    create_precision_recall_comparison(df, figures_dir)

    logger.info(f"Visualizations saved to {figures_dir}")


def create_barplot(
    data,
    x,
    y,
    title,
    xlabel,
    ylabel,
    filename,
    figures_dir,
    hue=None,
    palette=None,
    figsize=(10, 6),
    ylim=(0.80, 0.96),
) -> None:
    """Helper function to create and save bar plots with consistent styling."""
    plt.figure(figsize=figsize)

    ax = sns.barplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        palette=palette,
        legend=False if hue is None else True,
    )

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylim(*ylim)
    if hue is not None:
        plt.legend(title=hue)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300)
    plt.close()


def prepare_class_data(df, value_cols) -> pd.DataFrame:
    """Helper function to prepare data for class-based comparisons."""
    data = {"Model": [], "Class": [], "Value": []}

    for col in value_cols:
        class_name = "REAL" if "real" in col else "FAKE"
        data["Model"].extend(df["description"].tolist())
        data["Class"].extend([class_name] * len(df))
        data["Value"].extend(df[col].tolist())

    return pd.DataFrame(data)


def create_accuracy_comparison(df, figures_dir) -> None:
    """Create and save accuracy comparison chart."""
    create_barplot(
        data=df.copy(),
        x="description",
        y="accuracy",
        title="Model Accuracy Comparison",
        xlabel="Model",
        ylabel="Accuracy",
        filename="accuracy_comparison.png",
        figures_dir=figures_dir,
        hue="description",
    )


def create_f1_comparison(df, figures_dir) -> None:
    """Create and save F1-score comparison chart."""
    f1_df = prepare_class_data(df, ["real_f1", "fake_f1"])
    f1_df = f1_df.rename(columns={"Value": "F1-Score"})

    create_barplot(
        data=f1_df,
        x="Model",
        y="F1-Score",
        title="F1-Score Comparison by Class",
        xlabel="Model",
        ylabel="F1-Score",
        filename="f1_score_comparison.png",
        figures_dir=figures_dir,
        hue="Class",
        palette="Set2",
        figsize=(12, 6),
    )


def create_precision_recall_comparison(df, figures_dir) -> None:
    """Create and save precision-recall comparison chart."""
    plt.figure(figsize=(14, 10))

    # Create data for both plots
    precision_df = prepare_class_data(df, ["real_precision", "fake_precision"])
    precision_df = precision_df.rename(columns={"Value": "Precision"})

    recall_df = prepare_class_data(df, ["real_recall", "fake_recall"])
    recall_df = recall_df.rename(columns={"Value": "Recall"})

    # Plot both charts using the same function
    for idx, (plot_data, metric, title) in enumerate(
        [
            (precision_df, "Precision", "Precision Comparison by Class"),
            (recall_df, "Recall", "Recall Comparison by Class"),
        ]
    ):
        plt.subplot(2, 1, idx + 1)
        ax = sns.barplot(
            x="Model", y=metric, hue="Class", data=plot_data, palette="Set1"
        )

        plt.title(title, fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel(metric, fontsize=14)

        # Add value labels
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.ylim(0.80, 0.96)
        plt.legend(title="Class")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "precision_recall_comparison.png"), dpi=300)
    plt.close()
