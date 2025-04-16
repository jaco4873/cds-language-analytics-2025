"""Visualization Module

This module provides functions for visualizing the results of topic modeling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
from typing import Any
from pathlib import Path
from bertopic import BERTopic
from src.config.settings import settings


def save_topic_wordclouds(model: BERTopic, output_path: Path) -> Path:
    """Save topic word clouds visualization.
    
    Args:
        model: Fitted BERTopic model
        output_path: Directory to save the visualization
        
    Returns:
        Path to the saved visualization file
    """
    fig = model.visualize_topics()
    output_file = output_path / "topic_wordclouds.html"
    fig.write_html(str(output_file))
    return output_file


def save_topic_hierarchy(model: BERTopic, output_path: Path) -> Path:
    """Save topic hierarchy visualization.
    
    Args:
        model: Fitted BERTopic model
        output_path: Directory to save the visualization
        
    Returns:
        Path to the saved visualization file
    """
    fig = model.visualize_hierarchy()
    output_file = output_path / "topic_hierarchy.html"
    fig.write_html(str(output_file))
    return output_file


def save_topic_barchart(model: BERTopic, output_path: Path) -> Path:
    """Save topic bar chart visualization.
    
    Args:
        model: Fitted BERTopic model
        output_path: Directory to save the visualization
        
    Returns:
        Path to the saved visualization file
    """
    fig = model.visualize_barchart(top_n_topics=20)
    output_file = output_path / "topic_barchart.html"
    fig.write_html(str(output_file))
    return output_file


def create_topic_category_heatmap(data: list[dict[str, Any]], output_path: Path) -> Path:
    """Create a heatmap showing the distribution of topics per category.
    
    Args:
        data: Data with topic assignments
        output_path: Directory to save the visualization
        
    Returns:
        Path to the saved visualization file
    """
    # Create DataFrame for analysis
    df = pd.DataFrame(data)
    
    # Create cross-tabulation of topic vs category
    # For each topic, show distribution across categories (normalize by topic)
    ct = pd.crosstab(df["topic"], df["category"], normalize="index")
    
    # Get all unique topics and sort them
    topics = sorted(list(ct.index))
    
    # Group topics into chunks of 17 (for better readability)
    max_topics_per_facet = 17
    topic_chunks = [topics[i:i + max_topics_per_facet] for i in range(0, len(topics), max_topics_per_facet)]
    
    # Determine the number of facets needed
    n_facets = len(topic_chunks)
    
    # Calculate grid dimensions
    n_cols = min(3, math.ceil(math.sqrt(n_facets)))  
    n_rows = math.ceil(n_facets / n_cols)
    
    # Set the colormap 
    colormap = 'viridis' 
    
    # Base size for a single heatmap
    base_size = 8
    
    # Create figure with extra space at bottom for colorbar
    fig = plt.figure(figsize=(base_size * n_cols, base_size * n_rows + 1))
    
    # Create a GridSpec to have control over the layout
    gs = plt.GridSpec(n_rows + 1, n_cols, height_ratios=[*[1] * n_rows, 0.1])
    
    # Store all heatmap objects to use for colorbar
    heatmaps = []
    
    # Track min and max values for consistent color scaling
    vmin, vmax = 0, 1
    
    # For each chunk of topics, create a heatmap in the corresponding subplot
    for i, topic_chunk in enumerate(topic_chunks):
        if i < n_rows * n_cols:
            # Calculate row and column
            row = i // n_cols
            col = i % n_cols
            
            # Create subplot at the specific position
            ax = fig.add_subplot(gs[row, col])
            
            # Extract the subset of the crosstab for this chunk of topics
            ct_subset = ct.loc[topic_chunk]
            
            # Create heatmap without colorbar
            hm = sns.heatmap(
                ct_subset, 
                cmap=colormap, 
                annot=False, 
                ax=ax, 
                cbar=False,  # No individual colorbars
                robust=True   # Use robust quantile range to improve color contrast
            )
            heatmaps.append(hm)
            
            # Add title and labels
            ax.set_title(f"Topics {topic_chunk[0]} to {topic_chunk[-1]}", 
                        fontsize=settings.FONT_SIZE)
            ax.set_xlabel("Category", fontsize=settings.FONT_SIZE - 2)
            ax.set_ylabel("Topic", fontsize=settings.FONT_SIZE - 2)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", 
                    fontsize=settings.FONT_SIZE - 4)
            
            # Make y-axis labels horizontal
            plt.setp(ax.get_yticklabels(), rotation=0, 
                    fontsize=settings.FONT_SIZE - 4)
    
    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    # Add a shared colorbar at the bottom
    colorbar_ax = fig.add_subplot(gs[-1, :])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=colorbar_ax, orientation='horizontal')
    cbar.set_label('Topic Distribution (proportion)', fontsize=settings.FONT_SIZE)
    
    # Add a main title for the entire figure
    fig.suptitle("Topic Distribution Across Categories", 
                fontsize=settings.FONT_SIZE + 4, y=0.98)
    
    # Adjust layout with more space between subplots
    plt.tight_layout()
    fig.subplots_adjust(top=0.95, wspace=0.3, hspace=0.4, bottom=0.1)
    
    # Save the figure
    output_file = output_path / "topic_category_heatmap.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    return output_file


def create_topic_prevalence_per_category(
    data: list[dict[str, Any]],
    topic_info: dict[int, dict[str, Any]],
    category: str,
    n_topics: int = 10,
    output_path: Path | None = None
) -> Path | None:
    """Create a bar chart showing the most prevalent topics for a specific category.
    
    Args:
        data: Data with topic assignments
        topic_info: Dictionary mapping topic ID to topic information
        category: Category to analyze
        n_topics: Number of top topics to show
        output_path: Directory to save the visualization, if None, just displays the plot
        
    Returns:
        Path to the saved visualization file, if output_path is provided
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter for the selected category
    category_df = df[df["category"] == category]
    
    # Count topics in this category
    topic_counts = category_df["topic"].value_counts().reset_index()
    topic_counts.columns = ["topic", "count"]
    
    # Sort by count and take top N
    topic_counts = topic_counts.sort_values("count", ascending=False).head(n_topics)
    
    # Add topic name from topic_info
    def get_topic_name(topic_id):
        if topic_id == -1:
            return "Outlier"
        return topic_info[topic_id]["Name"] if topic_id in topic_info else f"Topic {topic_id}"
    
    topic_counts["name"] = topic_counts["topic"].apply(get_topic_name)
    
    # Create plot
    plt.figure(figsize=(settings.FIGSIZE_WIDTH, settings.FIGSIZE_HEIGHT // 2))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(topic_counts)))
    ax = sns.barplot(x="count", y="name", data=topic_counts, palette=colors)
    
    # Add count labels to bars
    for i, count in enumerate(topic_counts["count"]):
        ax.text(count + 1, i, str(count), va="center")
    
    plt.title(f"Top {n_topics} Topics in Category: {category}", fontsize=settings.FONT_SIZE + 2)
    plt.xlabel("Document Count", fontsize=settings.FONT_SIZE)
    plt.ylabel("Topic", fontsize=settings.FONT_SIZE)
    plt.tight_layout()
    
    # Save
    if output_path:
        filename = f"topic_prevalence_{category.lower().replace(' ', '_')}.png"
        output_file = output_path / filename
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        return output_file
    else:
        plt.show()
        plt.close()
        return None


def create_category_prevalence_per_topic(
    data: list[dict[str, Any]],
    topic_id: int,
    topic_name: str,
    n_categories: int = 7,
    output_path: Path | None = None
) -> Path | None:
    """Create a bar chart showing the most prevalent categories for a specific topic.
    
    Args:
        data: Data with topic assignments
        topic_id: Topic ID to analyze
        topic_name: Name of the topic
        n_categories: Number of top categories to show
        output_path: Directory to save the visualization, if None, just displays the plot
        
    Returns:
        Path to the saved visualization file, if output_path is provided
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter for the selected topic
    topic_df = df[df["topic"] == topic_id]
    
    # Count categories in this topic
    category_counts = topic_df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]
    
    # Sort by count and take top N
    category_counts = category_counts.sort_values("count", ascending=False).head(n_categories)
    
    # Create plot
    plt.figure(figsize=(settings.FIGSIZE_WIDTH, settings.FIGSIZE_HEIGHT // 2))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(category_counts)))
    ax = sns.barplot(x="count", y="category", data=category_counts, palette=colors)
    
    # Add count labels to bars
    for i, count in enumerate(category_counts["count"]):
        ax.text(count + 1, i, str(count), va="center")
    
    plt.title(f"Categories Count in Topic: {topic_name}", fontsize=settings.FONT_SIZE + 2)
    plt.xlabel("Document Count", fontsize=settings.FONT_SIZE)
    plt.ylabel("Category", fontsize=settings.FONT_SIZE)
    plt.tight_layout()
    
    # Save
    if output_path:
        topic_name_safe = str(topic_id) if topic_id == -1 else topic_name.split("_")[0]
        filename = f"category_prevalence_topic_{topic_name_safe}.png"
        output_file = output_path / filename
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        return output_file
    else:
        plt.show()
        plt.close()
        return None


def create_topics_per_class_plot(model: BERTopic, docs: list[str], classes: list[str], output_path: Path) -> Path:
    """Create a topics per class visualization.
    
    Args:
        model: Fitted BERTopic model
        docs: List of document texts
        classes: List of class labels for each document
        output_path: Directory to save the visualization
        
    Returns:
        Path to the saved visualization file
    """
    # Calculate topics per class
    topics_per_class = model.topics_per_class(docs, classes=classes)
    
    # Visualize
    fig = model.visualize_topics_per_class(topics_per_class)
    
    # Save visualization
    output_file = output_path / "topics_per_class.html"
    fig.write_html(str(output_file))
    
    return output_file 