"""Main Module

This module orchestrates the entire topic modeling workflow.
"""

import logging

from src.config.settings import settings
from src.data.loader import prepare_data
from src.models.topic_model import (
    create_topic_model, fit_topic_model, add_topic_info, get_topic_info
)
from src.visualization.plotter import (
    save_topic_wordclouds, save_topic_hierarchy, save_topic_barchart,
    create_topic_category_heatmap, create_topic_prevalence_per_category,
    create_category_prevalence_per_topic, create_topics_per_class_plot
)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def run_topic_modeling() -> None:
    """Run the complete topic modeling workflow."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting topic modeling workflow")
    
    # Prepare data
    logger.info("Preparing data")
    data, headlines, categories, embeddings = prepare_data()
    logger.info(f"Loaded {len(headlines)} documents with {embeddings.shape} embeddings")
    
    # Create and fit topic model
    logger.info("Creating topic model")
    model = create_topic_model()
    
    logger.info("Fitting topic model")
    topics, probs = fit_topic_model(model, headlines, embeddings)
    
    # Process results
    logger.info("Processing results")
    data_with_topics = add_topic_info(data, topics)
    topic_info = get_topic_info(model)
    
    # Get unique categories and topics for analysis
    unique_categories = sorted(set(categories))
    top_topics = sorted(
        [(topic, topic_info[topic]["Count"]) for topic in topic_info if topic != -1],
        key=lambda x: x[1],
        reverse=True
    )[:5]  # Top 5 topics by count
    
    # Create output directory if it doesn't exist
    output_path = settings.OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Save BERTopic's built-in visualizations
    logger.info("Saving topic word clouds")
    save_topic_wordclouds(model, output_path)
    
    logger.info("Saving topic hierarchy")
    save_topic_hierarchy(model, output_path)
    
    logger.info("Saving topic bar chart")
    save_topic_barchart(model, output_path)
    
    # Create custom visualizations
    logger.info("Creating topic category heatmap")
    create_topic_category_heatmap(data_with_topics, output_path)
    
    # Create topics per class visualization
    logger.info("Creating topics per class visualization")
    create_topics_per_class_plot(model, headlines, categories, output_path)
    
    # Generate category-specific visualizations for top categories
    logger.info("Creating category-specific visualizations")
    for category in unique_categories:
        logger.info(f"Analyzing category: {category}")
        create_topic_prevalence_per_category(
            data_with_topics, topic_info, category, output_path=output_path
        )
    
    # Generate topic-specific visualizations for top topics
    logger.info("Creating topic-specific visualizations")
    for topic_id, count in top_topics:
        topic_name = topic_info[topic_id]["Name"]
        logger.info(f"Analyzing topic: {topic_name}")
        create_category_prevalence_per_topic(
            data_with_topics, topic_id, topic_name, output_path=output_path
        )
    
    # Create summary file with basic info
    logger.info("Creating summary file")
    with open(output_path / "analysis_summary.txt", "w") as f:
        f.write("Topic Modeling Analysis Summary\n")
        f.write("==============================\n\n")
        
        f.write(f"Documents analyzed: {len(headlines)}\n")
        f.write(f"Number of unique categories: {len(unique_categories)}\n")
        
        f.write("\nTopics by Document Count:\n")
        for i, (topic_id, topic_name, count) in enumerate(
            [(t, topic_info[t]["Name"], topic_info[t]["Count"]) 
             for t in topic_info]
        ):
            f.write(f"{i+1}. Topic {topic_id}: {topic_name} ({count} documents)\n")
    
    logger.info("Topic modeling workflow completed successfully")


if __name__ == "__main__":
    run_topic_modeling() 