"""Topic Modeling Module

This module provides a wrapper around BERTopic for topic modeling.
"""

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
import numpy as np
from typing import Any
from src.config.settings import settings


class TopicModelBuilder:
    """Builder for configuring and creating a BERTopic model."""
    
    def __init__(self):
        """Initialize the topic model builder with default settings."""
        self.min_topic_size = settings.MIN_TOPIC_SIZE
        self.umap_model = UMAP(
            n_neighbors=settings.UMAP_N_NEIGHBORS,
            n_components=settings.UMAP_N_COMPONENTS,
            min_dist=settings.UMAP_MIN_DIST,
            random_state=42
        )
        self.vectorizer_model = None
        self.ctfidf_model = None
        self._configure_vectorizer()
        self._configure_ctfidf()
    
    def _configure_vectorizer(self) -> None:
        """Configure the vectorizer model based on settings."""
        if settings.REMOVE_STOPWORDS:
            self.vectorizer_model = CountVectorizer(stop_words="english")
        else:
            self.vectorizer_model = CountVectorizer()
    
    def _configure_ctfidf(self) -> None:
        """Configure the c-TF-IDF model based on settings."""
        self.ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=settings.REDUCE_FREQUENT_WORDS
        )
    
    def build(self) -> BERTopic:
        """Build and return the configured BERTopic model.
        
        Returns:
            Configured BERTopic model
        """
        return BERTopic(
            language="english",
            min_topic_size=self.min_topic_size,
            umap_model=self.umap_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            verbose=True
        )


def create_topic_model() -> BERTopic:
    """Create a topic model with the configured settings.
    
    Returns:
        BERTopic model configured according to settings
    """
    builder = TopicModelBuilder()
    return builder.build()


def fit_topic_model(model: BERTopic, docs: list[str], embeddings: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Fit the topic model to the documents using pre-trained embeddings.
    
    Args:
        model: BERTopic model to fit
        docs: List of document texts
        embeddings: Pre-trained document embeddings
        
    Returns:
        Tuple containing:
        - List of topic assignments for each document
        - Topic probability matrix
    """
    return model.fit_transform(docs, embeddings=embeddings)


def add_topic_info(data: list[dict[str, Any]], topics: list[int]) -> list[dict[str, Any]]:
    """Add topic information to the original data.
    
    Args:
        data: Original data list
        topics: List of topic assignments
        
    Returns:
        Data with topic assignments added
    """
    for i, entry in enumerate(data):
        entry["topic"] = topics[i]
    return data


def get_topic_info(model: BERTopic) -> dict[int, dict[str, Any]]:
    """Get information about topics.
    
    Args:
        model: Fitted BERTopic model
        
    Returns:
        Dictionary mapping topic ID to topic information
    """
    topic_info = model.get_topic_info()
    return {row["Topic"]: row for _, row in topic_info.iterrows()} 