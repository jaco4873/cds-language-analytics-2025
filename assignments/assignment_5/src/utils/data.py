"""
Load and prep IMDb movie reviews dataset.
"""

from datasets import load_dataset, concatenate_datasets, Dataset
from .logger import logger


def load_and_prepare_imdb(
    sample_size: int = 10000,
    val_split: float = 0.1,
    random_state: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare the IMDb dataset using HuggingFace datasets.

    Args:
        sample_size: Maximum number of samples to use
        val_split: Proportion of training data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Loading IMDb dataset from HuggingFace")

    # Load dataset directly using the datasets library
    imdb_dataset = load_dataset("stanfordnlp/imdb")

    # Standardize column names
    imdb_dataset = imdb_dataset.rename_column("text", "review")
    imdb_dataset = imdb_dataset.rename_column("label", "sentiment")

    # Calculate actual sample sizes
    train_sample_size = min(sample_size, len(imdb_dataset["train"]))
    test_sample_size = min(sample_size // 5, len(imdb_dataset["test"]))

    # Sample the datasets with stratification
    train_full = imdb_dataset["train"].shuffle(seed=random_state)
    test_set = (
        imdb_dataset["test"].shuffle(seed=random_state).select(range(test_sample_size))
    )

    # Ensure stratification by sampling equally from each class
    train_pos = train_full.filter(lambda x: x["sentiment"] == 1).select(
        range(train_sample_size // 2)
    )
    train_neg = train_full.filter(lambda x: x["sentiment"] == 0).select(
        range(train_sample_size // 2)
    )

    # Combine positive and negative samples
    train_full = concatenate_datasets([train_pos, train_neg]).shuffle(seed=random_state)

    # Create validation split
    split_dataset = train_full.train_test_split(test_size=val_split, seed=random_state)
    train_set = split_dataset["train"]
    val_set = split_dataset["test"]

    # Log dataset statistics
    logger.info(
        f"Dataset: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )

    # Log sentiment distribution
    train_pos_count = len(train_set.filter(lambda x: x["sentiment"] == 1))
    train_neg_count = len(train_set.filter(lambda x: x["sentiment"] == 0))
    logger.info(
        f"Train sentiment distribution: 0 (negative): {train_neg_count}, 1 (positive): {train_pos_count}"
    )

    return train_set, val_set, test_set
