"""
Assignment 1: Extracting linguistic features using spaCy

This script processes text files from the Uppsala Student English Corpus (USE),
extracts linguistic features using spaCy, and saves the results as CSV files.

Features extracted:
- Relative frequency of Nouns, Verbs, Adjectives, and Adverbs per 100 words
- Total number of unique PER, LOC, and ORG entities
"""

import os
import re
import csv
import glob
import spacy
import logging
import argparse
from pathlib import Path
from collections import Counter


def clean_text(text: str) -> str:
    """
    Clean the text by removing metadata enclosed in angle brackets.

    Args:
        text (str): The raw text from the file

    Returns:
        str: The cleaned text with metadata removed
    """
    # Remove metadata in angle brackets
    text = re.sub(r"<.*?>", "", text)

    return text.strip()


def read_file(file_path: str) -> str:
    """
    Read a file with appropriate encoding.

    Args:
        file_path (str): Path to the text file

    Returns:
        str: The content of the file
    """
    # Attempt system default encoding
    try:
        with open(file_path, "r") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
    
    return clean_text(text)


def extract_features(doc: spacy.tokens.Doc) -> dict:
    """
    Extract linguistic features from a spaCy Doc object.

    Args:
        doc (spacy.tokens.Doc): Processed spaCy document

    Returns:
        dict: Dictionary containing the extracted features
    """
    # Count total tokens (excluding punctuation and whitespace)
    total_tokens = len(
        [token for token in doc if not token.is_punct and not token.is_space]
    )

    # Count POS tags
    pos_counts = Counter()
    for token in doc:
        if not token.is_punct and not token.is_space:
            pos_counts[token.pos_] += 1

    logging.debug(f"Total tokens: {total_tokens}")
    logging.debug(f"POS counts: {pos_counts}")

    # Calculate relative frequencies per 100 words
    rel_freq_noun = (pos_counts["NOUN"] / total_tokens) * 100 if "NOUN" in pos_counts else 0
    rel_freq_verb = (pos_counts["VERB"] / total_tokens) * 100 if "VERB" in pos_counts else 0
    rel_freq_adj = (pos_counts["ADJ"] / total_tokens) * 100 if "ADJ" in pos_counts else 0
    rel_freq_adv = (pos_counts["ADV"] / total_tokens) * 100 if "ADV" in pos_counts else 0

    # Extract unique named entities
    unique_per = set()
    unique_loc = set()
    unique_org = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            unique_per.add(ent.text.lower())
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            unique_loc.add(ent.text.lower())
        elif ent.label_ == "ORG":
            unique_org.add(ent.text.lower())

    # Return the results
    return {
        "RelFreq NOUN": round(rel_freq_noun, 2),
        "RelFreq VERB": round(rel_freq_verb, 2),
        "RelFreq ADJ": round(rel_freq_adj, 2),
        "RelFreq ADV": round(rel_freq_adv, 2),
        "No. Unique PER": len(unique_per),
        "No. Unique LOC": len(unique_loc),
        "No. Unique ORG": len(unique_org),
    }


def process_folder(folder_path: str, output_dir: str, nlp: spacy.Language) -> None:
    """
    Process all text files in a folder and save results to a CSV file.

    Args:
        folder_path (str): Path to the folder containing text files
        output_dir (str): Directory to save the output CSV file
        nlp (spacy.Language): Loaded spaCy model

    Returns:
        None
    """
    # Get the folder name
    folder_path = folder_path.rstrip("/\\")
    folder_name = os.path.basename(folder_path)

    # Get all text files in the folder
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))
    logging.info(f"Processing folder: {folder_name} with {len(text_files)} files")

    # Read all files and prepare for processing
    texts = []
    filenames = []
    for file_path in text_files:
        filenames.append(os.path.basename(file_path))
        logging.debug(f"Reading {file_path}")
        texts.append(read_file(file_path))

    # Process all texts in batch using pipe()
    results = []
    for filename, doc in zip(filenames, nlp.pipe(texts)):
        logging.debug(f"Extracting features from {filename}")
        features = extract_features(doc)
        features["Filename"] = filename
        results.append(features)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    output_file = os.path.join(output_dir, f"{folder_name}.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Filename",
            "RelFreq NOUN",
            "RelFreq VERB",
            "RelFreq ADJ",
            "RelFreq ADV",
            "No. Unique PER",
            "No. Unique LOC",
            "No. Unique ORG",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"Results saved to {output_file}")


def main():
    """
    Main function to process all folders in the corpus.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract linguistic features using spaCy")
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    parser.add_argument(
        "--corpus-dir", 
        type=str, 
        default="data/use-corpus/USEcorpus",
        help="Path to the corpus directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Path to the output directory"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load spaCy model with optimized pipeline
    # Disable components we don't need (parser, lemmatizer)
    logging.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_md", disable=["parser", "lemmatizer"])
    logging.info(f"Active pipeline components: {nlp.pipe_names}")

    # Define paths
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)

    # Get all subdirectories
    subdirs = glob.glob(os.path.join(corpus_dir, "*/"))

    # Process each subdirectory
    for subdir in subdirs:
        process_folder(subdir, output_dir, nlp)

    logging.info("All folders processed successfully!")


if __name__ == "__main__":
    main()
