#!/usr/bin/env python3
"""
Assignment 1: Extracting linguistic features using spaCy

This script processes text files from the Uppsala Student English Corpus (USE),
extracts linguistic features using spaCy, and saves the results as CSV files.

Features extracted:
- Relative frequency of Nouns, Verbs, Adjectives, and Adverbs per 100 words
- Total number of unique PER, LOC, and ORG entities

Author: Jacob Lillelund
Date: 2025-03-01
"""

import os
import re
import csv
import glob
import spacy
from pathlib import Path
from collections import Counter


def clean_text(text):
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


def process_file(file_path, nlp):
    """
    Process a single text file to extract linguistic features.

    Args:
        file_path (str): Path to the text file
        nlp (spacy.Language): Loaded spaCy model

    Returns:
        dict: Dictionary containing the extracted features
    """
    # Attempt system default encoding
    try:
        with open(file_path, "r") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
            print(f"Note: File {file_path} read using latin-1 encoding")

    text = clean_text(text)

    # Process the text with spaCy
    doc = nlp(text)

    # Count total tokens (excluding punctuation and whitespace)
    total_tokens = len(
        [token for token in doc if not token.is_punct and not token.is_space]
    )

    # Count POS tags
    pos_counts = Counter()
    for token in doc:
        if not token.is_punct and not token.is_space:
            pos_counts[token.pos_] += 1

    print(f"Total tokens: {total_tokens}")
    print(f"POS counts: {pos_counts}")

    # Calculate relative frequencies per 100 words
    rel_freq_noun = (pos_counts["NOUN"] / total_tokens) * 100
    rel_freq_verb = (pos_counts["VERB"] / total_tokens) * 100
    rel_freq_adj = (pos_counts["ADJ"] / total_tokens) * 100
    rel_freq_adv = (pos_counts["ADV"] / total_tokens) * 100

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

    # Get the filename without path
    filename = os.path.basename(file_path)

    # Return the results
    return {
        "Filename": filename,
        "RelFreq NOUN": round(rel_freq_noun, 2),
        "RelFreq VERB": round(rel_freq_verb, 2),
        "RelFreq ADJ": round(rel_freq_adj, 2),
        "RelFreq ADV": round(rel_freq_adv, 2),
        "No. Unique PER": len(unique_per),
        "No. Unique LOC": len(unique_loc),
        "No. Unique ORG": len(unique_org),
    }


def process_folder(folder_path, output_dir, nlp):
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

    # Process each file
    results = []
    for file_path in text_files:
        print(f"Processing {file_path}...")
        result = process_file(file_path, nlp)
        results.append(result)

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

    print(f"Results saved to {output_file}")


def main():
    """
    Main function to process all folders in the corpus.
    """
    # Load spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_md")

    # Define paths
    corpus_dir = Path("data/use-corpus/USEcorpus")
    output_dir = Path("output")

    # Get all subdirectories
    subdirs = glob.glob(os.path.join(corpus_dir, "*/"))

    # Process each subdirectory
    for subdir in subdirs:
        print(f"\nProcessing folder: {subdir}")
        process_folder(subdir, output_dir, nlp)

    print("\nAll folders processed successfully!")


if __name__ == "__main__":
    main()
