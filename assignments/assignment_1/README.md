# Assignment 1: Extracting Linguistic Features using spaCy

## Introduction
This project analyzes the Uppsala Student English Corpus (USE) by extracting linguistic features from student texts using spaCy. The analysis focuses on part-of-speech frequencies and named entity recognition to provide quantitative insights into language usage patterns across different corpus sections.

## Data
The Uppsala Student English Corpus (USE) contains texts written by Swedish university students of English. The corpus is organized into 14 subfolders (a1-a5, b1-b8, c1) representing different text categories and student levels. 

## Project Structure

```
.
├── README.md                  # This file
├── assignment_1_guidelines.md # Assignment instructions
├── data/                      # Data directory
│   └── use-corpus/            # Uppsala Student English Corpus
│       ├── USE_data_manual.md # Documentation for the corpus
│       └── USEcorpus/         # The actual corpus files
│           ├── a1/            # Subfolder with text files
│           ├── a2/
│           └── ...
├── output/                    # Output directory for CSV files
├── pyproject.toml             # Project dependencies
├── run.sh                     # Script to run the analysis
├── setup.sh                   # Setup script
├── src/                       # Source code
│   └── main.py                # Main script
└── uv.lock                    # Lock file for dependencies
```

## Getting Started
To set up the project environment and run the analysis:

```bash
./run.sh [LOG_LEVEL]
```
Where `LOG_LEVEL` is optional (DEBUG, INFO, WARNING, ERROR, CRITICAL; default is INFO).

This script will create a virtual environment by invoking `setup.sh`, install dependencies, download the spaCy model, and prompt you to run the analysis.


## Methods
The analysis extracts two types of linguistic features from each text:

1. Relative Frequencies (per 100 words) of:
   - Nouns (NOUN)
   - Verbs (VERB)
   - Adjectives (ADJ)
   - Adverbs (ADV)

2. Named Entity Counts of unique:
   - Person entities (PER)
   - Location entities (LOC), including Geo-Political Entities
   - Organization entities (ORG)

Key implementation details:
- Metadata in angle brackets (`<>`) is removed during preprocessing
- Named entities are counted as unique based on their lowercase form
- Relative frequencies are calculated per 100 words and rounded to 2 decimal places
- Processing is optimized using spaCy's `pipe()` method and disabling unnecessary pipeline components

## Results
The analysis generates 14 CSV files (one per subfolder) in the `output` directory. Each file contains the extracted features for every text in that subfolder:

| Filename | RelFreq NOUN | RelFreq VERB | RelFreq ADJ | RelFreq ADV | No. Unique PER | No. Unique LOC | No. Unique ORG |
|----------|-------------|-------------|------------|------------|---------------|--------------|--------------|
| file1.txt | 25.3 | 18.7 | 10.2 | 5.6 | 3 | 2 | 1 |
| file2.txt | 22.1 | 19.5 | 9.8 | 6.2 | 5 | 4 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... |

These metrics provide quantitative data that could be used for further comparative analysis of writing styles, language proficiency levels, or genre differences across the corpus.

## Requirements

- Python 3.12 or higher
- spaCy
- pip (for downloading the spaCy model)
- en_core_web_md model for spaCy

## Author

Jacob Lillelund

## References

- Uppsala Student English Corpus (USE). Department of English, Uppsala University.