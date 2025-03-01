# Assignment 1: Extracting Linguistic Features using spaCy

This project analyzes the Uppsala Student English Corpus (USE) using spaCy to extract linguistic features from text files. It processes multiple folders of text files and generates CSV reports with linguistic statistics.

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

## Quickstart

To set up the project environment and run the analysis:

```bash
./setup.sh
```

This script will:
1. Create a virtual environment
2. Install all required dependencies
3. Download the necessary spaCy language model
4. Finally, it will prompt whether you would like to run the analysis.

**Running the analysis manually**

The analysis can also be run directly via:

```bash
./run.sh
```

The script will:
1. Process all text files in each subfolder of the corpus (a1, a2, a3, etc.)
2. Extract the linguistic features
3. Save a CSV file for each subfolder in the `output` directory

## Analysis Details

For each text file, the script extracts:

1. **Relative Frequencies (per 100 words)**:
   - Nouns (NOUN)
   - Verbs (VERB)
   - Adjectives (ADJ)
   - Adverbs (ADV)

2. **Named Entity Counts**:
   - Number of unique Person entities (PER)
   - Number of unique Location entities (LOC)
   - Number of unique Organization entities (ORG)

### Output

The script generates one CSV file per subfolder in the `output` directory (14 files total). Each CSV file contains a table with the following columns:

| Filename | RelFreq NOUN | RelFreq VERB | RelFreq ADJ | RelFreq ADV | No. Unique PER | No. Unique LOC | No. Unique ORG |
|----------|-------------|-------------|------------|------------|---------------|--------------|--------------|
| file1.txt | 25.3 | 18.7 | 10.2 | 5.6 | 3 | 2 | 1 |
| file2.txt | 22.1 | 19.5 | 9.8 | 6.2 | 5 | 4 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Implementation Details

- The script removes metadata enclosed in angle brackets (`<>`) from the text files before processing.
- Named entities are counted as unique based on their lowercase text to avoid duplicates due to capitalization differences.
- GPE (Geo-Political Entity) entities are included in the LOC (Location) count.
- Relative frequencies are calculated per 100 words and rounded to 2 decimal places.

## Requirements

- Python 3.12 or higher
- spaCy
- pip (for downloading the spaCy model)
- en_core_web_md model for spaCy

## Author

Jacob Lillelund

## References

- Uppsala Student English Corpus (USE). Department of English, Uppsala University.