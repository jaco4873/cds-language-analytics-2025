# Assignment 1

## Extracting linguistic features using spaCy

This assignment concerns using spaCy to extract linguistic information from a corpus of texts.

The assignment should be submitted as ZIP file. See Assignment guidelines.

The corpus is an interesting one: The Uppsala Student English Corpus (USE).

All of the data can be found in the shared data drive on UCloud, but you can access more documentation via this link.

## For this exercise, you should write some code which does the following:

1. Loop over folders and text files in the data folder.
2. Extract the following information:
   - Relative frequency of Nouns, Verbs, Adjective, and Adverbs per 100 words - percent, if you will.
   - Total number of unique PER, LOC, and ORG entities
3. For each sub-folder (a1, a2, a3, ...) save a table which shows the following information:

| Filename | RelFreq NOUN | RelFreq VERB | RelFreq ADJ | RelFreq ADV | No. Unique PER | No. Unique LOC | No. Unique ORG |
|----------|-------------|-------------|------------|------------|---------------|--------------|--------------|
| file1.txt |             |             |            |            |               |              |              |
| file2.txt |             |             |            |            |               |              |              |
| file3.txt |             |             |            |            |               |              |              |
| ...      |             |             |            |            |               |              |              |

## Objective

This assignment is designed to test that you can:

- Work with multiple input data arranged hierarchically in folders;
- Use spaCy to extract linguistic information from text data;
- Save those results in a clear way which can be shared or used for future analysis

## Hints

- The data is arranged in various subfolders related to their content - there is a README file with more information. You'll need to think a little bit about how to create an output file per folder. You should be able do it using a combination of things we've already looked at such as `glob.glob()` and for loops. Since the folder structure is nested, you will likely also have to nest your code blocks.
- The text files contain some extra information that such as document ID and other metadata that occurs between pointed brackets `<>`. Make sure to remove these as part of your preprocessing steps.
- There are 14 subfolders (a1, a2, a3, etc), so when completed the output folder should have 14 CSV files. No more, no less. Else, something is wrong.
- If you are unsure where to start, try to think of the task in three levels:
  - **File level**: disregarding that there are many files, try to think about what you need to do for one file.
  - **Subfolder level**: when you know how to do something for one file, expand to do it for multiple files in one folder. Here you will likely need a loop.
  - **Global level**: when you know how to do something for one folder, expand to do it for multiple folders. Here you will likely need another loop.