# Type-to-token ratios for chapters in _Moby Dick_

## Introduction
In this project, I calculate **type-to-token ratios** (TTRs) for chapters of _Moby Dick_.

TTR reflects lexical diversity of a text. It ranges from 0 to 1. The more lexically diverse the text is, the closer to 1 it will be; the more repetitive a text is, the lower it will be.

## Data
The input data is _Moby Dick_ by Herman Melville. The text file is retrieved from [Project Gutenberg](https://www.gutenberg.org/).

## Procedure
The main script (`main.py`) runs as follows:
1. It reads in the text contents from Moby Dick.
2. Using regular expressions, it splits the contents into chapters and removes preface and licensing information.
3. Then, for each chapter: 
   1. The chapter is tokenized with the _Punkt_ tokenizer (Kiss and Strunk 2006) as [implemented in the Natural Language Toolkit](https://www.nltk.org/api/nltk.tokenize.punkt.html) (NLTK, Bird et al. 2009).
   2. It calculates TTR for raw tokens, lowercased tokens and for stemmed tokens, respectively. Stemming is done with the Porter stemmer, as [implemented in NLTK](https://www.nltk.org/_modules/nltk/stem/porter.html). 
4. Finally, all TTRs are output in a `.csv` file.

## References
Bird, S., E. Loper and E. Klein (2009). _Natural Language Processing with Python_. O’Reilly Media Inc.

Kiss, T. and J. Strunk (2006). "Unsupervised Multilingual Sentence
  Boundary Detection". _Computational Linguistics_ 32: 485-525.

Porter, M. (1980). "An algorithm for suffix stripping". _Program_ 14.3: 130-137.