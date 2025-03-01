# Expected Knowledge Document

## Text Processing and Analysis

### 1. Text Loading and Preprocessing
- Load text files from sources like Project Gutenberg
- Split text into chapters using regular expressions
- Perform basic text normalization:
  - Tokenization: breaking text into words/tokens
  - Lowercasing: converting text to lowercase
  - Filtering: removing stopwords, punctuation
  - Stemming/lemmatization: reducing words to base forms

### 2. Text Analysis
- Create frequency distributions using Counter objects
- Compare different preprocessing approaches
- Analyze type/token ratios
- Evaluate information retention across different preprocessing steps
- Extend analysis across chapters and multiple books
- Create reusable functions for text processing tasks

### 3. Regular Expressions
- Create patterns to match specific words and variations (e.g., "Woodchuck/woodchuck/woodchucks")
- Match numeric patterns (decimal numbers, integers, numbers with thousands separators)
- Extract capitalized words and phrases
- Split text into sentences and words using regex
- Handle special cases in text segmentation

### 4. Part-of-Speech Tagging
- Understand the concept of grammatical categories (parts of speech)
- Distinguish between open class words (nouns, verbs, adjectives, adverbs) and closed class words (determiners, pronouns, prepositions)
- Know the Universal Dependencies tagset (17 POS tags)
- Understand POS tagging as a sequence labeling task
- Recognize the challenges of POS tagging (word ambiguity)
- Know that POS tagging accuracy is around 97% for English
- Understand applications of POS tagging (parsing, machine translation, sentiment analysis, text-to-speech)

### 5. Named Entity Recognition (NER)
- Understand named entities: persons, locations, organizations, geo-political entities
- Recognize NER as a span recognition problem
- Know BIO tagging scheme and variants (IO, BIOES)
  - B: beginning of entity
  - I: inside entity
  - O: outside any entity
  - E: end of entity (BIOES)
  - S: single-token entity (BIOES)
- Understand challenges in NER: segmentation and type ambiguity
- Know common approaches to NER using sequence labeling algorithms

### 6. Sequence Labeling Algorithms
- Hidden Markov Models (HMMs)
- Conditional Random Fields (CRFs)
- Maximum Entropy Markov Models (MEMMs)
- Neural sequence models (RNNs, Transformers)
- Large Language Models (like BERT)

### 7. Evaluation Metrics
- Accuracy for POS tagging
- Most frequent class baseline
- Type/token analysis for text processing