import csv
import re
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm

# download models from NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')

stemmer = PorterStemmer()

# read text from Moby Dick; split on chapters
chapter_splitter = re.compile(r'\n{3}CHAPTER \d+. ')
with open('data/moby_dick.txt') as f:
    raw_text = f.read()
    # first, removing license info from the end by splitting and keeping the part up to the split
    without_license = raw_text.split('*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK; OR, THE WHALE ***')[0]
    # removing title, preface etc. up to chapter 1 by slicing
    chapters = chapter_splitter.split(raw_text)[1:]


# calculate type-to-token ratio per chapter for each stemmer type
# calculate for these setups: raw, lowercased, stemmed

# also save data for writing a CSV file
csv_data = [
    ["chapter", "raw", "lowercase", "stemmed"]
]


for i, chapter in enumerate(tqdm(chapters, desc="Processing chapters"), start=1):
    # raw, i.e. unprocessed
    tokens = word_tokenize(chapter)
    types = set(tokens)
    raw_ttr = len(types) / len(tokens)

    # lowercased
    lowercase_tokens = [t.lower() for t in tokens]
    lowercase_types = set(lowercase_tokens)
    lowercase_ttr = len(lowercase_types) / len(lowercase_tokens)

    # stemmed
    stemmed_tokens = [stemmer.stem(t) for t in tokens]
    stemmed_types = set(stemmed_tokens)
    stemmed_ttr = len(stemmed_types) / len(stemmed_tokens)

    csv_data.append(
        [i, raw_ttr, lowercase_ttr, stemmed_ttr]
    )

# write collected data for CSV to a file
with open('output/ttrs.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
