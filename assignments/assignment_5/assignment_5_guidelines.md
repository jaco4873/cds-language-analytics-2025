# Assignment 5 - Mini Project

## Instructions

We have seen how pretrained language models can be used for a wide range of tasks. In this assignment, we're going to use these models to perform some computational text analysis.

This assignment is a more open one. It allows you to draw on expertise from your respective background fields in a way that (hopefully) excites you.

Your code should produce interpretable output, like plots or tables. Furthermore, your readme should include a (very) short report of your analysis which includes the output and addresses a simple overarching research question.

## Requirements and Limitations

While the assignment is quite open, it is not entirely open. Below are requirements and limitations that you should adhere to.

- Your analysis must include use of at least one huggingface-transformer model. You can supplement this however you like, e.g. vibes analysis with n-grams, topic analysis, alternative classifier models or even prompt-based approaches.
- The data should be of a reasonable size for a mini project like this. This is open to interpretation, but aim for something like a 50-100k word corpus (5-10k headlines or 1-2k short documents). Consider creating a subset of a dataset if it is too big. Document how the subset was created, if possible, e.g. if you filter by certain criteria.
- Create a 1-3 page report (incl. figures/tables)*. I suggest that you structure it according to the IMRAD structure. Briefly describe your code in a Methods section.
- Create 1-5 tables or figures to include in your report.
- The submission should include all output, but should still be runnable from end to end.

\* Following AU standards, with 2400 key strokes for a page and each figure counting as 800.

If you are in doubt whether your idea meets the requirements, ask and get it approved. :-)

## Examples

| Data | Huggingface model task | RQ/hypothesis | Output |
|------|------------------------|---------------|--------|
| Game of Thrones subtitles | Emotion detection | How does the emotional profile change over GoT seasons? | Distribution of emotion in whole series<br>Proportion of emotions by season |
| News Corpus Dataset* | Named Entity Recognition | Specific named entities tend to appear within one news category because they are mostly significant in one domain. | Top named entities per category<br>Purity of named entities and categories |
| Twitter Sentiment | Sentiment analysis | How well does a pre-trained sentiment analysis model perform compared to more traditional models trained on the specific data? | Performance reports<br>Confusion matrices<br>Recall/precision plots |

\* There is a ready-to-use dataset from assignment 4 on UCloud that you can also use here.

## Objectives

This assignment is designed to test that you can:

- Find and approach a structured dataset that helps you address a research question;
- Make good use of skills and knowledge that you have picked in the course;
- Extract meaningful structured information from unstructured text data;
- Output interpretable results from the extracted information;
- Interpret and contextualize these results from a cultural data science perspective.

## Notes and Hints

- You may find that you have many subtasks as part of your project, e.g. fetching data, data cleaning, NLP analysis and plot creation. It might prove useful to break these into individual scripts that each perform one subtask and saves output which can serve as input for the next.
- This is a mini project. It is good to be ambitious and do something fun, but be mindful of what is realistic to do within the bounds of the assignment.
