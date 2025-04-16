# Assignment 4 - Topic Analysis with BERTopic

## Overview

We have seen how BERTopic serves as a ready-to-use/off-the-shelf solution for performing topic analysis. In this assignment, we're going to use these models to perform computational text analysis of news headlines.

The original dataset (which is obtained from [here](https://www.kaggle.com/rmisra/news-category-dataset)) contains 210k headlines. That's a bit too much for one assignment, so I have created a subset from a handful of select categories which can be found on UCloud as `News_Category_Dataset_v3_subset.jsonl`.

## Requirements

For this assignment, you should write code which does the following:

1. Loads in the data in a way that facilitates later tasks.
2. Create a topic model for the text part(s) of the data. You are free to play around with parameters for your topic model.
3. Associate each document entry in your data with a topic label. 
4. Using plots or any other tool at your disposal, create some interpretable output that illustrates to which degree the BERT topic model fits with the labeled categories. Think about questions like these:
   - For a given category (like "SPORTS"), which topics are most prevalent?
   - For a given popular topic, which categories are most prevalent?
5. Finally, your repository should include a written summary and interpretation of what you think this analysis might being showing. Put it in your readme. You do not need to be a media studies expert or topic wizard here - just describe what you see and what that might mean in this context.

## Objectives

This assignment is designed to test that you can:

- Approach a structured dataset with metadata
- Make good use of external libraries
- Extract meaningful structured information from unstructured text data
- Interpret and contextualize these results from a cultural data science perspective

## Important Update: Using Pretrained Embeddings

Creating embeddings with BERTopic can be very slow and resource-intensive on UCloud. Therefore, **it is strongly recommended to use the pretrained embeddings** that have been made available for both "headlines" and "headlines+short_description".

## Common Issues and Solutions

- If you encounter the error `nameerror: name 'init_empty_weights' is not defined`, you can solve this by running:
  ```
  uv add accelerate
  ```

## Notes and Hints

- You'll need to make use of many of the tools you've learned this semester. This includes - but is not necessarily limited to - libraries such as pandas and matplotlib.

- To save a plot in matplotlib, do something like the following:
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  # create plot
  plot = sns.scatterplot(x=[1, 2, 3], y=[2, 3, 1])

  # save plot
  plt.savefig("saved-plot.png")
  plt.clf() # clear figure, if necessary
  ```

- While trying things out, you may want to avoid retraining your model constantly. Consider splitting up training and plot generation in multiple scripts and saving/loading between them.

- To avoid too many topics which can be hard to get an overview of, you may want to limit the topics by size:
  ```python
  topic_model = BERTopic(language="english",
                         verbose=True,  # to follow progress
                         min_topic_size=50,  # bigger and fewer topics
                         )
  ```

- BERTopic tends to get some stopwords into topic labels other than the -1 noise topic. If that creates noise in your analysis, have a look [here](https://maartengr.github.io/BERTopic/faq.html#how-do-i-remove-stop-words).

- BERTopic has built-in way of looking into topics per class: [Topics Per Class](https://maartengr.github.io/BERTopic/getting_started/topicsperclass/topicsperclass.html)

- A way to provide a full overview of categories and topics is with a heatmap. Assuming that you have categories and topic labels in a dataframe, you can create a heatmap like this:
  ```python
  # Cross-tabulation creates a contingency table between two
  # variables.
  # With normalize="index", a row will show a normalized distribution
  # over column values. In this case, how a topic is distributed across
  # categories.
  ct = pd.crosstab(df["topic"], df["category"], normalize="index")

  # Plot heatmap
  sns.heatmap(ct, cmap="Blues")
  plt.title("Category Overlap Heatmap")

  # Explicitly set y-ticks
  # seaborn wants to hide them if there are many
  plt.yticks(range(len(ct)), ct.index, rotation=0)  

  plt.savefig(
      "../output/overlap-heatmap.png",
      bbox_inches="tight"  # do not crop labels
  )
  ```

- It can be hard to see in detail what is going on, though. You may have to supplement the heatmap with one or more plots or numbers.