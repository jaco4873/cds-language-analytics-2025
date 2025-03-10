# Assignment 2 - Text Classification Benchmarks

This assignment is about using scikit-learn to train binary classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write two different scripts:
1. One should train a logistic regression classifier on the data
2. The second should train a neural network on the same dataset

Both notebooks should do the following:
- Save the classification report to a text file in the folder called `output`
- Save the trained models and vectorizers to the folder called `models`

## Objective

This assignment is designed to test that you can:

1. Train simple machine learning classifiers on structured text data and measure their performance
2. Produce understandable outputs and trained models which can be reused
3. Save those results in a clear way which can be shared or used for future analysis

## Some Notes

- Saving the classification report to a text file can be a little tricky. You might need to Google this part!
- You might want to challenge yourself to create a third script which vectorizes the data separately, and saves the new feature extracted dataset. That way, you only have to vectorize the data once in total, instead of once per script.
- Your code should include functions that you have written wherever possible. Try to break your code down into smaller self-contained parts, rather than having it as one long set of instructions.