# Expected Knowledge Document

## Classification Fundamentals

### 1. Performance Evaluation
- Understand train-test split methodology
- Apply cross-validation techniques
- Evaluate model performance on held-out test sets
- Interpret model performance "in the wild"

### 2. Multi-class Classification
- Distinguish between binary and multi-class classification problems
- Calculate and interpret evaluation metrics for multi-class scenarios:
  - Precision and recall for individual classes
  - Micro-average precision
  - Macro-average precision
  - Pooled metrics

### 3. Regression Concepts
- Understand the relationship between independent and dependent variables
- Identify linear relationships in data
- Fit a line of best fit to model relationships
- Recognize limitations of linear regression for categorical variables
- Apply logistic regression for categorical outcomes
- Understand the logistic function as a link function to "squash" values between 0 and 1

## Machine Learning Implementation

### 1. Text Classification
- Implement logistic regression classifiers for text data (e.g., fake news detection)
- Configure and optimize text vectorizers:
  - Number of features
  - Lowercasing options
  - Document frequency parameters
- Inspect data structures used in classification pipelines
- Apply classification techniques to different domains (e.g., sentiment analysis)
- Compare traditional ML algorithms (Naive Bayes, Support Vector Machines)

### 2. Gradient Descent
- Understand the concept of loss functions
- Recognize how parameter adjustments influence loss values
- Apply gradient descent to find optimal parameters
- Implement stochastic gradient descent for efficiency

## Neural Networks

### 1. Neural Network Fundamentals
- Understand the biological inspiration for artificial neurons
- Recognize the flow of information in neural networks
- Compare neural networks to logistic regression
- Identify key components of artificial neurons:
  - Inputs and weights
  - Net sum calculation
  - Activation functions
  - Output values

### 2. Network Architecture
- Design feed-forward neural networks
- Understand the role of hidden layers
- Recognize how information flows from input through hidden layers to output
- Calculate the number of parameters in a network
- Appreciate neural networks as universal function approximators
- Understand how layers enable modeling of complex, non-linear patterns

### 3. Training Neural Networks
- Apply backpropagation for weight adjustment
- Minimize loss functions in neural networks
- Recognize the importance of non-linear activation functions
- Understand how neural networks learn feature interactions