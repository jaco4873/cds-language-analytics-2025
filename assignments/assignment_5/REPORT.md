# IMDb Sentiment Analysis: Transformer vs. Traditional Methods

## Introduction
Sentiment analysis is a fundamental NLP task with applications across domains from marketing to social media monitoring. This study compares transformer-based and traditional machine learning approaches for sentiment classification on movie reviews. Specifically, we investigate:

**Research Question**: How do transformer-based models (DistilBERT) compare to traditional ML pipelines (TF-IDF + Logistic Regression) for IMDb sentiment analysis in terms of performance and computational efficiency?

The importance of this question lies in determining the most appropriate approach for different NLP application contexts, balancing accuracy requirements against computational constraints.

## Methods
### Data Processing
We used the IMDb movie reviews dataset with binary sentiment labels (positive/negative) from HuggingFace's datasets library. The data processing pipeline included:

1. Loading the dataset directly using the `load_dataset()` function
2. Standardizing column names ("text" → "review", "label" → "sentiment")  
3. Creating a balanced dataset of 10,000 reviews through stratified sampling (5,000 positive, 5,000 negative)
4. Implementing a 90/10 train/validation split while preserving class balance
5. Using a separate test set of 2,000 reviews with equal class distribution


### Model Implementation
#### Baseline: TF-IDF + Logistic Regression
Our baseline model combined TF-IDF vectorization with logistic regression:

1. **Vectorization**: n-gram range of 1-3 (unigrams, bigrams, trigrams)
2. **Hyperparameter Optimization**: Bayesian search with 20 iterations using 3-fold cross-validation
3. **Search Space**:
   - max_features: 5,000-20,000 features
   - min_df: 2-15 minimum document frequency
   - max_df: 0.6-0.95 maximum document frequency  
   - sublinear_tf: True/False for log scaling
   - C: 0.01-100 regularization parameter (log-uniform prior)
4. **Training**: Multi-threaded training with early stopping based on validation performance

#### Transformer: DistilBERT
The transformer approach used a fine-tuned DistilBERT model:

1. **Model Initialization**: Pre-trained "distilbert-base-uncased" with added classification head
2. **Text Processing**:
   - Tokenization with padding and truncation to 512 tokens
   - Conversion to PyTorch tensors with attention masks
3. **Training Configuration**:
   - Batch size of 16
   - Learning rate of 1e-5 with AdamW optimizer
   - Weight decay of 0.01
   - Linear learning rate scheduler with 10% warmup
   - 3 epochs with early stopping (patience=3)
   - Gradient clipping with max_grad_norm=0.8

#### Parameter Choice Rationale
For the transformer model, we employed reasonable parameters rather than extensive tuning due to computational constraints. Full hyperparameter optimization for transformers requires prohibitive resources (potentially days of GPU time), so we limited optimization to early stopping. We selected a batch size of 16 and maximum of 3 epochs as a balance between training stability and computational efficiency. Learning rate (1e-5) is within the recommended range for DistilBERT fine-tuning. The dynamic warmup schedule (10% of total steps) was implemented to stabilize early training and prevent gradient issues, while weight decay (0.01) helps control overfitting.

### Evaluation and Visualization
We evaluated both models using:

1. **Performance Metrics**: Accuracy, precision, recall, and F1 score on the test set
2. **Confusion Matrices**: Normalized matrices showing class-specific performance
3. **Length Analysis**: Performance stratified by review length categories (0-100, 101-200, 201-300, 301-500, 501-1000, 1000+ words)

Visualizations were created using Matplotlib with custom functions for metrics comparison, confusion matrices, and length-based performance analysis.

## Results
The transformer model achieved 90.8% accuracy (precision: 0.910, recall: 0.906, F1: 0.908) compared to the logistic regression model's 88.7% accuracy (precision: 0.887, recall: 0.887, F1: 0.887), representing a 2.1 percentage point improvement. However, this performance gain came with significantly higher computational costs: 35 minutes training time for DistilBERT versus approximately 2 minutes for the logistic regression pipeline, including hyperparameter optimization.

Performance analysis by review length revealed both models achieved peak accuracy (>95%) on medium-length reviews (201-300 words). DistilBERT demonstrated stronger performance on longer reviews (501+ words), while the traditional model performed comparably on shorter texts. The confusion matrices showed slightly better classification across both positive and negative classes for the transformer model.

### Results table
| Model | Accuracy | Precision | Recall | F1 | Training Time |
|-------|----------|-----------|--------|----|----|
| Logistic Regression | 0.887 | 0.887 | 0.887 | 0.887 | ~2.15 minutes* |
| DistilBERT | 0.908 | 0.910 | 0.906 | 0.908 | ~35 minutes |

*Table 1: Model performance metrics and training time comparison*
\* *Including Bayesian hyperparameter optimization with 20 iterations*

### Results graphs
![Metrics Comparison](outputs/figures/metrics_comparison.png)
*Figure 1: Performance metrics comparison between Logistic Regression and DistilBERT models*

![Review Length Performance](outputs/figures/review_length_performance.png)
*Figure 2: Model accuracy across different review length categories*

![Confusion Matrices](outputs/figures/confusion_matrices.png)
*Figure 3: Confusion matrices for Logistic Regression (left) and DistilBERT (right) models*

## Discussion
The results demonstrate that transformer models provide measurable but modest performance improvements over traditional approaches, though with substantially higher computational requirements.

Several observations are notable:
1. The optimized logistic regression model performs remarkably well, achieving nearly 89% accuracy with minimal computational resources
2. The transformer's 2.1% accuracy improvement represents an 18.6% reduction in error rate, which may be significant for certain applications
3. We observed training instability in the transformer, with gradient norm spikes reaching 23.5, suggesting potential benefit from more aggressive gradient clipping

The performance advantage of transformers varies by review length, with stronger benefits for longer documents where attention mechanisms likely help capture long-range dependencies better than bag-of-words approaches. Analyzing Figure 3 reveals that both models achieve peak performance on medium-length reviews (201-300 words), suggesting this length provides optimal information without excess noise. The traditional TF-IDF model performs comparably or slightly better on shorter reviews (101-500 words), while DistilBERT shows clearer advantages in longer documents (501+ words).

The sample distribution is highly uneven across length categories (n=6 for shortest vs. n=910 for longest), which may affect the reliability of comparisons in the smallest categories. This length-based analysis suggests applications with predominantly longer documents may benefit more from transformer architectures, while those dealing with concise text might see minimal benefit over traditional approaches.

For practical implementation, decision factors should include:
1. Available computational resources for training and inference
2. Performance requirements (is the modest accuracy gain worth the resource investment?)
3. Typical document length in the target application
4. Frequency of model retraining needs

### Limitations
Our study used a simplified subset of the IMDb dataset and focused only on binary classification. The transformer implementation used a smaller DistilBERT model with fixed parameters, while the logistic regression pipeline received full hyperparameter optimization.

Future work could explore fine-grained sentiment analysis, performance on other domains, hybrid approaches, and systematic transformer hyperparameter optimization.