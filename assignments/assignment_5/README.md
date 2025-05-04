# IMDb Movie Review Sentiment Analysis: A Comparison of Transformer and Traditional Approaches

This project evaluates and compares the performance of transformer models with traditional machine learning approaches for sentiment analysis on the IMDb movie review dataset.

## Overview

Sentiment analysis is a fundamental NLP task with widespread applications. This project explores how modern transformer architectures compare to classic machine learning techniques by:

1. Using the IMDb movie review dataset with binary sentiment labels (positive/negative)
2. Implementing a DistilBERT transformer model for sentiment classification
3. Comparing against a TF-IDF + Logistic Regression baseline
4. Visualizing results and analyzing the performance differences based on review characteristics

## Research Question

**To what extent does a transformer-based approach (DistilBERT) outperform a traditional ML pipeline (TF-IDF + Logistic Regression) for IMDb sentiment analysis, and is this performance difference justified given the significantly higher computational requirements?**

## Project Structure

```
assignment_5/
├── output/                    # Output directory
│   ├── model/                 # Saved model files
│   └── figures/               # Visualization outputs
├── src/                       # Source code
│   ├── main.py                # Main execution script
│   ├── settings.py            # Centralized configuration
│   ├── models/                # Model implementations
│   │   ├── distil_bert.py     # Transformer model
│   │   └── logistic_regression.py # Baseline model
│   ├── utils/                 # Utility functions
│   │   ├── data.py            # Data loading and preprocessing
│   │   └── visualization.py   # Visualization functions
├── README.md                  # Project documentation
├── run.sh                     # Execution script
└── setup.sh                   # Environment setup script
```

## Implementation Details

### Data Processing

The project uses the IMDb movie review dataset from HuggingFace:

- Binary sentiment classification (positive/negative reviews)
- Dataset loaded directly using HuggingFace Datasets library
- **Balanced dataset creation** through stratified sampling to maintain equal positive/negative distributions
- Flexible train/validation/test splitting with class balance preservation

The data preparation process ensures:
1. Equal representation of sentiment classes to prevent bias
2. Stratified sampling that maintains the original class distribution when using subsets
3. Proportional validation split that preserves class balance
4. Consistent handling of both training and test datasets

### Models

**Baseline Model:**
- TF-IDF vectorization with unigrams, bigrams, and trigrams (n-gram range: 1-3)
- Bayesian hyperparameter optimization using BayesSearchCV with 20 iterations using 3-fold cross-validation
- Hyperparameters tuned include:
  - max_features: 5000-20000 features
  - min_df: 2-15 minimum document frequency
  - max_df: 0.6-0.95 maximum document frequency
  - sublinear_tf: True/False for log scaling of term frequencies
  - C: 0.01-100 regularization parameter (log-uniform prior)
- Logistic Regression classifier with parallel processing (n_jobs=-1)
- Early stopping during hyperparameter search based on validation performance

The baseline model applies sublinear TF scaling to term frequencies rather than using raw counts. This accounts for the diminishing returns of repeated terms in sentiment analysis—a sentiment word appearing ten times doesn't make a review ten times more positive/negative than if it appeared once. This scaling helps balance feature importance across reviews of varying lengths, preventing longer reviews from dominating purely because of word repetition while still preserving the relative importance of terms.

**Transformer Model:**
- DistilBERT (distilbert-base-uncased) for sequence classification
- Uses HuggingFace Transformers library for implementation
- Text truncation/padding to 512 tokens maximum length
- Full fine-tuning approach with early stopping (patience=3)
- Training configuration:
  - Batch size: 16
  - Learning rate: 1e-5 with linear scheduler
  - Weight decay: 0.01
  - AdamW optimizer with dynamic warmup (10% of total training steps)
  - Evaluation strategy: once per epoch
  - 3 training epochs (default)
  - Gradient clipping with max_grad_norm=0.8 for training stability

### Visualization & Analysis

The project generates several visualizations to compare model performance:

1. **Metrics Comparison** (metrics_comparison.png):
   - Bar chart comparing accuracy, precision, recall, and F1 for both models
   - Quantifies the performance difference between approaches

2. **Review Length Performance** (review_length_performance.png):
   - Compares model accuracy across different review length categories (0-100, 101-200, 201-300, 301-500, 501-1000, 1000+ words)
   - Reveals how text length affects performance for different model types

3. **Confusion Matrices** (confusion_matrices.png):
   - Normalized matrices showing class-specific performance
   - Provides insight into model classification behavior across sentiment classes

## Getting Started

### Quick Start

Simply run the provided script to start the analysis:

```bash
./run.sh
```

This script handles everything automatically:
- Checks if the environment is set up, and prompts to run setup if needed
- Displays current configuration settings
- Runs the IMDb sentiment analysis pipeline

On first run, the system will automatically download the IMDb dataset from Hugging Face and cache it locally for future use.

### CLI Options

If you prefer more control, you can run the analysis with custom parameters:

```bash
python src/main.py run --sample-size 10000 --model transformer --num-epochs 3
```

Available parameters:
- `--sample-size`: Number of samples to use (default: 10000)
- `--model`: Which model to train - "logistic", "transformer", or "both" (default: "both")
- `--num-epochs`: Number of epochs for transformer training (default: 3)
- `--resume-from`: Resume training from a checkpoint (e.g., checkpoint-3000)

### Resuming Training

You can continue training from a saved checkpoint:

```bash
python src/main.py run --model transformer --resume-from checkpoint-3000
```

## Technology Stack

- **Data Processing**: HuggingFace Datasets
- **Machine Learning**: scikit-learn, HuggingFace Transformers
- **Hyperparameter Optimization**: scikit-optimize (Bayesian optimization)
- **Visualization**: Matplotlib
- **Configuration**: Pydantic
- **CLI Interface**: Click

## Results

### Performance Comparison

Our analysis reveals significant differences between the two approaches:

| Model | Accuracy | Precision | Recall | F1 | Training Time |
|-------|----------|-----------|--------|----|----|
| Logistic Regression | 0.887 | 0.887 | 0.887 | 0.887 | ~2.15 minutes* |
| DistilBERT | 0.908 | 0.910 | 0.906 | 0.908 | ~35 minutes |

*Including Bayesian hyperparameter optimization with 20 iterations*

The transformer model achieved approximately 2.1 percentage points higher performance in accuracy, with similar improvements across other metrics. However, this improvement comes at a substantial computational cost, with training time increasing from minutes to tens of minutes.

![Metrics Comparison](output/figures/metrics_comparison.png)

### Performance by Review Length

Our analysis of how review length affects model performance revealed interesting patterns:

![Review Length Performance](output/figures/review_length_performance.png)

Key findings:
- Both models achieve peak accuracy (>95%) on medium-length reviews (201-300 words)
- DistilBERT demonstrates stronger performance on longer reviews (501+ words)
- The traditional TF-IDF model performs comparably or slightly better on shorter reviews (101-500 words)
- Performance analysis suggests that medium-length reviews provide optimal information without excess noise

## Discussion

The results demonstrate that while transformer models provide measurable but modest performance improvements for sentiment analysis (~2.1% better accuracy), the magnitude of this improvement may not always justify their substantially higher computational requirements for all use cases.

### Implications

1. **Resource Tradeoffs**: For applications requiring real-time processing or deployment on resource-constrained environments, the traditional pipeline remains highly competitive with 88.7% accuracy while training in approximately 2 minutes.

2. **Review Length Sensitivity**: The transformer model's advantage is most pronounced for longer reviews (501+ words), where attention mechanisms likely help capture long-range dependencies better than bag-of-words approaches.

3. **Production Considerations**: The relatively fast training time of the logistic regression model makes it suitable for frequent retraining with updated data, while the transformer approach requires more careful planning around when to retrain.

4. **Diminishing Returns**: The 2.1% accuracy improvement from transformers represents an 18.6% reduction in error rate, which may be significant for certain high-stakes applications but negligible for others.

### Limitations

- The current analysis uses a simplified subset of the full IMDb dataset
- We focused only on binary sentiment classification
- The transformer implementation uses a small-scale DistilBERT model rather than larger architectures
- Fixed parameters were used for the transformer model, while the logistic regression pipeline receives full hyperparameter optimization
- The sample distribution is highly uneven across length categories, which may affect the reliability of comparisons in the smallest categories

### Future Work
Future extensions could explore:
- Fine-grained sentiment analysis beyond binary classification
- Performance comparison on other domains beyond movie reviews
- Hybrid approaches combining the speed of traditional methods with the power of transformers
- Systematic hyperparameter optimization for the transformer model
- Learning rate scheduling and advanced training techniques for the transformer model

## Acknowledgments

- This project uses the IMDb dataset created by Maas et al. The dataset was introduced in the paper "Learning Word Vectors for Sentiment Analysis" (Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C., 2011) and accessed through the HuggingFace platform.
- Transformer models powered by the [HuggingFace Transformers](https://huggingface.co/transformers/) library
- Dataset handling provided by [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
