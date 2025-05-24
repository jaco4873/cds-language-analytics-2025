"""
DistilBERT transformer model for IMDb sentiment analysis.
"""

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
    EarlyStoppingCallback,
)
from torch.optim import AdamW
from ..settings import settings
from ..utils.logger import logger


def compute_metrics(pred) -> dict[str, float]:
    """
    Calculate model performance metrics.

    Args:
        pred: Prediction output from Trainer

    Returns:
        Dictionary with performance metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train_distilbert(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_name: str = settings.TRANSFORMER_MODEL,
    num_epochs: int = settings.NUM_EPOCHS,
    num_labels: int = settings.NUM_SENTIMENT_CLASSES,
    id2label: dict[int, str] = settings.SENTIMENT_ID2LABEL,
    resume_from_checkpoint: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Train a DistilBERT model on the IMDb dataset.

    Args:
        train_dataset: Training data
        val_dataset: Validation data
        model_name: Name of the pre-trained model to use
        num_epochs: Number of training epochs
        num_labels: Number of output classes
        id2label: Mapping from label IDs to label names
        resume_from_checkpoint: Path to checkpoint to resume training from

    Returns:
        Trained model and tokenizer
    """
    logger.info(f"Training {model_name} model for {num_epochs} epochs")

    label2id = settings.SENTIMENT_LABEL2ID

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model - either from checkpoint or fresh
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
            resume_from_checkpoint, id2label=id2label, label2id=label2id
        )
    else:
        logger.info(f"Starting new training with model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
        )

    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["review"], padding="max_length", truncation=True, max_length=512
        )

    # Apply tokenization to datasets
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "sentiment"]
    )
    val_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "sentiment"]
    )

    # Rename sentiment column to labels for the trainer
    train_tokenized = train_tokenized.rename_column("sentiment", "labels")
    val_tokenized = val_tokenized.rename_column("sentiment", "labels")

    # Calculate total training steps for scheduler
    total_steps = len(train_tokenized) // settings.BATCH_SIZE * num_epochs

    # Calculate warmup steps as 10% of total steps
    warmup_steps = int(total_steps * 0.1)

    # Set up training
    training_args = TrainingArguments(
        output_dir=settings.MODEL_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=settings.BATCH_SIZE,
        per_device_eval_batch_size=settings.BATCH_SIZE,
        learning_rate=settings.LEARNING_RATE,
        warmup_steps=warmup_steps,
        logging_steps=settings.LOGGING_STEPS,
        weight_decay=settings.WEIGHT_DECAY,
        save_strategy=settings.SAVE_STRATEGY,
        save_total_limit=settings.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=settings.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=settings.METRIC_FOR_BEST_MODEL,
        eval_strategy=settings.EVAL_STRATEGY,
        report_to="none",
        max_grad_norm=settings.MAX_GRAD_NORM,
    )

    # Set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY,
    )

    # Create scheduler
    lr_scheduler = get_scheduler(
        name=settings.LEARNING_RATE_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=warmup_steps,
    )

    # Set up trainer with custom optimizer and scheduler
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=settings.EARLY_STOPPING_PATIENCE
            )
        ],
    )

    # Train model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Validation results
    eval_result = trainer.evaluate()
    logger.info(f"Validation results: {eval_result}")

    return model, tokenizer


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
) -> tuple[dict[str, float], list[int]]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test data

    Returns:
        Metrics dictionary and prediction list
    """
    logger.info("Evaluating model on test data")

    # Tokenize the test dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["review"], padding="max_length", truncation=True, max_length=512
        )

    # Apply tokenization
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    test_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "sentiment"]
    )

    # Rename sentiment column to labels for the trainer
    test_tokenized = test_tokenized.rename_column("sentiment", "labels")

    # Set up evaluation trainer
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    # Get results and predictions
    pred_output = trainer.predict(test_tokenized)
    metrics_with_prefix = pred_output.metrics
    predictions = pred_output.predictions.argmax(-1).tolist()

    # Remove the 'test_' prefix from metric keys to prepare for visualization
    results = {
        "accuracy": metrics_with_prefix["test_accuracy"],
        "precision": metrics_with_prefix["test_precision"],
        "recall": metrics_with_prefix["test_recall"],
        "f1": metrics_with_prefix["test_f1"],
    }

    logger.info(f"Test results: {results}")

    return results, predictions
