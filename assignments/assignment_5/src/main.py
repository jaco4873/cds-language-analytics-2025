"""
Compare transformer vs traditional approaches for IMDb sentiment analysis.
"""
import click
import warnings
from typing import Literal
from .settings import settings
from .utils.data import load_and_prepare_imdb
from .models.logistic_regression import train_test_logistic_regression, evaluate_logistic_regression_model
from .models.distil_bert import train_distilbert, evaluate_model
from .utils.visualization import visualize_results
from .utils.logger import logger

# Filterwarnings
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on MPS")

@click.group()
def cli():
    """Run IMDb sentiment analysis comparing transformers and traditional ML."""
    pass


@cli.command()
@click.option(
    "--sample-size", 
    default=settings.SAMPLE_SIZE,
    help="Number of samples for training"
)
@click.option(
    "--model", 
    type=click.Choice(["logistic", "transformer", "both"]),
    default=settings.DEFAULT_MODEL,
    help="Model to train"
)
@click.option(
    "--num-epochs", 
    default=settings.NUM_EPOCHS,
    help="Number of epochs for transformer"
)
@click.option(
    "--resume-from",
    help="Resume training from checkpoint (e.g., checkpoint-3000)"
)
def run(
    sample_size: int,
    model: Literal["logistic", "transformer", "both"],
    num_epochs: int,
    resume_from: str | None = None
) -> None:
    """Run the IMDb sentiment analysis pipeline.
    
    Loads data, trains models, evaluates performance and generates visualizations.
    """    
    # Get data
    train_dataset, val_dataset, test_dataset = load_and_prepare_imdb(
        sample_size=sample_size,
        val_split=settings.VALIDATION_SPLIT,
        random_state=settings.RANDOM_SEED
    )
    
    # Store results
    results = {}
    
    # Train logistic regression 
    if model in ["logistic", "both"]:
        logger.info("Training logistic regression")
        
        # Train and evaluate logistic regression
        lr_model, lr_metrics = train_test_logistic_regression(
            train_dataset=train_dataset, 
            val_dataset=val_dataset,
            random_state=settings.RANDOM_SEED
        )
        
        # Get test predictions
        test_metrics, test_predictions = evaluate_logistic_regression_model(
            lr_model, 
            test_dataset
        )
        
        # Store the metrics and predictions
        results["logistic_regression"] = test_metrics
        results["logistic_regression_preds"] = test_predictions
    
    # Train transformer
    if model in ["transformer", "both"]:
        logger.info(f"Training transformer ({num_epochs} epochs)")
        
        # If resuming from checkpoint
        checkpoint_dir = None
        if resume_from:
            checkpoint_dir = settings.MODEL_DIR / resume_from
            if not checkpoint_dir.exists():
                logger.error(f"Checkpoint directory {checkpoint_dir} not found")
                return
            logger.info(f"Resuming training from checkpoint: {resume_from}")
        
        model, tokenizer = train_distilbert(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=num_epochs,
            resume_from_checkpoint=str(checkpoint_dir) if checkpoint_dir else None
        )
        
        # Get test predictions
        transformer_metrics, transformer_preds = evaluate_model(
            model, tokenizer, test_dataset
        )
        
        # Store the metrics and predictions
        results["transformer"] = transformer_metrics
        results["transformer_preds"] = transformer_preds
    
    # Make plots
    if results:
        logger.info("Visualizing results")
        
        # Create visualizations
        visualize_results(
            results=results,
            output_dir=settings.OUTPUT_DIR,
            test_dataset=test_dataset
        )
    
    logger.info("IMDb sentiment analysis complete")


if __name__ == "__main__":
    cli() 