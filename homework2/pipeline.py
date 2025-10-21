from pathlib import Path
from typing import Dict

import logging
import joblib

from homework2.config import PipelineConfig, load_config
from homework2.data_generation import generate_dataset, split_dataset
from homework2.model_factory import create_model
from homework2.train import evaluate_model, train_model


logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)


def run_pipeline(config: PipelineConfig) -> Dict[str, float]:
    """
    Run the complete ML pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Starting ML pipeline")
    logger.info(f"Model type: {config.model.model_type.value}")

    # Generate dataset
    logger.info("Generating dataset...")
    x, y = generate_dataset(config.data)
    logger.info(f"Dataset shape: {x.shape}, Classes: {len(set(y))}")

    # Split dataset
    logger.info("Splitting dataset...")
    x_train, x_test, y_train, y_test = split_dataset(x, y, config.data)
    logger.info(f"Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config.model)
    logger.info(f"Model created: {type(model).__name__}")

    # Train model
    logger.info("Training model...")
    model = train_model(model, x_train, y_train)
    logger.info("Model training completed")

    # Evaluate model
    logger.info("Evaluating model...")
    _, metrics = evaluate_model(model, x_test, y_test)

    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    # Save model
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{config.model.model_type.value}_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return metrics


def main():
    """Main entry point for the pipeline."""
    config = load_config()
    metrics = run_pipeline(config)
    logger.info(f"Pipeline completed successfully. Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    logger.warning("hello")
    main()
