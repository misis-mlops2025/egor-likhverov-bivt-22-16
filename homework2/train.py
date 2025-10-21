from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_model(model: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Train a machine learning model.

    Args:
        model: Scikit-learn model instance
        x_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: Any, x_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate a trained model.

    Args:
        model: Trained scikit-learn model
        x_test: Test features
        y_test: Test labels

    Returns:
        Tuple of predictions and metrics dictionary
    """
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return y_pred, metrics
