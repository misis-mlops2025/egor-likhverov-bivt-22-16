from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from homework2.config import DataConfig


def generate_dataset(config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset.

    Args:
        config: Configuration for dataset generation

    Returns:
        Tuple of features (X) and labels (y)
    """
    x, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_classes=config.n_classes,
        random_state=config.random_state,
    )
    return x, y


def split_dataset(
    x: np.ndarray, y: np.ndarray, config: DataConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.

    Args:
        X: Features
        y: Labels
        config: Configuration with test_size parameter

    Returns:
        Tuple of X_train, X_test, y_train, y_test
    """
    return train_test_split(x, y, test_size=config.test_size, random_state=config.random_state)
