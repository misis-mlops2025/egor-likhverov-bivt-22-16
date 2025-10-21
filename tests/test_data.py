import numpy as np
import pytest


from homework2.config import DataConfig
from homework2.data_generation import generate_dataset, split_dataset


def test_generate_dataset_default():
    """Test dataset generation with default config."""
    config = DataConfig()
    x, y = generate_dataset(config)

    assert x.shape[0] == config.n_samples
    assert x.shape[1] == config.n_features
    assert len(y) == config.n_samples
    assert len(set(y)) == config.n_classes


def test_generate_dataset_custom():
    """Test dataset generation with custom config."""
    config = DataConfig(n_samples=500, n_features=10, n_classes=3)
    print(config)
    x, y = generate_dataset(config)

    assert x.shape == (500, 10)
    assert len(set(y)) == 3


def test_split_dataset():
    """Test dataset splitting."""
    config = DataConfig(n_samples=100, test_size=0.3)
    x, y = generate_dataset(config)
    x_train, x_test, y_train, y_test = split_dataset(x, y, config)

    assert x_train.shape[0] == 70
    assert x_test.shape[0] == 30
    assert len(y_train) == 70
    assert len(y_test) == 30


def test_split_dataset_reproducibility():
    """Test that splitting is reproducible with same random_state."""
    config = DataConfig(random_state=42)
    x, y = generate_dataset(config)

    x_train1, x_test1, _, _ = split_dataset(x, y, config)
    x_train2, x_test2, _, _ = split_dataset(x, y, config)

    np.testing.assert_array_equal(x_train1, x_train2)
    np.testing.assert_array_equal(x_test1, x_test2)
