import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from homework2.config import ModelConfig, ModelType
from homework2.model_factory import create_model
from homework2.train import evaluate_model, train_model


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    x = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return x[:80], x[80:], y[:80], y[80:]


def test_create_logistic_regression():
    """Test creation of logistic regression model."""
    config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
    model = create_model(config)
    assert isinstance(model, LogisticRegression)


def test_create_random_forest():
    """Test creation of random forest model."""
    config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
    model = create_model(config)
    assert isinstance(model, RandomForestClassifier)


def test_create_decision_tree():
    """Test creation of decision tree model."""
    config = ModelConfig(model_type=ModelType.DECISION_TREE)
    model = create_model(config)
    assert isinstance(model, DecisionTreeClassifier)


def test_train_model(sample_data):
    """Test model training."""
    x_train, _, y_train, _ = sample_data
    config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
    model = create_model(config)

    trained_model = train_model(model, x_train, y_train)
    assert hasattr(trained_model, "predict")


def test_evaluate_model(sample_data):
    """Test model evaluation."""
    x_train, x_test, y_train, y_test = sample_data
    config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
    model = create_model(config)

    model = train_model(model, x_train, y_train)
    predictions, metrics = evaluate_model(model, x_test, y_test)

    assert len(predictions) == len(y_test)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1
