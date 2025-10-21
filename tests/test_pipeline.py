"""Additional tests for utilities and edge cases.

This module includes:
- Tests for configuration validation
- Tests for data generation edge cases
- Tests for model factory error handling
- Tests for trainer edge cases
"""

import numpy as np
import pytest
from pydantic import ValidationError

from homework2.config import (
    DataConfig,
    DecisionTreeConfig,
    LogisticRegressionConfig,
    ModelConfig,
    ModelType,
    PipelineConfig,
    RandomForestConfig,
)
from homework2.data_generation import generate_dataset, split_dataset
from homework2.model_factory import create_model
from homework2.train import evaluate_model, train_model


class TestDataConfigValidation:
    """Tests for DataConfig validation rules."""

    def test_valid_data_config(self):
        """Test that valid config is accepted."""
        config = DataConfig(
            n_samples=500,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            test_size=0.25
        )
        assert config.n_samples == 500
        assert config.n_features == 10

    def test_negative_samples_rejected(self):
        """Test that negative samples are rejected."""
        with pytest.raises(ValidationError):
            DataConfig(n_samples=-100)

    def test_too_small_samples_rejected(self):
        """Test that too small sample size is rejected."""
        with pytest.raises(ValidationError):
            DataConfig(n_samples=50)  # Less than minimum of 100

    def test_invalid_test_size_rejected(self):
        """Test that invalid test sizes are rejected."""
        # Too small
        with pytest.raises(ValidationError):
            DataConfig(test_size=0.05)

        # Too large
        with pytest.raises(ValidationError):
            DataConfig(test_size=0.6)

    def test_n_informative_exceeds_n_features(self):
        """Test that n_informative > n_features is rejected."""
        with pytest.raises(ValidationError):
            DataConfig(n_features=10, n_informative=15)

    def test_boundary_values(self):
        """Test boundary values for configuration."""
        # Minimum valid values
        config = DataConfig(
            n_samples=100,
            n_features=2,
            n_informative=2,
            test_size=0.1
        )
        assert config.n_samples == 100

        # Maximum test_size
        config = DataConfig(test_size=0.5)
        assert config.test_size == 0.5


class TestModelConfigValidation:
    """Tests for model configuration validation."""

    def test_logistic_regression_config_validation(self):
        """Test LogisticRegression config validation."""
        # Valid config
        config = LogisticRegressionConfig(max_iter=2000)
        assert config.max_iter == 2000

        # Invalid: too few iterations
        with pytest.raises(ValidationError):
            LogisticRegressionConfig(max_iter=50)

    def test_random_forest_config_validation(self):
        """Test RandomForest config validation."""
        # Valid config
        config = RandomForestConfig(n_estimators=200, max_depth=15)
        assert config.n_estimators == 200

        # Invalid: too few estimators
        with pytest.raises(ValidationError):
            RandomForestConfig(n_estimators=5)

        # Invalid: min_samples_split too small
        with pytest.raises(ValidationError):
            RandomForestConfig(min_samples_split=1)

    def test_decision_tree_config_validation(self):
        """Test DecisionTree config validation."""
        # Valid config
        config = DecisionTreeConfig(max_depth=10)
        assert config.max_depth == 10

        # Invalid: max_depth too small
        with pytest.raises(ValidationError):
            DecisionTreeConfig(max_depth=0)


class TestDataGeneratorEdgeCases:
    """Tests for edge cases in data generation."""

    def test_minimum_features(self):
        """Test with minimum number of features."""
        config = DataConfig(n_samples=100, n_features=2, n_informative=2)
        X, y = generate_dataset(config)

        assert X.shape == (100, 2)
        assert len(y) == 100

    def test_all_features_informative(self):
        """Test when all features are informative."""
        config = DataConfig(
            n_samples=200,
            n_features=10,
            n_informative=10,
            n_redundant=0
        )
        X, y = generate_dataset(config)

        assert X.shape == (200, 10)

    def test_balanced_classes(self):
        """Test that classes are reasonably balanced."""
        config = DataConfig(n_samples=1000, n_classes=2)
        X, y = generate_dataset(config)

        class_counts = np.bincount(y)
        # Classes should be roughly balanced (within 30%)
        assert all(count > 300 for count in class_counts)

    def test_multiclass_generation(self):
        """Test generation with multiple classes."""
        for n_classes in [3, 4, 5]:
            config = DataConfig(n_samples=300, n_classes=n_classes)
            X, y = generate_dataset(config)

            unique_classes = set(y)
            assert len(unique_classes) == n_classes

    def test_reproducibility_with_random_state(self):
        """Test that same random_state produces same data."""
        config = DataConfig(random_state=42)

        X1, y1 = generate_dataset(config)
        X2, y2 = generate_dataset(config)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_random_states_produce_different_data(self):
        """Test that different random states produce different data."""
        config1 = DataConfig(random_state=42)
        config2 = DataConfig(random_state=123)

        X1, y1 = generate_dataset(config1)
        X2, y2 = generate_dataset(config2)

        # Data should be different
        assert not np.array_equal(X1, X2)


class TestDataSplitting:
    """Tests for dataset splitting functionality."""

    def test_split_proportions(self):
        """Test that split produces correct proportions."""
        config = DataConfig(n_samples=100, test_size=0.2)
        X, y = generate_dataset(config)
        X_train, X_test, y_train, y_test = split_dataset(X, y, config)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_preserves_features(self):
        """Test that split preserves number of features."""
        config = DataConfig(n_features=15)
        X, y = generate_dataset(config)
        X_train, X_test, y_train, y_test = split_dataset(X, y, config)

        assert X_train.shape[1] == 15
        assert X_test.shape[1] == 15

    def test_split_no_data_leakage(self):
        """Test that train and test sets don't overlap."""
        config = DataConfig(n_samples=100, random_state=42)
        X, y = generate_dataset(config)
        X_train, X_test, y_train, y_test = split_dataset(X, y, config)

        # Check that no rows from train appear in test
        # (This is a basic check, not foolproof for floating point)
        train_set = set(map(tuple, X_train))
        test_set = set(map(tuple, X_test))
        assert len(train_set.intersection(test_set)) == 0

    def test_split_different_test_sizes(self):
        """Test splitting with various test sizes."""
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)

        test_sizes = [0.1, 0.2, 0.3, 0.4]
        for test_size in test_sizes:
            config = DataConfig(test_size=test_size)
            X_train, X_test, _, _ = split_dataset(X, y, config)

            expected_test = int(200 * test_size)
            expected_train = 200 - expected_test

            assert len(X_test) == expected_test
            assert len(X_train) == expected_train


class TestModelFactoryErrorHandling:
    """Tests for model factory error handling."""

    def test_create_all_supported_models(self):
        """Test creation of all supported model types."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        model_types = [
            (ModelType.LOGISTIC_REGRESSION, LogisticRegression),
            (ModelType.RANDOM_FOREST, RandomForestClassifier),
            (ModelType.DECISION_TREE, DecisionTreeClassifier)
        ]

        for model_type, expected_class in model_types:
            config = ModelConfig(model_type=model_type)
            model = create_model(config)
            assert isinstance(model, expected_class)

    def test_model_parameters_applied_correctly(self):
        """Test that model parameters are applied from config."""
        # Logistic Regression
        config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
        config.logistic_regression.max_iter = 1500
        model = create_model(config)
        assert model.max_iter == 1500

        # Random Forest
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        config.random_forest.n_estimators = 150
        config.random_forest.max_depth = 12
        model = create_model(config)
        assert model.n_estimators == 150
        assert model.max_depth == 12

        # Decision Tree
        config = ModelConfig(model_type=ModelType.DECISION_TREE)
        config.decision_tree.max_depth = 8
        model = create_model(config)
        assert model.max_depth == 8


class TestTrainerFunctionality:
    """Tests for trainer module functionality."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(30, 5)
        y_test = np.random.randint(0, 2, 30)
        return X_train, X_test, y_train, y_test

    def test_train_model_returns_fitted_model(self, simple_dataset):
        """Test that train_model returns a fitted model."""
        X_train, _, y_train, _ = simple_dataset
        config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
        model = create_model(config)

        trained_model = train_model(model, X_train, y_train)

        # Check that model has been fitted
        assert hasattr(trained_model, 'classes_')
        assert hasattr(trained_model, 'coef_')

    def test_evaluate_model_returns_predictions_and_metrics(self, simple_dataset):
        """Test that evaluate returns predictions and metrics."""
        X_train, X_test, y_train, y_test = simple_dataset
        config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
        model = create_model(config)
        model = train_model(model, X_train, y_train)

        predictions, metrics = evaluate_model(model, X_test, y_test)

        # Check predictions
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1] for pred in predictions)

        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_metrics_in_valid_range(self, simple_dataset):
        """Test that all metrics are in valid range [0, 1]."""
        X_train, X_test, y_train, y_test = simple_dataset
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        model = create_model(config)
        model = train_model(model, X_train, y_train)

        _, metrics = evaluate_model(model, X_test, y_test)

        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} = {value} is out of range"

    def test_perfect_predictions_give_perfect_metrics(self):
        """Test that perfect predictions result in perfect metrics."""
        # Create perfectly separable data
        X_train = np.array([[0], [1], [0], [1]])
        y_train = np.array([0, 1, 0, 1])
        X_test = np.array([[0], [1]])
        y_test = np.array([0, 1])

        config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
        config.logistic_regression.max_iter = 1000
        model = create_model(config)
        model = train_model(model, X_train, y_train)

        _, metrics = evaluate_model(model, X_test, y_test)

        # With perfectly separable data, metrics should be 1.0
        assert metrics['accuracy'] == 1.0

    def test_evaluation_with_multiclass(self):
        """Test evaluation with multiclass classification."""
        np.random.seed(42)
        X_train = np.random.rand(150, 5)
        y_train = np.random.randint(0, 3, 150)  # 3 classes
        X_test = np.random.rand(50, 5)
        y_test = np.random.randint(0, 3, 50)

        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        model = create_model(config)
        model = train_model(model, X_train, y_train)

        predictions, metrics = evaluate_model(model, X_test, y_test)

        assert len(predictions) == 50
        assert all(pred in [0, 1, 2] for pred in predictions)
        assert 'accuracy' in metrics

    def test_train_with_single_class_raises_no_error(self):
        """Test that training with data that might have issues still works."""
        # Note: sklearn models handle this, but we test for graceful behavior
        X_train = np.random.rand(50, 5)
        y_train = np.zeros(50, dtype=int)  # All same class

        config = ModelConfig(model_type=ModelType.DECISION_TREE)
        model = create_model(config)

        # Should not raise an error
        trained_model = train_model(model, X_train, y_train)
        assert trained_model is not None


class TestPipelineConfigIntegration:
    """Tests for complete pipeline configuration."""

    def test_default_pipeline_config(self):
        """Test that default pipeline config is valid."""
        config = PipelineConfig()

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert config.output_dir == "models"

    def test_nested_config_modification(self):
        """Test modifying nested configuration."""
        config = PipelineConfig()

        # Modify data config
        config.data.n_samples = 5000
        config.data.n_features = 25

        # Modify model config
        config.model.model_type = ModelType.RANDOM_FOREST
        config.model.random_forest.n_estimators = 200

        assert config.data.n_samples == 5000
        assert config.model.random_forest.n_estimators == 200

    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        config = PipelineConfig()
        config.data.n_samples = 1500
        config.model.model_type = ModelType.DECISION_TREE

        # Convert to dict
        config_dict = config.model_dump()

        # Recreate from dict
        new_config = PipelineConfig(**config_dict)

        assert new_config.data.n_samples == 1500
        assert new_config.model.model_type == ModelType.DECISION_TREE


class TestDataIntegrity:
    """Tests for data integrity throughout pipeline."""

    def test_no_nan_values_in_generated_data(self):
        """Test that generated data contains no NaN values."""
        config = DataConfig(n_samples=500)
        X, y = generate_dataset(config)

        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_no_inf_values_in_generated_data(self):
        """Test that generated data contains no infinite values."""
        config = DataConfig(n_samples=500)
        X, y = generate_dataset(config)

        assert not np.any(np.isinf(X))
        assert not np.any(np.isinf(y))

    def test_data_types_are_correct(self):
        """Test that generated data has correct types."""
        config = DataConfig()
        X, y = generate_dataset(config)

        assert X.dtype in [np.float32, np.float64]
        assert y.dtype in [np.int32, np.int64]

    def test_label_range_is_correct(self):
        """Test that labels are in correct range."""
        config = DataConfig(n_classes=4)
        X, y = generate_dataset(config)

        assert np.min(y) >= 0
        assert np.max(y) < 4
        assert set(y).issubset(set(range(4)))


class TestModelPersistence:
    """Tests for model saving and loading."""

    def test_model_file_format(self, tmp_path):
        """Test that model is saved in correct format."""
        from pathlib import Path
        import joblib

        config = DataConfig(n_samples=100)
        X, y = generate_dataset(config)
        X_train, X_test, y_train, y_test = split_dataset(X, y, config)

        model_config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        model = create_model(model_config)
        model = train_model(model, X_train, y_train)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Check file exists and can be loaded
        assert model_path.exists()
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

    def test_loaded_model_predictions_match(self, tmp_path):
        """Test that loaded model produces same predictions."""
        import joblib

        config = DataConfig(n_samples=100, random_state=42)
        X, y = generate_dataset(config)
        X_train, X_test, y_train, y_test = split_dataset(X, y, config)

        model_config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
        model = create_model(model_config)
        model = train_model(model, X_train, y_train)

        # Get predictions before saving
        predictions_before = model.predict(X_test)

        # Save and load model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)
        loaded_model = joblib.load(model_path)

        # Get predictions after loading
        predictions_after = loaded_model.predict(X_test)

        # Predictions should be identical
        np.testing.assert_array_equal(predictions_before, predictions_after)


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_mismatched_data_dimensions_raises_error(self):
        """Test that mismatched dimensions raise appropriate errors."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(30, 10)  # Different number of features!

        config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
        model = create_model(config)
        model = train_model(model, X_train, y_train)

        # This should raise an error due to feature mismatch
        with pytest.raises((ValueError, Exception)):
            model.predict(X_test)

    def test_empty_dataset_handling(self):
        """Test behavior with empty dataset."""
        X_train = np.array([]).reshape(0, 5)
        y_train = np.array([])

        config = ModelConfig(model_type=ModelType.DECISION_TREE)
        model = create_model(config)

        # Training with empty data should raise an error
        with pytest.raises((ValueError, Exception)):
            train_model(model, X_train, y_train)


class TestMetricsCalculation:
    """Tests specifically for metrics calculation."""

    def test_weighted_average_for_imbalanced_data(self):
        """Test that weighted average works for imbalanced data."""
        # Create imbalanced dataset
        X_train = np.random.rand(200, 5)
        y_train = np.array([0] * 180 + [1] * 20)  # Highly imbalanced
        X_test = np.random.rand(50, 5)
        y_test = np.array([0] * 45 + [1] * 5)

        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        model = create_model(config)
        model = train_model(model, X_train, y_train)

        _, metrics = evaluate_model(model, X_test, y_test)

        # Metrics should still be calculated (using weighted average)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert not np.isnan(metrics['precision'])
        assert not np.isnan(metrics['recall'])

    def test_all_metrics_types_are_float(self):
        """Test that all metrics are returned as floats."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(30, 5)
        y_test = np.random.randint(0, 2, 30)

        config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
        model = create_model(config)
        model = train_model(model, X_train, y_train)

        _, metrics = evaluate_model(model, X_test, y_test)

        for metric_name, value in metrics.items():
            assert isinstance(value, (float, np.floating)), \
                f"{metric_name} is not float: {type(value)}"


class TestConfigurationEdgeCases:
    """Tests for edge cases in configuration."""

    def test_max_valid_test_size(self):
        """Test with maximum valid test size."""
        config = DataConfig(test_size=0.5)  # Maximum allowed
        X, y = generate_dataset(config)
        X_train, X_test, y_train, y_test = split_dataset(X, y, config)

        assert len(X_test) == len(X_train)

    def test_min_valid_test_size(self):
        """Test with minimum valid test size."""
        config = DataConfig(test_size=0.1)  # Minimum allowed
        X, y = generate_dataset(config)
        X_train, X_test, y_train, y_test = split_dataset(X, y, config)

        assert len(X_test) < len(X_train)

    def test_config_with_none_max_depth(self):
        """Test config with None max_depth (unlimited)."""
        config = RandomForestConfig(max_depth=None)
        assert config.max_depth is None

        config = DecisionTreeConfig(max_depth=None)
        assert config.max_depth is None


# Performance and stress tests
class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.slow
    def test_large_dataset_performance(self, tmp_path):
        """Test pipeline with large dataset (marked as slow test)."""
        config = PipelineConfig()
        config.data.n_samples = 10000
        config.data.n_features = 50
        config.output_dir = str(tmp_path)

        from homework2.pipeline import run_pipeline

        metrics = run_pipeline(config)
        assert metrics is not None

    @pytest.mark.slow
    def test_many_estimators_random_forest(self, tmp_path):
        """Test Random Forest with many estimators."""
        config = PipelineConfig()
        config.data.n_samples = 500
        config.model.model_type = ModelType.RANDOM_FOREST
        config.model.random_forest.n_estimators = 500
        config.output_dir = str(tmp_path)

        from homework2.pipeline import run_pipeline

        metrics = run_pipeline(config)
        assert metrics is not None


# Fixtures for reusable test data
@pytest.fixture(scope="module")
def standard_dataset():
    """Create a standard dataset for multiple tests."""
    config = DataConfig(n_samples=300, random_state=42)
    X, y = generate_dataset(config)
    return X, y, config


@pytest.fixture(scope="module")
def trained_models(standard_dataset, tmp_path_factory):
    """Pre-train all models for testing."""
    X, y, config = standard_dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y, config)

    models = {}
    for model_type in ModelType:
        model_config = ModelConfig(model_type=model_type)
        model = create_model(model_config)
        model = train_model(model, X_train, y_train)
        models[model_type] = model

    return models, X_test, y_test


def test_all_trained_models_work(trained_models):
    """Test that all pre-trained models can make predictions."""
    models, X_test, y_test = trained_models

    for model_type, model in models.items():
        predictions, metrics = evaluate_model(model, X_test, y_test)
        assert len(predictions) == len(y_test)
        assert 'accuracy' in metrics
