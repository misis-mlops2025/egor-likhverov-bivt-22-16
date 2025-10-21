from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from homework2.config import ModelConfig, ModelType


def create_model(config: ModelConfig) -> Any:
    """
    Create a machine learning model based on configuration.

    Args:
        config: Model configuration

    Returns:
        Initialized scikit-learn model

    Raises:
        ValueError: If model type is not supported
    """
    if config.model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(
            max_iter=config.logistic_regression.max_iter,
            random_state=config.logistic_regression.random_state,
            solver=config.logistic_regression.solver,
        )
    if config.model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=config.random_forest.n_estimators,
            max_depth=config.random_forest.max_depth,
            random_state=config.random_forest.random_state,
            min_samples_split=config.random_forest.min_samples_split,
        )
    if config.model_type == ModelType.DECISION_TREE:
        return DecisionTreeClassifier(
            max_depth=config.decision_tree.max_depth,
            random_state=config.decision_tree.random_state,
            min_samples_split=config.decision_tree.min_samples_split,
        )

    raise ValueError(f"Unsupported model type: {config.model_type}")
