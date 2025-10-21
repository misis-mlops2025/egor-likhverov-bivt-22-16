from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelType(str, Enum):
    """Supported model types."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"


class DataConfig(BaseModel):
    """Configuration for dataset generation."""
    
    n_samples: int = Field(default=1000, ge=100, description="Number of samples")
    n_features: int = Field(default=20, ge=2, description="Number of features")
    n_informative: Optional[int] = Field(default=None, ge=1, description="Number of informative features")
    n_redundant: Optional[int] = Field(default=None, ge=0, description="Number of redundant features")
    n_classes: int = Field(default=2, ge=2, description="Number of classes")
    random_state: Optional[int] = Field(default=42, description="Random state for reproducibility")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")

    # @field_validator("n_informative", "n_redundant", mode="before")
    # @classmethod
    # def handle_none_values(cls, v: Optional[int]) -> int:
    #     """Convert None to 0 for n_informative and n_redundant."""
    #     return v if v is not None else 0

    @field_validator("n_informative", "n_redundant")
    @classmethod
    def validate_features(cls, v: int, info) -> int:
        """Validate that n_informative and n_redundant constraints are met."""
        values = info.data
        n_features = values.get("n_features", 0)
        
        if v > n_features:
            field = "n_informative" if info.field_name == "n_informative" else "n_redundant"
            raise ValueError(f"{field} must be <= n_features ({n_features})")
            
        n_informative = values.get("n_informative", 0)
        n_redundant = values.get("n_redundant", 0)
        
        if info.field_name == "n_redundant":
            n_informative = values.get("n_informative", 0)
        elif info.field_name == "n_informative":
            n_redundant = values.get("n_redundant", 0)
            
        if n_informative + n_redundant > n_features:
            raise ValueError(
                f"n_informative ({n_informative}) + n_redundant ({n_redundant}) "
                f"must be <= n_features ({n_features})"
            )
        return v
    
    @model_validator(mode="after")
    def validate_features_sum(self):
        """
        Ensure that n_informative + n_redundant == n_features.
        If one or both are None, assign automatically.
        """
        n_features = self.n_features
        n_informative = self.n_informative
        n_redundant = self.n_redundant

        if n_informative is None and n_redundant is None:
            n_informative = int(0.75 * n_features)
            n_redundant = n_features - n_informative

        elif n_informative is None:
            n_informative = n_features - n_redundant
        elif n_redundant is None:
            n_redundant = n_features - n_informative
        print(n_redundant, n_informative)
        
        if n_informative + n_redundant != n_features:
            self.n_informative = n_features
            self.n_redundant = 0

        if n_informative < 0 or n_redundant < 0:
            raise ValueError("n_informative and n_redundant must be non-negative")

        self.n_informative = n_informative
        self.n_redundant = n_redundant
        return self
    

class LogisticRegressionConfig(BaseModel):
    """Configuration for Logistic Regression."""

    max_iter: int = Field(default=1000, ge=100)
    random_state: int = Field(default=42)
    solver: str = Field(default="lbfgs")


class RandomForestConfig(BaseModel):
    """Configuration for Random Forest."""

    n_estimators: int = Field(default=100, ge=10)
    max_depth: Optional[int] = Field(default=10, ge=1)
    random_state: int = Field(default=42)
    min_samples_split: int = Field(default=2, ge=2)


class DecisionTreeConfig(BaseModel):
    """Configuration for Decision Tree."""

    max_depth: Optional[int] = Field(default=5, ge=1)
    random_state: int = Field(default=42)
    min_samples_split: int = Field(default=2, ge=2)


class ModelConfig(BaseModel):
    """Configuration for model selection and parameters."""

    model_type: ModelType = Field(default=ModelType.LOGISTIC_REGRESSION)
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    decision_tree: DecisionTreeConfig = DecisionTreeConfig()


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    output_dir: str = Field(default="models", description="Directory to save models")


def load_config() -> PipelineConfig:
    """Load configuration from default values."""
    return PipelineConfig()
