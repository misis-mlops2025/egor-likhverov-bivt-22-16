import pytest
from pydantic import ValidationError

from homework2.config import DataConfig, ModelConfig, ModelType, PipelineConfig


def test_data_config_default():
    """Test DataConfig with default values."""
    config = DataConfig()
    assert config.n_samples == 1000
    assert config.n_features == 20
    assert config.test_size == 0.2


def test_data_config_validation():
    """Test DataConfig validation."""
    with pytest.raises(ValidationError):
        DataConfig(n_samples=-100)

    with pytest.raises(ValidationError):
        DataConfig(test_size=0.9)


def test_model_config_default():
    """Test ModelConfig with default values."""
    config = ModelConfig()
    assert config.model_type == ModelType.LOGISTIC_REGRESSION


def test_pipeline_config():
    """Test PipelineConfig initialization."""
    config = PipelineConfig()
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, ModelConfig)
    assert config.output_dir == "models"
