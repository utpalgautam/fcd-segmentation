from pathlib import Path

import pytest
from pydantic import ValidationError

from fcd_segmentation.config import (
    EvaluationConfig,
    TrainingConfig,
    load_yaml,
    resolve_path,
)


def test_yaml_mapping(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("device: auto\nseed: 42\n", encoding="utf-8")
    assert load_yaml(config) == {"device": "auto", "seed": 42}


def test_yaml_non_mapping(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(config)


def test_resolve_path(tmp_path: Path):
    assert resolve_path("data", base_dir=tmp_path) == (tmp_path / "data").resolve()


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.seed == 42
    assert cfg.device == "auto"
    assert cfg.trainer == "FCDnnUNetTrainer_100epochs"
    assert cfg.epochs == 100
    assert cfg.batch_size == 1
    assert cfg.learning_rate == 0.001


def test_training_config_from_yaml(tmp_path: Path):
    yaml_file = tmp_path / "training.yaml"
    yaml_file.write_text(
        "seed: 123\n"
        "device: cuda\n"
        "epochs: 50\n"
        "batch_size: 2\n"
        "learning_rate: 0.005\n"
        "trainer: FCDnnUNetTrainer_50epochs\n",
        encoding="utf-8",
    )
    cfg = TrainingConfig.from_yaml(yaml_file)
    assert cfg.seed == 123
    assert cfg.device == "cuda"
    assert cfg.epochs == 50
    assert cfg.batch_size == 2
    assert cfg.learning_rate == 0.005
    assert cfg.trainer == "FCDnnUNetTrainer_50epochs"


def test_training_config_invalid_device():
    with pytest.raises(ValidationError):
        TrainingConfig(device="tpu")


def test_training_config_invalid_numeric_bounds():
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=-0.01)


def test_evaluation_config_defaults():
    cfg = EvaluationConfig()
    assert cfg.split == "test"
    assert cfg.threshold == 0.5
    assert "dice" in cfg.metrics
    assert "sensitivity" in cfg.metrics
    assert "hd95" in cfg.metrics


def test_evaluation_config_invalid_metric():
    with pytest.raises(ValidationError):
        EvaluationConfig(metrics=["dice", "nonexistent_metric"])


def test_evaluation_config_invalid_threshold():
    with pytest.raises(ValidationError):
        EvaluationConfig(threshold=1.5)
