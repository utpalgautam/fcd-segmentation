"""Configuration loading, validation models, and portable path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load and parse a YAML file, ensuring mapping root structure."""
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_path(value: Path | str, base_dir: Path | str | None = None) -> Path:
    """Resolve a relative or absolute path against base_dir."""
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve()


class TrainingConfig(BaseModel):
    """Validated schema for training workflow configuration."""

    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: str = Field(default="auto", description="Compute device: auto, cuda, cpu, or mps")
    model: str = Field(default="flair", description="Model modality: flair, t1, fusion, or all")
    data_config: str | Path | None = Field(default="configs/data.yaml", description="Path to data configuration")
    batch_size: int = Field(default=1, ge=1, description="Batch size for training")
    epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Initial learning rate")
    num_workers: int = Field(default=0, ge=0, description="Number of dataloader workers")
    mixed_precision: bool = Field(default=False, description="Enable mixed precision training")
    checkpoint_dir: str | Path = Field(default="outputs/checkpoints", description="Directory for model checkpoints")
    experiment_dir: str | Path = Field(default="outputs/experiments", description="Directory for experiment outputs")

    # nnU-Net v2 specific settings
    nnunet_raw: str | Path = Field(default="data/processed", description="nnUNet_raw directory")
    nnunet_preprocessed: str | Path = Field(default="data/nnunet_preprocessed", description="nnUNet_preprocessed directory")
    nnunet_results: str | Path = Field(default="outputs/nnunet_results", description="nnUNet_results directory")
    datasets: dict[str | int, str] = Field(
        default={
            "101": "Dataset101_FLAIR",
            "102": "Dataset102_T1",
            "103": "Dataset103_Fusion",
        },
        description="Dataset mapping ID to folder name",
    )
    configuration: str = Field(default="3d_fullres", description="nnU-Net configuration to train")
    trainer: str = Field(default="FCDnnUNetTrainer_100epochs", description="Trainer class name")
    plans: str = Field(default="nnUNetPlans", description="Plans identifier")
    min_cases_for_five_fold: int = Field(default=5, ge=1, description="Minimum training cases guard")
    keep_going: bool = Field(default=False, description="Continue if a fold or dataset fails")
    verify_dataset_integrity: bool = Field(default=True, description="Verify dataset integrity during preparation")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        norm = v.strip().lower()
        if norm not in {"auto", "cuda", "cpu", "mps"} and not norm.startswith("cuda:"):
            raise ValueError(f"Invalid device '{v}'. Must be 'auto', 'cuda', 'cpu', 'mps', or 'cuda:X'.")
        return norm

    @field_validator("trainer")
    @classmethod
    def validate_trainer(cls, v: str) -> str:
        valid_trainers = {
            "nnUNetTrainer",
            "nnUNetTrainer_100epochs",
            "FCDnnUNetTrainer_100epochs",
            "nnUNetTrainer_50epochs",
            "FCDnnUNetTrainer_50epochs",
        }
        if v not in valid_trainers:
            # Allow custom string but log or accept
            pass
        return v

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainingConfig:
        data = load_yaml(path)
        return cls(**data)


class EvaluationConfig(BaseModel):
    """Validated schema for evaluation configuration."""

    device: str = Field(default="auto", description="Inference compute device")
    checkpoint: str | Path | None = Field(default=None, description="Path to checkpoint file or checkpoint name")
    split: Literal["test", "val", "train"] = Field(default="test", description="Cohort split to evaluate")
    fold: int | str = Field(default=0, description="Fold to evaluate (0-4 or 'all')")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability threshold for binarization")
    metrics: list[str] = Field(
        default=["dice", "sensitivity", "specificity", "precision", "hd95"],
        description="Metrics to compute",
    )
    output_dir: str | Path = Field(default="outputs/evaluation", description="Directory to save evaluation reports")
    predictions_dir: str | Path | None = Field(default=None, description="Directory containing existing predictions")
    dataset_id: str = Field(default="101", description="Dataset identifier (101, 102, 103, or all)")
    save_per_subject: bool = Field(default=True, description="Save per-subject metrics to CSV")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        valid_metrics = {"dice", "sensitivity", "specificity", "precision", "hd95", "recall"}
        for m in v:
            if m.lower() not in valid_metrics:
                raise ValueError(f"Unsupported metric: '{m}'. Supported: {sorted(valid_metrics)}")
        return [m.lower() for m in v]

    @classmethod
    def from_yaml(cls, path: Path | str) -> EvaluationConfig:
        data = load_yaml(path)
        return cls(**data)
