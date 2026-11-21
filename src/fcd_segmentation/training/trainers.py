"""Custom nnU-Net v2 trainer classes for FCD lesion segmentation.

Provides configurable training length variants (e.g. 100 epochs, 50 epochs)
with adjusted learning rate schedules and discovery verification.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Type

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

logger = logging.getLogger(__name__)


class FCDnnUNetTrainer_100epochs(nnUNetTrainer):
    """FCD project trainer configuring training length to exactly 100 epochs
    instead of nnU-Net default 1000 epochs. Automatically adjusts the polynomial
    learning rate schedule to decay to 0 at epoch 100.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.initial_lr = 1e-3
        self.save_every = 25


class nnUNetTrainer_100epochs(FCDnnUNetTrainer_100epochs):
    """Discovery alias for configuration files specifying 'nnUNetTrainer_100epochs'."""

    pass


class FCDnnUNetTrainer_50epochs(nnUNetTrainer):
    """50 epochs variant for rapid ablation, testing, and CI verification."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.initial_lr = 1e-3
        self.save_every = 10


class nnUNetTrainer_50epochs(FCDnnUNetTrainer_50epochs):
    """Discovery alias for configuration files specifying 'nnUNetTrainer_50epochs'."""

    pass


class FCDnnUNetTrainer_5epochs(nnUNetTrainer):
    """5 epochs variant for rapid verification and smoke testing."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.initial_lr = 1e-3
        self.save_every = 1


class FCDnnUNetTrainer_1epoch(nnUNetTrainer):
    """1 epoch variant for smoke testing the end-to-end training pipeline."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.initial_lr = 1e-3
        self.save_every = 1


def verify_trainer_resolution(
    trainer_name: str,
    allow_builtin: bool = False,
) -> Type[nnUNetTrainer]:
    """Resolve and verify trainer class identity before training or prediction.

    Ensures that custom project trainers (e.g. FCDnnUNetTrainer_100epochs) are
    discovered properly from the project package via nnUNet_extTrainer and do
    not unintentionally resolve to nnU-Net built-in classes with conflicting names.

    Parameters
    ----------
    trainer_name : str
        The requested trainer class name.
    allow_builtin : bool
        If False, rejects trainers that resolve to built-in nnU-Net classes
        when custom functionality was expected.

    Returns
    -------
    The resolved trainer class.

    Raises
    ------
    ValueError or RuntimeError if the trainer cannot be found or resolves to an unintended class.
    """
    # Ensure nnUNet_extTrainer is configured in current process
    training_pkg_dir = str(Path(__file__).parent.resolve())
    existing_ext = os.environ.get("nnUNet_extTrainer", "")
    if training_pkg_dir not in existing_ext.split(os.pathsep):
        if existing_ext:
            os.environ["nnUNet_extTrainer"] = f"{training_pkg_dir}{os.pathsep}{existing_ext}"
        else:
            os.environ["nnUNet_extTrainer"] = training_pkg_dir

    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

    try:
        cls = recursive_find_trainer_class_by_name(trainer_name)
    except Exception as e:
        raise RuntimeError(f"Could not resolve trainer '{trainer_name}': {e}") from e

    if cls is None:
        raise RuntimeError(f"Trainer '{trainer_name}' could not be located by nnU-Net discovery.")

    module_name = getattr(cls, "__module__", "")

    # Check for built-in collision on nnUNetTrainer_100epochs
    if trainer_name == "nnUNetTrainer_100epochs" and module_name.startswith("nnunetv2."):
        raise ValueError(
            f"Requested trainer '{trainer_name}' resolved to nnU-Net built-in class "
            f"'{module_name}.{cls.__name__}'. This built-in class does not contain the FCD project "
            f"customizations (initial_lr=1e-3, save_every=25). "
            f"Fix: Use 'FCDnnUNetTrainer_100epochs' instead."
        )

    # If an FCD-prefixed custom trainer was requested, verify it was loaded from the project
    if trainer_name.startswith("FCD") and module_name.startswith("nnunetv2."):
        raise RuntimeError(
            f"Custom trainer '{trainer_name}' resolved to unintended built-in module '{module_name}'. "
            f"Expected it to load from project package '{training_pkg_dir}'."
        )

    if not allow_builtin and module_name.startswith("nnunetv2.") and trainer_name != "nnUNetTrainer":
        raise ValueError(
            f"Trainer '{trainer_name}' resolved to built-in class '{module_name}.{cls.__name__}'. "
            f"If you intended to use the FCD custom trainer, specify 'FCDnnUNetTrainer_100epochs'."
        )

    logger.info("Verified trainer '%s' -> %s.%s", trainer_name, module_name, cls.__name__)
    return cls


__all__ = [
    "FCDnnUNetTrainer_100epochs",
    "nnUNetTrainer_100epochs",
    "FCDnnUNetTrainer_50epochs",
    "nnUNetTrainer_50epochs",
    "FCDnnUNetTrainer_5epochs",
    "FCDnnUNetTrainer_1epoch",
    "verify_trainer_resolution",
]
