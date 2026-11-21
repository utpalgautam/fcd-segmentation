from pathlib import Path

import pytest
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from fcd_segmentation.training.trainers import (
    FCDnnUNetTrainer_50epochs,
    FCDnnUNetTrainer_100epochs,
    verify_trainer_resolution,
)


def test_custom_trainer_resolution_success():
    cls = verify_trainer_resolution("FCDnnUNetTrainer_100epochs")
    assert cls.__name__ == "FCDnnUNetTrainer_100epochs"
    assert issubclass(cls, nnUNetTrainer)

    cls_50 = verify_trainer_resolution("FCDnnUNetTrainer_50epochs")
    assert cls_50.__name__ == "FCDnnUNetTrainer_50epochs"
    assert issubclass(cls_50, nnUNetTrainer)

    cls_5 = verify_trainer_resolution("FCDnnUNetTrainer_5epochs")
    assert cls_5.__name__ == "FCDnnUNetTrainer_5epochs"
    assert issubclass(cls_5, nnUNetTrainer)

    cls_1 = verify_trainer_resolution("FCDnnUNetTrainer_1epoch")
    assert cls_1.__name__ == "FCDnnUNetTrainer_1epoch"
    assert issubclass(cls_1, nnUNetTrainer)


def test_builtin_trainer_collision_rejection():
    # Attempting to use the conflicting built-in name nnUNetTrainer_100epochs must be rejected
    with pytest.raises(ValueError, match="resolved to nnU-Net built-in class"):
        verify_trainer_resolution("nnUNetTrainer_100epochs")


def test_nonexistent_trainer_raises():
    with pytest.raises(RuntimeError, match="Could not resolve trainer"):
        verify_trainer_resolution("NonExistentTrainer_999epochs")
