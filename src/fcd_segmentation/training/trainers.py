"""Custom nnU-Net v2 trainer classes for FCD lesion segmentation.

Provides configurable training length variants (e.g. 100 epochs, 50 epochs).
"""

from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


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


class FCDnnUNetTrainer_50epochs(nnUNetTrainer):
    """50 epochs variant for rapid ablation and testing."""

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

