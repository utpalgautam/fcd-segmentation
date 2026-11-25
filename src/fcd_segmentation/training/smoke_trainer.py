import torch
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from fcd_segmentation.training import trainers as _trainers


class FCDnnUNetTrainer_Smoke3epochs(_trainers.FCDnnUNetTrainer_100epochs):
    """Low-memory 100-epoch trainer using the previous experiment's optimizer."""

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=None,
    ):
        if device is None:
            super().__init__(plans, configuration, fold, dataset_json)
        else:
            super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 3
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.save_every = 1

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            momentum=0.99,
            nesterov=True,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            self.num_epochs,
        )

        return optimizer, lr_scheduler
