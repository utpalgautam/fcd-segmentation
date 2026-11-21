"""nnU-Net v2 model inference interface for FCD lesion segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

from fcd_segmentation.training.paths import NNUNetPaths, resolve_nnunet_paths
from fcd_segmentation.training.runner import CommandResult, find_executable, run_command

logger = logging.getLogger(__name__)


def build_predict_command(
    input_folder: Path | str,
    output_folder: Path | str,
    dataset_name_or_id: str | int,
    configuration: str = "3d_fullres",
    folds: list[int | str] | int | str = 0,
    trainer: str | None = "FCDnnUNetTrainer_100epochs",
    plans: str | None = None,
    device: str | None = None,
    save_probabilities: bool = False,
    disable_tta: bool = False,
) -> list[str]:
    """Construct nnUNetv2_predict command arguments."""
    exe = find_executable("nnUNetv2_predict")
    cmd = [
        exe,
        "-i",
        str(input_folder),
        "-o",
        str(output_folder),
        "-d",
        str(dataset_name_or_id),
        "-c",
        str(configuration),
    ]

    # Folds
    if isinstance(folds, list):
        cmd.extend(["-f"] + [str(f) for f in folds])
    elif str(folds).lower() == "all":
        cmd.extend(["-f", "0", "1", "2", "3", "4"])
    else:
        cmd.extend(["-f", str(folds)])

    if trainer:
        cmd.extend(["-tr", str(trainer)])
    if plans:
        cmd.extend(["-p", str(plans)])
    if device:
        cmd.extend(["-device", str(device)])
    if save_probabilities:
        cmd.append("--save_probabilities")
    if disable_tta:
        cmd.append("--disable_tta")

    return cmd


def predict(
    input_folder: Path | str,
    output_folder: Path | str,
    dataset_name_or_id: str | int,
    configuration: str = "3d_fullres",
    folds: list[int | str] | int | str = 0,
    trainer: str | None = "FCDnnUNetTrainer_100epochs",
    plans: str | None = None,
    device: str | None = None,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    paths: NNUNetPaths | None = None,
    log_file: Path | None = None,
) -> CommandResult:
    """Run model prediction using nnUNetv2_predict with proper environment variables."""
    effective_paths = paths or resolve_nnunet_paths()
    out_dir = Path(output_folder).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_predict_command(
        input_folder=input_folder,
        output_folder=out_dir,
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        folds=folds,
        trainer=trainer,
        plans=plans,
        device=device,
        save_probabilities=save_probabilities,
        disable_tta=disable_tta,
    )

    logger.info("Executing prediction: %s", " ".join(cmd))
    return run_command(cmd, paths=effective_paths, log_file=log_file)
