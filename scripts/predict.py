#!/usr/bin/env python3
"""CLI entry point for running nnU-Net v2 model inference on test and validation cases."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fcd_segmentation.inference.predict import predict
from fcd_segmentation.training.dataset_validator import (
    DATASET_MAP,
    normalize_dataset_selector,
)
from fcd_segmentation.training.paths import resolve_nnunet_paths
from fcd_segmentation.training.trainers import verify_trainer_resolution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("predict")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FCD nnU-Net v2 Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="101",
        help="Dataset identifier: '101' (FLAIR), '102' (T1), '103' (Fusion), or 'all'",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Cohort split to predict on: 'test' (held-out test set in imagesTs) or 'val' (validation fold in imagesTr)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory containing images to segment (defaults to imagesTs for test, imagesTr for val)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory to save predicted segmentation masks",
    )
    parser.add_argument(
        "--fold",
        default="0",
        help="Model fold checkpoint to use (0, 1, 2, 3, 4, or 'all' for ensemble)",
    )
    parser.add_argument(
        "--configuration",
        "-c",
        default="3d_fullres",
        help="nnU-Net configuration ('3d_fullres', '2d', etc.)",
    )
    parser.add_argument(
        "--trainer",
        default="FCDnnUNetTrainer_100epochs",
        help="Trainer class that produced the checkpoint (must match training trainer)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use ('cuda', 'cpu', 'mps')",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save softmax probabilities (.npz files) alongside segmentations",
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Disable test-time augmentation (mirroring) for faster prediction",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Pre-inference verification of trainer identity
    try:
        verify_trainer_resolution(args.trainer)
    except Exception as e:
        logger.error("Trainer resolution failed for prediction: %s", e)
        return 1

    paths = resolve_nnunet_paths(project_root=PROJECT_ROOT)
    dataset_ids = normalize_dataset_selector(args.dataset)

    success = True
    for d_id in dataset_ids:
        d_name = DATASET_MAP[d_id]

        # Standardized input and output path defaults
        if args.split == "val":
            in_dir = args.input_dir or (paths.raw / d_name / "imagesTr")
            out_dir = args.output_dir or (paths.project_root / "outputs" / "predictions" / d_name / f"fold_{args.fold}")
        else:
            in_dir = args.input_dir or (paths.raw / d_name / "imagesTs")
            out_dir = args.output_dir or (paths.project_root / "outputs" / "predictions" / d_name / "test")

        if not Path(in_dir).is_dir():
            logger.error("Input directory does not exist: %s", in_dir)
            success = False
            continue

        res = predict(
            input_folder=in_dir,
            output_folder=out_dir,
            dataset_name_or_id=d_id,
            configuration=args.configuration,
            folds=args.fold,
            trainer=args.trainer,
            device=args.device,
            save_probabilities=args.save_probabilities,
            disable_tta=args.disable_tta,
            paths=paths,
        )

        if not res.success:
            logger.error("Inference failed for dataset %s (exit code %d)", d_id, res.exit_code)
            success = False
        else:
            logger.info("Inference completed successfully for %s -> %s", d_name, out_dir)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
