#!/usr/bin/env python3
"""CLI entry point for FCD lesion segmentation evaluation and metric reporting.

Evaluates predicted segmentation masks against ground-truth labels across:
- Held-out test cohort (28 FCD subjects in labelsTs)
- Cross-validation folds (specific fold validation subjects in labelsTr)
- Baseline models: FLAIR (101), T1 (102), and Multimodal Fusion (103)

Outputs:
- Per-subject metrics CSV (Dice, Sensitivity, Specificity, Precision, HD95)
- Aggregate summary statistics JSON and CSV (mean, std, median, IQR, finite/infinite HD95 counts)
- Subgroup breakdown by clinical group (FCD vs HC)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fcd_segmentation.config import EvaluationConfig
from fcd_segmentation.evaluation.evaluator import DatasetEvaluator, compare_modalities
from fcd_segmentation.training.dataset_validator import (
    DATASET_MAP,
    normalize_dataset_selector,
)
from fcd_segmentation.training.paths import resolve_nnunet_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FCD Lesion Segmentation Evaluation & Benchmarking CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "evaluation.yaml",
        help="Path to evaluation YAML configuration",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset identifier: '101' (FLAIR), '102' (T1), '103' (Fusion), or 'all'",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default=None,
        help="Cohort split to evaluate: 'test' (held-out test set) or 'val' (cross-validation fold)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=None,
        help="Cross-validation fold index (0..4) to evaluate when --split val is used",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory containing predicted NIfTI files (*.nii.gz)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Directory containing ground truth label files (*.nii.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save evaluation reports and CSV files",
    )
    parser.add_argument(
        "--participants-tsv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dataset" / "participants.tsv",
        help="Path to participants.tsv for metadata and subgroup stratification",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Path to splits_final.json (defaults to data/nnunet_preprocessed/<dataset>/splits_final.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for converting continuous predictions to binary mask",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run cross-modality comparison across all three datasets (101, 102, 103)",
    )
    return parser


def find_default_predictions_dir(
    project_root: Path,
    d_name: str,
    split: str,
    fold: int | None,
    custom_pred_dir: Path | str | None = None,
) -> Path | None:
    """Find prediction directory matching default output conventions."""
    if custom_pred_dir:
        candidate = Path(custom_pred_dir).resolve()
        if (candidate / d_name).is_dir():
            return candidate / d_name
        if candidate.is_dir():
            return candidate

    pred_root = project_root / "outputs" / "predictions" / d_name

    if split == "val":
        fold_str = f"fold_{fold}" if fold is not None else "fold_0"
        candidates = [
            pred_root / fold_str,
            pred_root / "val",
            pred_root,
        ]
    else:  # test
        candidates = [
            pred_root / "test",
            pred_root,
        ]

    for cand in candidates:
        if cand.is_dir() and any(cand.glob("*.nii.gz")):
            return cand

    # Return primary default candidate even if not created yet (for error reporting)
    return candidates[0]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = resolve_nnunet_paths(project_root=PROJECT_ROOT)

    # 1. Load configuration from YAML if available
    cfg = None
    if args.config and args.config.is_file():
        try:
            cfg = EvaluationConfig.from_yaml(args.config)
            logger.info("Loaded evaluation config from %s", args.config)
        except Exception as e:
            logger.error("Failed to parse evaluation config %s: %s", args.config, e)
            return 1
    else:
        cfg = EvaluationConfig()

    # 2. CLI options take strict precedence over YAML configuration
    effective_split = args.split if args.split is not None else cfg.split
    effective_threshold = args.threshold if args.threshold is not None else cfg.threshold
    effective_output_dir = Path(args.output_dir or cfg.output_dir).resolve()
    effective_dataset_arg = args.dataset or cfg.dataset_id
    effective_fold = args.fold if args.fold is not None else (int(cfg.fold) if str(cfg.fold).isdigit() else 0)

    if effective_split == "val" and args.fold is None and cfg.fold is None:
        logger.warning("No fold specified for validation evaluation; defaulting to --fold 0.")
        effective_fold = 0

    effective_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = ["101", "102", "103"] if args.compare_all else normalize_dataset_selector(effective_dataset_arg)
    reports = []
    has_failure = False

    for d_id in dataset_ids:
        d_name = DATASET_MAP[d_id]
        fold_tag = f"fold_{effective_fold}" if effective_split == "val" else "test"
        logger.info("Evaluating dataset %s (%s) on '%s' (%s)...", d_id, d_name, effective_split, fold_tag)

        # Determine ground-truth directory
        if args.ground_truth_dir:
            gt_dir = Path(args.ground_truth_dir).resolve()
        else:
            # Validation cases are part of the training cohort stored in labelsTr
            split_folder = "labelsTs" if effective_split == "test" else "labelsTr"
            gt_dir = paths.raw / d_name / split_folder

        # Determine predictions directory with consistent path resolution
        if args.predictions_dir:
            pred_dir = Path(args.predictions_dir).resolve()
            if len(dataset_ids) > 1 and (pred_dir / d_name).is_dir():
                pred_dir = pred_dir / d_name
        else:
            pred_dir = find_default_predictions_dir(
                project_root=paths.project_root,
                d_name=d_name,
                split=effective_split,
                fold=effective_fold,
                custom_pred_dir=cfg.predictions_dir,
            )

        if pred_dir is None or not pred_dir.is_dir():
            logger.error(
                "Predictions directory does not exist: %s\n"
                "Please generate predictions first (e.g. using 'python scripts/predict.py --dataset %s --split %s') "
                "or specify --predictions-dir.",
                pred_dir,
                d_id,
                effective_split,
            )
            has_failure = True
            continue

        if not gt_dir.is_dir():
            logger.error("Ground-truth directory does not exist: %s", gt_dir)
            has_failure = True
            continue

        # Determine splits file for validation fold evaluation
        splits_file_path = args.splits_file
        if splits_file_path is None and effective_split == "val":
            splits_file_path = paths.preprocessed / d_name / "splits_final.json"

        try:
            report = DatasetEvaluator.evaluate_predictions_dir(
                name=d_name,
                predictions_dir=pred_dir,
                ground_truth_dir=gt_dir,
                participants_tsv=args.participants_tsv,
                split=effective_split,
                fold=effective_fold if effective_split == "val" else None,
                splits_file=splits_file_path,
                threshold=effective_threshold,
            )

            # Output filenames
            file_suffix = f"{effective_split}_{fold_tag}" if effective_split == "val" else effective_split
            csv_path = effective_output_dir / f"metrics_{d_name}_{file_suffix}.csv"
            json_path = effective_output_dir / f"summary_{d_name}_{file_suffix}.json"
            report.save_csv(csv_path)
            report.save_json(json_path)

            logger.info("Saved per-subject metrics to %s", csv_path)
            logger.info("Saved summary statistics to %s", json_path)

            print(f"\n=== Evaluation Summary for {d_name} ({effective_split} cohort: {report.num_cases} cases) ===")
            if report.scientific_notes:
                for note in report.scientific_notes:
                    print(f"  NOTE: {note}")

            for metric, stats in report.summary_statistics.items():
                if metric == "hd95" and stats.get("infinite_count", 0) > 0:
                    print(
                        f"  {metric:<15}: Finite-only Mean={stats['mean']:.4f} ± {stats['std']:.4f} | "
                        f"Median={stats['median']:.4f} [IQR: {stats['iqr_25']:.4f} - {stats['iqr_75']:.4f}] | "
                        f"Infinite/Failed={int(stats['infinite_count'])}/{int(stats['total_count'])} ({stats['infinite_percentage']}%)"
                    )
                else:
                    print(
                        f"  {metric:<15}: Mean={stats['mean']:.4f} ± {stats['std']:.4f} | "
                        f"Median={stats['median']:.4f} [IQR: {stats['iqr_25']:.4f} - {stats['iqr_75']:.4f}]"
                    )

            reports.append(report)

        except Exception as e:
            logger.error("Evaluation failed for %s: %s", d_name, e)
            has_failure = True

    if args.compare_all and len(reports) > 1:
        comp = compare_modalities(reports)
        file_suffix = f"{effective_split}_{fold_tag}" if effective_split == "val" else effective_split
        comp_path = effective_output_dir / f"modality_comparison_{file_suffix}.json"
        with comp_path.open("w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)
        logger.info("Saved cross-modality comparison to %s", comp_path)

    if has_failure or not reports:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
