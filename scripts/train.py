#!/usr/bin/env python3
"""
CLI entry point for nnU-Net v2 model training workflow.

Supports:
- Environment and path checking (--check)
- Dataset validation (--validate)
- nnU-Net dataset preparation and planning (--prepare)
- Single-fold, 5-fold, and all-dataset cross-validation training
- Resuming interrupted training (--continue)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path so package imports resolve seamlessly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from fcd_segmentation.training.paths import resolve_nnunet_paths
from fcd_segmentation.training.workflow import check_environment, run_training_workflow


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_training_config(config_path: Path) -> dict:
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning("Could not read config file %s: %e", config_path, e)
    return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FCD nnU-Net v2 Model Training & Preparation Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="Inspect environment, PyTorch, CUDA, nnU-Net, paths, and datasets without running training.",
    )
    mode_group.add_argument(
        "--validate",
        action="store_true",
        help="Perform strict dataset validation across selected datasets.",
    )
    mode_group.add_argument(
        "--prepare",
        action="store_true",
        help="Run nnU-Net dataset integrity verification, fingerprint extraction, and planning/preprocessing.",
    )

    parser.add_argument(
        "--dataset",
        default="101",
        help="Dataset identifier: '101' (FLAIR), '102' (T1), '103' (Fusion), or 'all'.",
    )
    parser.add_argument(
        "--fold",
        default="0",
        help="Fold to train (0, 1, 2, 3, 4) or 'all'. Ignored if --all-folds is set.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Train all 5 folds (0, 1, 2, 3, 4) sequentially.",
    )
    parser.add_argument(
        "--configuration",
        "-c",
        default="3d_fullres",
        help="nnU-Net configuration to train (e.g. '3d_fullres', '2d', '3d_lowres').",
    )
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Resume interrupted training from the latest checkpoint (--c flag in nnU-Net).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Override training compute device ('cuda', 'cpu', 'mps').",
    )
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer",
        help="Custom trainer class name.",
    )
    parser.add_argument(
        "--plans",
        "-p",
        default=None,
        help="Custom plans identifier (default: nnUNetPlans).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training.yaml",
        help="Path to training YAML configuration.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override nnUNet_raw directory.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help="Override nnUNet_preprocessed directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override nnUNet_results directory.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with subsequent folds/datasets if one fails.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Overwrite existing fingerprints/plans during --prepare.",
    )
    parser.add_argument(
        "--min-cases-guard",
        type=int,
        default=5,
        help="Minimum number of labeled training cases required for cross-validation.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    config_data = load_training_config(args.config)

    # Resolve paths
    paths = resolve_nnunet_paths(
        project_root=PROJECT_ROOT,
        raw_dir=args.raw_dir or config_data.get("nnunet_raw"),
        preprocessed_dir=args.preprocessed_dir
        or config_data.get("nnunet_preprocessed"),
        results_dir=args.results_dir or config_data.get("nnunet_results"),
    )

    # Determine action
    if args.check:
        action = "check"
    elif args.validate:
        action = "validate"
    elif args.prepare:
        action = "prepare"
    else:
        action = "train"

    # Print check header if --check
    if action == "check":
        print("=== nnU-Net Environment & System Status ===")
        env_status = check_environment(paths)
        print(
            f"Python: {env_status['python_version']} ({env_status['python_executable']})"
        )
        print(f"Platform: {env_status['platform']}")
        print(
            f"CUDA Available: {env_status['cuda_available']} (Device: {env_status.get('cuda_device_name', 'None')})"
        )
        print("Packages:")
        for pkg, ver in env_status["packages"].items():
            print(f"  {pkg}: {ver or 'NOT INSTALLED'}")
        print("\nEffective nnU-Net Environment Paths:")
        for k, v in paths.to_env_dict().items():
            print(f"  ${k} -> {v}")
        print("==========================================\n")

    summary = run_training_workflow(
        action=action,
        dataset_arg=args.dataset,
        fold_arg=args.fold,
        all_folds=args.all_folds,
        configuration=args.configuration,
        continue_training=args.continue_training,
        keep_going=args.keep_going,
        device=args.device
        or (config_data.get("device") if config_data.get("device") != "auto" else None),
        trainer=args.trainer,
        plans=args.plans,
        clean_preprocess=args.clean,
        paths=paths,
        min_cases_guard=args.min_cases_guard,
    )

    # Human-readable console feedback
    print(
        f"\nWorkflow action '{action}' finished. Status: {'SUCCESS' if summary.success else 'FAILED'}"
    )
    print(f"Selected Datasets: {summary.datasets}")
    for vr in summary.validation_results:
        status_str = "VALID" if vr["valid"] else "INVALID"
        print(
            f"  [{status_str}] {vr['dataset_name']} ({vr['num_training_cases']} case(s)): {vr['dataset_path']}"
        )
        if vr["errors"]:
            for err in vr["errors"]:
                print(f"    ERROR: {err}")
        if vr["warnings"]:
            for warn in vr["warnings"]:
                print(f"    WARNING: {warn}")

    if summary.cohort_comparison_issues:
        print("\nCohort Discrepancies:")
        for issue in summary.cohort_comparison_issues:
            print(f"  {issue}")

    print(f"\nCohort Guard: {'PASSED' if summary.guard_passed else 'BLOCKED'}")
    print(f"Guard Details: {summary.guard_message}")

    if summary.commands_executed:
        print("\nCommands Executed:")
        for c in summary.commands_executed:
            print(
                f"  [{'SUCCESS' if c['success'] else 'FAILED'}] {' '.join(c['command'])} (Exit {c['exit_code']})"
            )

    if not summary.success and summary.error_message:
        print(f"\nError: {summary.error_message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
