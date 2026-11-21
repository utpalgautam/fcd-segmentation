"""Orchestration of nnU-Net v2 dataset preparation, cross-validation, and training workflow."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fcd_segmentation.training.cross_validation import (
    check_and_enforce_splits_for_training,
    export_splits_to_nnunet,
    generate_stratified_splits,
    validate_splits,
)
from fcd_segmentation.training.dataset_validator import (
    DATASET_MAP,
    check_training_guard,
    compare_cohorts,
    normalize_dataset_selector,
    validate_dataset,
)
from fcd_segmentation.training.paths import (
    NNUNetPaths,
    ensure_nnunet_directories,
    resolve_nnunet_paths,
)
from fcd_segmentation.training.reproducibility import set_seed
from fcd_segmentation.training.runner import CommandResult, find_executable, run_command
from fcd_segmentation.training.trainers import verify_trainer_resolution

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSummary:
    action: str
    timestamp: str
    datasets: list[str]
    paths: dict[str, str]
    validation_results: list[dict[str, Any]]
    cohort_comparison_issues: list[str]
    guard_passed: bool
    guard_message: str
    commands_executed: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None
    artifacts_detected: dict[str, list[str]] = field(default_factory=dict)
    splits_exported: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "timestamp": self.timestamp,
            "datasets": self.datasets,
            "paths": self.paths,
            "validation_results": self.validation_results,
            "cohort_comparison_issues": self.cohort_comparison_issues,
            "guard_passed": self.guard_passed,
            "guard_message": self.guard_message,
            "commands_executed": self.commands_executed,
            "success": self.success,
            "error_message": self.error_message,
            "artifacts_detected": self.artifacts_detected,
            "splits_exported": self.splits_exported,
            "hyperparameters": self.hyperparameters,
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def check_environment(paths: NNUNetPaths) -> dict[str, Any]:
    """Inspect and report Python, PyTorch, CUDA, nnU-Net, and effective paths."""
    import importlib.metadata
    import platform

    status: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "paths": paths.to_env_dict(),
        "packages": {},
    }

    for pkg in ("numpy", "PyYAML", "nibabel", "scipy", "scikit-learn", "torch", "torchvision", "nnunetv2"):
        try:
            status["packages"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            status["packages"][pkg] = None

    try:
        import torch

        status["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["cuda_device_count"] = torch.cuda.device_count()
            status["cuda_device_name"] = torch.cuda.get_device_name(0)
            status["cuda_memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            status["cuda_memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
        else:
            status["cuda_device_count"] = 0
            status["cuda_device_name"] = None
    except ImportError:
        status["cuda_available"] = False
        status["cuda_device_count"] = 0
        status["cuda_device_name"] = None

    return status


def prepare_datasets(
    dataset_ids: list[str],
    paths: NNUNetPaths,
    configurations: list[str] | None = None,
    num_processes: list[int] | int | None = None,
    verify_dataset_integrity: bool = True,
    clean: bool = False,
    log_file: Path | None = None,
) -> tuple[bool, list[CommandResult]]:
    """Invoke nnUNetv2_plan_and_preprocess for selected dataset IDs."""
    ensure_nnunet_directories(paths)

    exe = find_executable("nnUNetv2_plan_and_preprocess")
    cmd = [exe, "-d"] + [str(d) for d in dataset_ids]

    if verify_dataset_integrity:
        cmd.append("--verify_dataset_integrity")
    if clean:
        cmd.append("--clean")

    if configurations:
        cmd.append("-c")
        cmd.extend(configurations)

    if num_processes is not None:
        cmd.append("-np")
        if isinstance(num_processes, list):
            cmd.extend([str(p) for p in num_processes])
        else:
            cmd.append(str(num_processes))

    logger.info("Starting nnU-Net plan and preprocess for dataset(s): %s", dataset_ids)
    res = run_command(cmd, paths=paths, log_file=log_file)
    return res.success, [res]


def build_training_command(
    dataset_id: str,
    configuration: str,
    fold: int,
    trainer: str | None = None,
    plans: str | None = None,
    continue_training: bool = False,
    device: str | None = None,
    disable_checkpointing: bool = False,
) -> list[str]:
    """Build nnUNetv2_train command argument list based on verified CLI options."""
    exe = find_executable("nnUNetv2_train")
    cmd = [exe, str(dataset_id), str(configuration), str(fold)]

    if trainer:
        cmd.extend(["-tr", str(trainer)])
    if plans:
        cmd.extend(["-p", str(plans)])
    if continue_training:
        cmd.append("--c")
    if device:
        cmd.extend(["-device", str(device)])
    if disable_checkpointing:
        cmd.append("--disable_checkpointing")

    return cmd


def execute_training_fold(
    dataset_id: str,
    configuration: str,
    fold: int,
    paths: NNUNetPaths,
    trainer: str | None = None,
    plans: str | None = None,
    continue_training: bool = False,
    device: str | None = None,
    disable_checkpointing: bool = False,
    log_file: Path | None = None,
) -> CommandResult:
    """Execute a single fold training command with nnU-Net."""
    ensure_nnunet_directories(paths)

    cmd = build_training_command(
        dataset_id=dataset_id,
        configuration=configuration,
        fold=fold,
        trainer=trainer,
        plans=plans,
        continue_training=continue_training,
        device=device,
        disable_checkpointing=disable_checkpointing,
    )

    logger.info(
        "Starting training: Dataset %s | Config %s | Fold %d (continue=%s)",
        dataset_id,
        configuration,
        fold,
        continue_training,
    )
    return run_command(cmd, paths=paths, log_file=log_file)


def run_training_workflow(
    action: str,  # 'check', 'validate', 'prepare', 'train', 'split'
    dataset_arg: str | int = "101",
    fold_arg: int | str | None = 0,
    all_folds: bool = False,
    configuration: str = "3d_fullres",
    continue_training: bool = False,
    keep_going: bool = False,
    device: str | None = None,
    trainer: str = "FCDnnUNetTrainer_100epochs",
    plans: str | None = None,
    verify_dataset_integrity: bool = True,
    clean_preprocess: bool = False,
    paths: NNUNetPaths | None = None,
    logs_dir: Path | None = None,
    min_cases_guard: int = 5,
    seed: int = 42,
    participants_tsv: Path | None = None,
    export_splits: bool = True,
    batch_size: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
) -> WorkflowSummary:
    """Master workflow orchestrator for model training, preparation, CV splits, and checks."""
    effective_paths = paths or resolve_nnunet_paths()
    set_seed(seed)
    now_iso = datetime.now(UTC).isoformat()
    norm_dataset_ids = normalize_dataset_selector(dataset_arg)

    run_log_dir = logs_dir or (effective_paths.project_root / "outputs" / "training")
    run_log_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = run_log_dir / f"{action}_{timestamp_slug}.log"
    summary_file = run_log_dir / f"{action}_{timestamp_slug}_summary.json"

    # Default participants.tsv path
    tsv_path = participants_tsv or (effective_paths.project_root / "data" / "raw" / "dataset" / "participants.tsv")

    # 1. Validate datasets
    validation_results = []
    for d_id in norm_dataset_ids:
        v_res = validate_dataset(
            raw_root=effective_paths.raw,
            dataset_id=d_id,
            participants_tsv_path=tsv_path if tsv_path.is_file() else None,
            check_nifti_headers=True,
            check_label_values=True,
            require_test_split=True,
        )
        validation_results.append(v_res)

    cohort_issues = compare_cohorts(validation_results)
    guard_passed, guard_msg = check_training_guard(
        validation_results, min_cases_for_five_fold=min_cases_guard
    )

    dataset_names = [DATASET_MAP[d] for d in norm_dataset_ids]

    summary = WorkflowSummary(
        action=action,
        timestamp=now_iso,
        datasets=dataset_names,
        paths=effective_paths.to_env_dict(),
        validation_results=[r.to_dict() for r in validation_results],
        cohort_comparison_issues=cohort_issues,
        guard_passed=guard_passed,
        guard_message=guard_msg,
        hyperparameters={
            "seed": seed,
            "trainer": trainer,
            "plans": plans,
            "configuration": configuration,
            "batch_size_note": "Defined by plans.json architecture in nnU-Net v2",
            "epochs_note": "Configured in FCDnnUNetTrainer_100epochs (100 epochs)",
            "learning_rate_note": "Configured in FCDnnUNetTrainer_100epochs (1e-3 initial lr)",
            "configured_batch_size": batch_size,
            "configured_epochs": epochs,
            "configured_learning_rate": learning_rate,
            "device": device,
        },
    )

    # ACTION: CHECK
    if action == "check":
        env_info = check_environment(effective_paths)
        summary.artifacts_detected["environment"] = [str(env_info)]
        summary.success = all(r.valid for r in validation_results) and len(cohort_issues) == 0
        summary.save_json(summary_file)
        return summary

    # ACTION: VALIDATE
    if action == "validate":
        summary.success = all(r.valid for r in validation_results) and len(cohort_issues) == 0
        summary.save_json(summary_file)
        return summary

    # ACTION: SPLIT (Generate and export stratified splits)
    if action == "split":
        if not tsv_path.is_file():
            summary.success = False
            summary.error_message = f"participants.tsv not found at {tsv_path}"
            summary.save_json(summary_file)
            return summary

        try:
            splits = generate_stratified_splits(tsv_path, seed=seed, n_splits=5)
            written = export_splits_to_nnunet(splits, effective_paths.preprocessed, dataset_names=dataset_names)
            summary.splits_exported = [str(p) for p in written]
            # Verify exported files
            check_and_enforce_splits_for_training(dataset_names, effective_paths.preprocessed, tsv_path)
            summary.success = True
        except Exception as e:
            summary.success = False
            summary.error_message = f"Split generation failed: {e}"

        summary.save_json(summary_file)
        return summary

    # ACTION: PREPARE
    if action == "prepare":
        if not all(r.valid for r in validation_results):
            summary.success = False
            summary.error_message = "Dataset validation failed before preparation."
            summary.save_json(summary_file)
            return summary

        if cohort_issues:
            summary.success = False
            summary.error_message = f"Cohort mismatch across datasets: {'; '.join(cohort_issues)}"
            summary.save_json(summary_file)
            return summary

        if not guard_passed:
            logger.warning(
                "Running preparation on dataset with fewer than %d cases (%s).",
                min_cases_guard,
                guard_msg,
            )

        success, results = prepare_datasets(
            dataset_ids=norm_dataset_ids,
            paths=effective_paths,
            verify_dataset_integrity=verify_dataset_integrity,
            clean=clean_preprocess,
            log_file=log_file,
        )
        summary.commands_executed = [r.to_dict() for r in results]
        summary.success = success
        if not success:
            summary.error_message = "nnUNetv2_plan_and_preprocess failed."

        # Automatically export and enforce group-stratified splits after preparation
        if success and export_splits and tsv_path.is_file():
            try:
                splits = generate_stratified_splits(tsv_path, seed=seed, n_splits=5)
                written = export_splits_to_nnunet(splits, effective_paths.preprocessed, dataset_names=dataset_names)
                summary.splits_exported = [str(p) for p in written]
                check_and_enforce_splits_for_training(dataset_names, effective_paths.preprocessed, tsv_path)
                logger.info("Successfully exported and verified stratified splits for datasets: %s", dataset_names)
            except Exception as e:
                summary.success = False
                summary.error_message = f"Split generation/verification failed after planning: {e}"

        summary.save_json(summary_file)
        return summary

    # ACTION: TRAIN
    if action == "train":
        # 1. Dataset validity and guard check
        if not guard_passed:
            logger.error("Training blocked by guard: %s", guard_msg)
            summary.success = False
            summary.error_message = guard_msg
            summary.save_json(summary_file)
            return summary

        # 2. Validate fold arguments strictly
        valid_folds = {0, 1, 2, 3, 4}
        if all_folds:
            folds_to_run = [0, 1, 2, 3, 4]
        else:
            if fold_arg is None:
                folds_to_run = [0]
            elif isinstance(fold_arg, str) and fold_arg.lower() == "all":
                folds_to_run = [0, 1, 2, 3, 4]
            else:
                try:
                    f_int = int(fold_arg)
                    if f_int not in valid_folds:
                        raise ValueError(f"Fold index {f_int} out of range (must be 0, 1, 2, 3, or 4).")
                    folds_to_run = [f_int]
                except ValueError as ve:
                    err_msg = f"Invalid fold argument '{fold_arg}': {ve}"
                    logger.error(err_msg)
                    summary.success = False
                    summary.error_message = err_msg
                    summary.save_json(summary_file)
                    return summary

        # 3. Block training if selected datasets have different cohorts
        if cohort_issues:
            err_msg = f"Cohort mismatch blocked training:\n  " + "\n  ".join(cohort_issues)
            logger.error(err_msg)
            summary.success = False
            summary.error_message = err_msg
            summary.save_json(summary_file)
            return summary

        # 4. Pre-training verification of custom trainer class identity
        try:
            verified_cls = verify_trainer_resolution(trainer, allow_builtin=False)
            logger.info("Pre-training trainer resolution verified: %s (%s)", verified_cls.__name__, verified_cls.__module__)
        except Exception as e:
            err_msg = f"Pre-training trainer resolution failed: {e}"
            logger.error(err_msg)
            summary.success = False
            summary.error_message = err_msg
            summary.save_json(summary_file)
            return summary

        # 5. Pre-training cross-validation split verification
        try:
            check_and_enforce_splits_for_training(dataset_names, effective_paths.preprocessed, tsv_path)
            logger.info("Pre-training cross-validation splits verified for datasets: %s", dataset_names)
        except Exception as e:
            err_msg = f"Pre-training cross-validation split verification failed: {e}"
            logger.error(err_msg)
            summary.success = False
            summary.error_message = err_msg
            summary.save_json(summary_file)
            return summary

        # 6. Execute training sequentially
        overall_success = True
        for d_id in norm_dataset_ids:
            for fold in folds_to_run:
                cmd_res = execute_training_fold(
                    dataset_id=d_id,
                    configuration=configuration,
                    fold=fold,
                    paths=effective_paths,
                    trainer=trainer,
                    plans=plans,
                    continue_training=continue_training,
                    device=device,
                    log_file=log_file,
                )
                summary.commands_executed.append(cmd_res.to_dict())

                if not cmd_res.success:
                    overall_success = False
                    if not keep_going:
                        summary.error_message = f"Training stopped due to failure on Dataset {d_id} Fold {fold}."
                        break
            if not overall_success and not keep_going:
                break

        summary.success = overall_success
        summary.save_json(summary_file)
        return summary

    raise ValueError(f"Unknown workflow action: {action}")
