"""Training module public exports."""

from fcd_segmentation.training.dataset_validator import (
    DATASET_FOLDERS,
    DATASET_IDS,
    DATASET_MAP,
    DatasetValidationResult,
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
from fcd_segmentation.training.runner import CommandResult, run_command
from fcd_segmentation.training.workflow import (
    WorkflowSummary,
    build_training_command,
    check_environment,
    prepare_datasets,
    run_training_workflow,
)

__all__ = [
    "DATASET_IDS",
    "DATASET_FOLDERS",
    "DATASET_MAP",
    "DatasetValidationResult",
    "validate_dataset",
    "compare_cohorts",
    "check_training_guard",
    "normalize_dataset_selector",
    "NNUNetPaths",
    "resolve_nnunet_paths",
    "ensure_nnunet_directories",
    "CommandResult",
    "run_command",
    "WorkflowSummary",
    "check_environment",
    "prepare_datasets",
    "build_training_command",
    "run_training_workflow",
]
