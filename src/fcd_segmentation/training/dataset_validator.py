"""nnU-Net dataset mapping, validation, and guard logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Dataset mapping for FCD lesion segmentation
DATASET_MAP: dict[str, str] = {
    "101": "Dataset101_FLAIR",
    "102": "Dataset102_T1",
    "103": "Dataset103_Fusion",
    "Dataset101_FLAIR": "Dataset101_FLAIR",
    "Dataset102_T1": "Dataset102_T1",
    "Dataset103_Fusion": "Dataset103_Fusion",
}

DATASET_IDS: list[str] = ["101", "102", "103"]
DATASET_FOLDERS: list[str] = ["Dataset101_FLAIR", "Dataset102_T1", "Dataset103_Fusion"]

EXPECTED_CHANNELS: dict[str, dict[str, str]] = {
    "Dataset101_FLAIR": {"0": "FLAIR"},
    "Dataset102_T1": {"0": "T1"},
    "Dataset103_Fusion": {"0": "FLAIR", "1": "T1"},
}


@dataclass
class DatasetValidationResult:
    dataset_name: str
    dataset_id: str
    dataset_path: Path
    valid: bool
    num_training_cases: int = 0
    case_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_id": self.dataset_id,
            "dataset_path": str(self.dataset_path),
            "valid": self.valid,
            "num_training_cases": self.num_training_cases,
            "case_ids": self.case_ids,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def normalize_dataset_selector(dataset_arg: str | int) -> list[str]:
    """Given '101', 'all', 'Dataset101_FLAIR', etc., return normalized list of dataset IDs ['101', ...]."""
    arg_str = str(dataset_arg).strip().lower()
    if arg_str == "all":
        return ["101", "102", "103"]

    # Try direct or mapped key
    for k, v in DATASET_MAP.items():
        if k.lower() == arg_str or v.lower() == arg_str:
            # return numeric ID string
            if k in DATASET_IDS:
                return [k]
            # else match v
            for num_id in DATASET_IDS:
                if DATASET_MAP[num_id] == v:
                    return [num_id]

    raise ValueError(
        f"Unknown dataset identifier: '{dataset_arg}'. "
        f"Supported options: {DATASET_IDS} or 'all' (or names {DATASET_FOLDERS})"
    )


def validate_dataset(
    raw_root: Path,
    dataset_id: str,
    check_nifti_headers: bool = True,
    check_label_values: bool = False,
) -> DatasetValidationResult:
    """Validate an nnU-Net dataset structure, dataset.json, channel order, and case consistency."""
    if dataset_id not in DATASET_MAP:
        raise ValueError(f"Invalid dataset_id: {dataset_id}")

    folder_name = DATASET_MAP[dataset_id]
    dataset_dir = raw_root / folder_name
    result = DatasetValidationResult(
        dataset_name=folder_name,
        dataset_id=dataset_id,
        dataset_path=dataset_dir,
        valid=True,
    )

    if not dataset_dir.is_dir():
        result.valid = False
        result.errors.append(
            f"Dataset directory does not exist: {dataset_dir}. "
            f"Action: Run preprocessing (scripts/prepare_dataset.py) first."
        )
        return result

    # Check dataset.json
    dataset_json_path = dataset_dir / "dataset.json"
    if not dataset_json_path.is_file():
        result.valid = False
        result.errors.append(
            f"Missing dataset.json at {dataset_json_path}. "
            f"Action: Verify preprocessing generated dataset.json properly."
        )
        return result

    try:
        with dataset_json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            result.metadata = metadata
    except Exception as e:
        result.valid = False
        result.errors.append(
            f"dataset.json at {dataset_json_path} is invalid JSON: {e}. "
            f"Action: Repair or re-generate dataset.json."
        )
        return result

    # Check required fields
    required_fields = ["channel_names", "labels", "file_ending"]
    for field_name in required_fields:
        if field_name not in metadata:
            result.valid = False
            result.errors.append(
                f"dataset.json missing required field '{field_name}'. "
                f"Action: Re-generate dataset.json with complete metadata."
            )

    # Check channel names
    expected_channels = EXPECTED_CHANNELS.get(folder_name, {})
    actual_channels = metadata.get("channel_names", {})
    # Channel names keys might be strings in JSON
    for ch_idx, ch_name in expected_channels.items():
        actual_name = actual_channels.get(ch_idx) or actual_channels.get(int(ch_idx))
        if actual_name != ch_name:
            result.valid = False
            result.errors.append(
                f"Channel {ch_idx} mismatch in {dataset_json_path}: "
                f"expected '{ch_name}', got '{actual_name}'. "
                f"Action: Ensure modality channel assignment conforms to project specification."
            )

    # Check labels
    labels = metadata.get("labels", {})
    if "background" not in labels or labels["background"] != 0:
        result.valid = False
        result.errors.append(
            f"Invalid background label in dataset.json: expected 0, got {labels.get('background')}"
        )
    if "lesion" not in labels or labels["lesion"] != 1:
        result.valid = False
        result.errors.append(
            f"Invalid lesion label in dataset.json: expected 1, got {labels.get('lesion')}"
        )

    # Check imagesTr and labelsTr
    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"

    if not images_tr_dir.is_dir():
        result.valid = False
        result.errors.append(f"Missing imagesTr directory: {images_tr_dir}")
        return result

    if not labels_tr_dir.is_dir():
        result.valid = False
        result.errors.append(f"Missing labelsTr directory: {labels_tr_dir}")
        return result

    # Discover labels
    label_files = sorted(labels_tr_dir.glob("*.nii.gz"))
    if not label_files:
        result.valid = False
        result.errors.append(f"No label files (.nii.gz) found in {labels_tr_dir}")
        return result

    case_ids = []
    num_channels = len(expected_channels)

    for label_path in label_files:
        case_id = label_path.name[: -len(".nii.gz")]
        if case_id in case_ids:
            result.valid = False
            result.errors.append(f"Duplicate case ID detected: {case_id}")
        case_ids.append(case_id)

        # Check label file size
        if label_path.stat().st_size == 0:
            result.valid = False
            result.errors.append(f"Empty label file: {label_path}")

        # Check matching images for each channel
        for ch_idx in range(num_channels):
            img_name = f"{case_id}_{ch_idx:04d}.nii.gz"
            img_path = images_tr_dir / img_name
            if not img_path.is_file():
                result.valid = False
                result.errors.append(
                    f"Missing expected image channel file for case '{case_id}': {img_path}"
                )
            elif img_path.stat().st_size == 0:
                result.valid = False
                result.errors.append(f"Empty image file: {img_path}")

    # Check for orphan image files in imagesTr
    all_image_files = sorted(images_tr_dir.glob("*.nii.gz"))
    expected_image_names = {
        f"{case_id}_{ch_idx:04d}.nii.gz"
        for case_id in case_ids
        for ch_idx in range(num_channels)
    }
    for img_file in all_image_files:
        if img_file.name not in expected_image_names:
            result.warnings.append(
                f"Unexpected/orphan image file in imagesTr: {img_file.name}"
            )

    result.num_training_cases = len(case_ids)
    result.case_ids = case_ids

    # Compare numTraining in dataset.json
    meta_num_training = metadata.get("numTraining")
    if meta_num_training != len(case_ids):
        result.warnings.append(
            f"dataset.json numTraining ({meta_num_training}) does not match actual cases count ({len(case_ids)})"
        )

    # Optional NIfTI headers and labels check
    if check_nifti_headers and case_ids:
        try:
            import nibabel as nib
            import numpy as np

            # Inspect first case or all if feasible
            sample_case = case_ids[0]
            sample_label_path = labels_tr_dir / f"{sample_case}.nii.gz"
            lbl_img = nib.load(sample_label_path)
            lbl_shape = lbl_img.shape

            for ch_idx in range(num_channels):
                sample_img_path = images_tr_dir / f"{sample_case}_{ch_idx:04d}.nii.gz"
                img = nib.load(sample_img_path)
                if img.shape != lbl_shape:
                    result.valid = False
                    result.errors.append(
                        f"Shape mismatch for case {sample_case} channel {ch_idx}: "
                        f"image shape {img.shape} != label shape {lbl_shape}"
                    )

            if check_label_values:
                lbl_data = lbl_img.get_fdata()
                unique_vals = set(np.unique(lbl_data).tolist())
                allowed_vals = {0.0, 1.0}
                if not unique_vals.issubset(allowed_vals):
                    result.valid = False
                    result.errors.append(
                        f"Unexpected label values in {sample_label_path}: {unique_vals}. "
                        f"Allowed label values are strictly 0 (background) and 1 (lesion)."
                    )
        except Exception as e:
            result.warnings.append(f"Could not verify NIfTI headers via nibabel: {e}")

    return result


def compare_cohorts(results: list[DatasetValidationResult]) -> list[str]:
    """Compare case IDs across datasets and return discrepancy warnings/errors."""
    issues = []
    if len(results) < 2:
        return issues

    base_res = results[0]
    base_cases = set(base_res.case_ids)

    for res in results[1:]:
        curr_cases = set(res.case_ids)
        if base_cases != curr_cases:
            diff1 = base_cases - curr_cases
            diff2 = curr_cases - base_cases
            msg = (
                f"Cohort case-ID mismatch between {base_res.dataset_name} and {res.dataset_name}!\n"
                f"  In {base_res.dataset_name} but not {res.dataset_name}: {sorted(diff1) if diff1 else 'None'}\n"
                f"  In {res.dataset_name} but not {base_res.dataset_name}: {sorted(diff2) if diff2 else 'None'}"
            )
            issues.append(msg)

    return issues


def check_training_guard(
    validation_results: list[DatasetValidationResult],
    min_cases_for_five_fold: int = 5,
) -> tuple[bool, str]:
    """Check cohort size guard.

    Returns:
        (can_train, message)
    """
    for res in validation_results:
        if not res.valid:
            return (
                False,
                f"Dataset {res.dataset_name} is invalid: {'; '.join(res.errors)}",
            )

        if res.num_training_cases < min_cases_for_five_fold:
            return False, (
                f"COHORT SIZE GUARD BLOCKED TRAINING for {res.dataset_name}:\n"
                f"  Found only {res.num_training_cases} labeled training case(s). "
                f"At least {min_cases_for_five_fold} valid cases are required for five-fold cross-validation.\n"
                f"  The dataset currently contains smoke-test data (generated with --limit {res.num_training_cases}).\n"
                f"  Action: Run full preprocessing on the complete cohort (e.g., python scripts/prepare_dataset.py) "
                f"before running model training."
            )

    return True, "All selected datasets meet the minimum case count guard."
