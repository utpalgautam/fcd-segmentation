"""nnU-Net dataset mapping, comprehensive strict validation, and training guard logic.

Enforces strict checks on:
- dataset.json structure, required fields, channel names, and label definitions
- imagesTr, labelsTr, imagesTs, and labelsTs directory existence and completeness
- Channel file completeness for training and test cohorts
- Rejection of orphan or unexpected image files in imagesTr and imagesTs
- Non-empty, validly formatted, and readable NIfTI files
- Exact binary label values ({0, 1})
- Spatial shape and affine matrix compatibility between images and labels
- Inspection of test-set geometry as well as training geometry
- Comparison of exact train/test subject membership against participants.tsv
- Cross-dataset cohort synchronization (Dataset101 == Dataset102 == Dataset103)
- Training guard that safely blocks execution on incomplete or invalid data
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

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
    num_test_cases: int = 0
    case_ids: list[str] = field(default_factory=list)
    test_case_ids: list[str] = field(default_factory=list)
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
            "num_test_cases": self.num_test_cases,
            "case_ids": self.case_ids,
            "test_case_ids": self.test_case_ids,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def normalize_dataset_selector(dataset_arg: str | int) -> list[str]:
    """Given '101', 'all', 'Dataset101_FLAIR', etc., return normalized list of dataset IDs ['101', ...]."""
    arg_str = str(dataset_arg).strip().lower()
    if arg_str == "all":
        return ["101", "102", "103"]

    for k, v in DATASET_MAP.items():
        if k.lower() == arg_str or v.lower() == arg_str:
            if k in DATASET_IDS:
                return [k]
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
    participants_tsv_path: Path | None = None,
    check_nifti_headers: bool = True,
    check_label_values: bool = True,
    max_cases_to_inspect: int = 10,
    require_test_split: bool = True,
) -> DatasetValidationResult:
    """Perform strict validation on an nnU-Net dataset directory and metadata.

    Validates:
    - dataset.json required fields, channel names, and label assignments
    - imagesTr, labelsTr, imagesTs, and labelsTs directory presence
    - Channel completeness for every training and test case
    - Rejection of orphan or unexpected image files in imagesTr and imagesTs
    - Non-empty and readable NIfTI files
    - Shape and affine compatibility between images and labels for both splits
    - Discrete binary label values strictly in {0, 1}
    - Exact train/test subject matching against participants.tsv
    """
    if dataset_id not in DATASET_MAP:
        raise ValueError(f"Invalid dataset_id: {dataset_id}")

    folder_name = DATASET_MAP[dataset_id]
    dataset_dir = (raw_root / folder_name).resolve()
    result = DatasetValidationResult(
        dataset_name=folder_name,
        dataset_id=dataset_id,
        dataset_path=dataset_dir,
        valid=True,
    )

    if not dataset_dir.is_dir():
        result.valid = False
        result.errors.append(f"Dataset directory does not exist: {dataset_dir}")
        return result

    # 1. Check dataset.json
    dataset_json_path = dataset_dir / "dataset.json"
    if not dataset_json_path.is_file():
        result.valid = False
        result.errors.append(f"Missing dataset.json at {dataset_json_path}")
        return result

    try:
        with dataset_json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            result.metadata = metadata
    except Exception as e:
        result.valid = False
        result.errors.append(f"dataset.json is invalid JSON: {e}")
        return result

    required_fields = ["channel_names", "labels", "file_ending"]
    for field_name in required_fields:
        if field_name not in metadata:
            result.valid = False
            result.errors.append(f"dataset.json missing required field '{field_name}'")

    expected_channels = EXPECTED_CHANNELS.get(folder_name, {})
    actual_channels = metadata.get("channel_names", {})
    for ch_idx, ch_name in expected_channels.items():
        actual_name = actual_channels.get(ch_idx) or actual_channels.get(int(ch_idx))
        if actual_name != ch_name:
            result.valid = False
            result.errors.append(
                f"Channel {ch_idx} mismatch in {dataset_json_path}: "
                f"expected '{ch_name}', got '{actual_name}'"
            )

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

    # 2. Check required directories
    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    images_ts_dir = dataset_dir / "imagesTs"
    labels_ts_dir = dataset_dir / "labelsTs"

    if not images_tr_dir.is_dir():
        result.valid = False
        result.errors.append(f"Missing required imagesTr directory: {images_tr_dir}")
    if not labels_tr_dir.is_dir():
        result.valid = False
        result.errors.append(f"Missing required labelsTr directory: {labels_tr_dir}")

    if require_test_split:
        if not images_ts_dir.is_dir():
            result.valid = False
            result.errors.append(f"Missing required imagesTs directory: {images_ts_dir}")
        if not labels_ts_dir.is_dir():
            result.valid = False
            result.errors.append(f"Missing required labelsTs directory: {labels_ts_dir}")

    if not result.valid:
        return result

    # 3. Validate training cohort files
    num_channels = len(expected_channels)
    tr_label_files = sorted(labels_tr_dir.glob("*.nii.gz"))
    if not tr_label_files:
        result.valid = False
        result.errors.append(f"No label files (.nii.gz) found in {labels_tr_dir}")

    case_ids: list[str] = []
    for label_path in tr_label_files:
        case_id = label_path.name[: -len(".nii.gz")]
        if case_id in case_ids:
            result.valid = False
            result.errors.append(f"Duplicate case ID detected in labelsTr: {case_id}")
        case_ids.append(case_id)

        if label_path.stat().st_size == 0:
            result.valid = False
            result.errors.append(f"Empty label file (0 bytes): {label_path}")

        # Ensure all channels exist for this case
        for ch_idx in range(num_channels):
            img_name = f"{case_id}_{ch_idx:04d}.nii.gz"
            img_path = images_tr_dir / img_name
            if not img_path.is_file():
                result.valid = False
                result.errors.append(
                    f"Missing expected image channel file for training case '{case_id}': {img_path}"
                )
            elif img_path.stat().st_size == 0:
                result.valid = False
                result.errors.append(f"Empty image file (0 bytes): {img_path}")

    # Reject orphan image files in imagesTr
    all_tr_images = sorted(images_tr_dir.glob("*.nii.gz"))
    expected_tr_images = {
        f"{case_id}_{ch_idx:04d}.nii.gz"
        for case_id in case_ids
        for ch_idx in range(num_channels)
    }
    for img_file in all_tr_images:
        if img_file.name not in expected_tr_images:
            result.valid = False
            result.errors.append(f"Unexpected/orphan image file in imagesTr: {img_file.name}")

    result.num_training_cases = len(case_ids)
    result.case_ids = case_ids

    meta_num_tr = metadata.get("numTraining")
    if meta_num_tr != len(case_ids):
        result.valid = False
        result.errors.append(
            f"dataset.json numTraining ({meta_num_tr}) does not match actual cases count ({len(case_ids)})"
        )

    # 4. Validate test cohort files
    test_case_ids: list[str] = []
    if images_ts_dir.is_dir() and labels_ts_dir.is_dir():
        ts_label_files = sorted(labels_ts_dir.glob("*.nii.gz"))
        if require_test_split and not ts_label_files:
            result.valid = False
            result.errors.append(f"No test label files (.nii.gz) found in {labels_ts_dir}")

        for ts_lbl in ts_label_files:
            ts_case_id = ts_lbl.name[: -len(".nii.gz")]
            if ts_case_id in test_case_ids:
                result.valid = False
                result.errors.append(f"Duplicate case ID detected in labelsTs: {ts_case_id}")
            test_case_ids.append(ts_case_id)

            if ts_lbl.stat().st_size == 0:
                result.valid = False
                result.errors.append(f"Empty test label file (0 bytes): {ts_lbl}")

            for ch_idx in range(num_channels):
                ts_img_name = f"{ts_case_id}_{ch_idx:04d}.nii.gz"
                ts_img_path = images_ts_dir / ts_img_name
                if not ts_img_path.is_file():
                    result.valid = False
                    result.errors.append(f"Missing test image file for case '{ts_case_id}': {ts_img_path}")
                elif ts_img_path.stat().st_size == 0:
                    result.valid = False
                    result.errors.append(f"Empty test image file (0 bytes): {ts_img_path}")

        # Reject orphan image files in imagesTs
        all_ts_images = sorted(images_ts_dir.glob("*.nii.gz"))
        expected_ts_images = {
            f"{ts_id}_{ch_idx:04d}.nii.gz"
            for ts_id in test_case_ids
            for ch_idx in range(num_channels)
        }
        for img_file in all_ts_images:
            if img_file.name not in expected_ts_images:
                result.valid = False
                result.errors.append(f"Unexpected/orphan image file in imagesTs: {img_file.name}")

        result.num_test_cases = len(test_case_ids)
        result.test_case_ids = test_case_ids

        # Verify strict disjointness between train and test
        leakage = set(case_ids).intersection(set(test_case_ids))
        if leakage:
            result.valid = False
            result.errors.append(f"CRITICAL: Train and test overlap detected for cases: {sorted(leakage)}")

    # 5. Check against participants.tsv
    tsv_candidate = participants_tsv_path or (raw_root.parents[0] / "raw" / "dataset" / "participants.tsv")
    if tsv_candidate and Path(tsv_candidate).is_file():
        try:
            with Path(tsv_candidate).open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)

            expected_train = {r["participant_id"].strip() for r in rows if r["split"].strip().lower() == "train"}
            expected_test = {r["participant_id"].strip() for r in rows if r["split"].strip().lower() == "test"}

            # Check exact training set membership
            missing_tr = expected_train - set(case_ids)
            extra_tr = set(case_ids) - expected_train
            if missing_tr:
                result.valid = False
                result.errors.append(f"Missing expected training subjects from imagesTr/labelsTr: {sorted(missing_tr)}")
            if extra_tr:
                result.valid = False
                result.errors.append(f"Unexpected subjects found in imagesTr/labelsTr: {sorted(extra_tr)}")

            # Check exact test set membership
            if require_test_split or test_case_ids:
                missing_ts = expected_test - set(test_case_ids)
                extra_ts = set(test_case_ids) - expected_test
                if missing_ts:
                    result.valid = False
                    result.errors.append(f"Missing expected test subjects from imagesTs/labelsTs: {sorted(missing_ts)}")
                if extra_ts:
                    result.valid = False
                    result.errors.append(f"Unexpected subjects found in imagesTs/labelsTs: {sorted(extra_ts)}")

        except Exception as e:
            result.valid = False
            result.errors.append(f"Failed to cross-reference against participants.tsv: {e}")

    # 6. Strict NIfTI header, geometry, shape, and affine inspection
    if check_nifti_headers:
        # Check both training and test sample cases
        inspection_groups = [
            ("training", case_ids[:max_cases_to_inspect], images_tr_dir, labels_tr_dir),
            ("test", test_case_ids[:max_cases_to_inspect], images_ts_dir, labels_ts_dir),
        ]

        for split_label, sample_ids, img_dir, lbl_dir in inspection_groups:
            for s_id in sample_ids:
                lbl_path = lbl_dir / f"{s_id}.nii.gz"
                try:
                    lbl_img = nib.load(lbl_path)
                    lbl_shape = lbl_img.shape
                    lbl_affine = lbl_img.affine
                except Exception as e:
                    result.valid = False
                    result.errors.append(f"Unreadable label NIfTI file for {split_label} case '{s_id}': {e}")
                    continue

                # Check channel images
                for ch_idx in range(num_channels):
                    img_path = img_dir / f"{s_id}_{ch_idx:04d}.nii.gz"
                    try:
                        img = nib.load(img_path)
                    except Exception as e:
                        result.valid = False
                        result.errors.append(f"Unreadable image NIfTI file for {split_label} case '{s_id}' channel {ch_idx}: {e}")
                        continue

                    # Shape check
                    if img.shape != lbl_shape:
                        result.valid = False
                        result.errors.append(
                            f"Shape mismatch for {split_label} case '{s_id}' channel {ch_idx}: "
                            f"image {img.shape} != label {lbl_shape}"
                        )

                    # Affine compatibility check
                    if not np.allclose(img.affine, lbl_affine, atol=1e-3, rtol=1e-3):
                        result.valid = False
                        result.errors.append(
                            f"Affine mismatch for {split_label} case '{s_id}' channel {ch_idx}: "
                            f"image affine does not match label affine"
                        )

                # Check label discrete values
                if check_label_values:
                    try:
                        lbl_data = np.asarray(lbl_img.dataobj)
                        unique_vals = set(np.unique(lbl_data).tolist())
                        allowed_vals = {0, 1, 0.0, 1.0}
                        if not unique_vals.issubset(allowed_vals):
                            result.valid = False
                            result.errors.append(
                                f"Unexpected label values in {lbl_path}: {unique_vals}. "
                                f"Allowed values are strictly 0 (background) and 1 (lesion)."
                            )
                    except Exception as e:
                        result.valid = False
                        result.errors.append(f"Could not read label voxel values from {lbl_path}: {e}")

    return result


def compare_cohorts(results: list[DatasetValidationResult]) -> list[str]:
    """Compare training and test case IDs across datasets to ensure strict synchronization."""
    issues: list[str] = []
    if len(results) < 2:
        return issues

    base_res = results[0]
    base_tr_cases = set(base_res.case_ids)
    base_ts_cases = set(base_res.test_case_ids)

    for res in results[1:]:
        curr_tr = set(res.case_ids)
        curr_ts = set(res.test_case_ids)

        if base_tr_cases != curr_tr:
            diff1 = base_tr_cases - curr_tr
            diff2 = curr_tr - base_tr_cases
            msg = (
                f"Training cohort case-ID mismatch between {base_res.dataset_name} and {res.dataset_name}!\n"
                f"  In {base_res.dataset_name} but not {res.dataset_name}: {sorted(diff1) if diff1 else 'None'}\n"
                f"  In {res.dataset_name} but not {base_res.dataset_name}: {sorted(diff2) if diff2 else 'None'}"
            )
            issues.append(msg)

        if base_ts_cases != curr_ts:
            diff1 = base_ts_cases - curr_ts
            diff2 = curr_ts - base_ts_cases
            msg = (
                f"Test cohort case-ID mismatch between {base_res.dataset_name} and {res.dataset_name}!\n"
                f"  In {base_res.dataset_name} but not {res.dataset_name}: {sorted(diff1) if diff1 else 'None'}\n"
                f"  In {res.dataset_name} but not {base_res.dataset_name}: {sorted(diff2) if diff2 else 'None'}"
            )
            issues.append(msg)

    return issues


def check_training_guard(
    validation_results: list[DatasetValidationResult],
    min_cases_for_five_fold: int = 5,
) -> tuple[bool, str]:
    """Check cohort size guard and validation status before executing training.

    Returns
    -------
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
                f"  Action: Run full preprocessing on the complete cohort before training."
            )

    return True, "All selected datasets meet the minimum case count guard."
