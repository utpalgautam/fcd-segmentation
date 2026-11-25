"""Reproducible group-stratified cross-validation splitting and nnU-Net splits export.

Ensures:
- participants.tsv has unique IDs, non-empty fields, and valid group ('fcd', 'hc') / split ('train', 'test') assignments
- 5-fold cross-validation is stratified on clinical group (FCD vs HC)
- Held-out test cases (split == 'test') are strictly excluded from all folds
- Every fold's training IDs are exactly expected_train_ids minus that fold's validation IDs
- No duplicate IDs within any train or validation fold
- Rejection of missing and unexpected IDs in every fold
- Identical splits are generated, exported, and verified across Dataset101, Dataset102, and Dataset103
- Pre-training validation blocks execution if splits_final.json is missing, stale, or inconsistent
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def validate_participants_tsv(tsv_path: Path | str) -> tuple[bool, list[str], list[dict[str, str]]]:
    """Strictly validate participants.tsv schema, uniqueness, and allowed values.

    Returns
    -------
    (is_valid, list_of_issues, parsed_rows)
    """
    path = Path(tsv_path).resolve()
    issues: list[str] = []
    if not path.is_file():
        return False, [f"participants.tsv not found at {path}"], []

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        return False, [f"participants.tsv is empty: {path}"], []

    required_cols = {"participant_id", "group", "split"}
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        return False, [f"participants.tsv missing required columns: {sorted(missing_cols)}"], []

    seen_ids: set[str] = set()
    allowed_groups = {"fcd", "hc"}
    allowed_splits = {"train", "test"}

    for idx, row in enumerate(rows, start=2):  # 1-indexed header is line 1
        pid = row.get("participant_id", "").strip()
        grp = row.get("group", "").strip().lower()
        splt = row.get("split", "").strip().lower()

        if not pid:
            issues.append(f"Line {idx}: empty or missing participant_id")
            continue

        if pid in seen_ids:
            issues.append(f"Line {idx}: duplicate participant_id '{pid}'")
        seen_ids.add(pid)

        if not grp:
            issues.append(f"Line {idx} ({pid}): missing group value")
        elif grp not in allowed_groups:
            issues.append(f"Line {idx} ({pid}): unsupported group '{grp}'. Allowed: {sorted(allowed_groups)}")

        if not splt:
            issues.append(f"Line {idx} ({pid}): missing split value")
        elif splt not in allowed_splits:
            issues.append(f"Line {idx} ({pid}): unsupported split '{splt}'. Allowed: {sorted(allowed_splits)}")

    return len(issues) == 0, issues, rows


def generate_stratified_splits(
    participants_tsv_path: Path | str,
    seed: int = 42,
    n_splits: int = 5,
) -> list[dict[str, list[str]]]:
    """Generate reproducible group-stratified cross-validation folds from participants.tsv."""
    is_valid, tsv_issues, rows = validate_participants_tsv(participants_tsv_path)
    if not is_valid:
        raise ValueError("participants.tsv validation failed:\n  " + "\n  ".join(tsv_issues))

    train_rows = [r for r in rows if r["split"].strip().lower() == "train"]
    test_rows = [r for r in rows if r["split"].strip().lower() == "test"]
    test_ids = {r["participant_id"].strip() for r in test_rows}

    if not train_rows:
        raise ValueError("No training cases found with split == 'train' in participants.tsv")

    case_ids = np.array([r["participant_id"].strip() for r in train_rows])
    groups = np.array([r["group"].strip().lower() for r in train_rows])

    # Sort deterministically before splitting
    sort_order = np.argsort(case_ids)
    case_ids = case_ids[sort_order]
    groups = groups[sort_order]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits: list[dict[str, list[str]]] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(case_ids, groups)):
        train_cases = sorted(case_ids[train_idx].tolist())
        val_cases = sorted(case_ids[val_idx].tolist())

        # Strict safety checks
        train_leak = set(train_cases).intersection(test_ids)
        val_leak = set(val_cases).intersection(test_ids)
        if train_leak:
            raise RuntimeError(f"Fold {fold} train split contains test cases: {train_leak}")
        if val_leak:
            raise RuntimeError(f"Fold {fold} val split contains test cases: {val_leak}")

        overlap = set(train_cases).intersection(set(val_cases))
        if overlap:
            raise RuntimeError(f"Fold {fold} has overlapping train and val cases: {overlap}")

        splits.append({
            "train": train_cases,
            "val": val_cases,
        })

    # Validate generated splits before returning
    splits_valid, split_issues = validate_splits(
        splits,
        participants_tsv_path=participants_tsv_path,
        expected_n_splits=n_splits,
    )
    if not splits_valid:
        raise RuntimeError("Generated cross-validation splits failed validation:\n  " + "\n  ".join(split_issues))

    logger.info(
        "Generated and validated %d-fold stratified splits for %d training cases (%d FCD, %d HC)",
        n_splits,
        len(train_rows),
        sum(1 for g in groups if g == "fcd"),
        sum(1 for g in groups if g == "hc"),
    )
    return splits


def validate_splits(
    splits: list[dict[str, list[str]]],
    participants_tsv_path: Path | str | None = None,
    expected_n_splits: int = 5,
) -> tuple[bool, list[str]]:
    """Validate cross-validation splits for structural integrity, leakage, subject coverage, and stratification.

    Requirements enforced:
    - Reject duplicate IDs within each fold's train list.
    - Reject duplicate IDs within each fold's validation list.
    - Verify that every fold's training IDs are exactly expected_train_ids minus that fold's validation IDs.
    - Reject missing and unexpected IDs in each fold.
    - Verify that every training subject appears in validation exactly once across folds.
    - Verify train and validation subjects are disjoint in every fold.
    - Verify no held-out test subject appears in any fold.
    - Verify group stratification across folds.

    Returns
    -------
    (is_valid, list_of_issues)
    """
    issues: list[str] = []

    if not isinstance(splits, list):
        return False, ["Splits object must be a list of fold dicts"]

    if len(splits) != expected_n_splits:
        issues.append(f"Expected {expected_n_splits} folds, but found {len(splits)}")

    val_counts: dict[str, int] = {}

    for idx, fold in enumerate(splits):
        if not isinstance(fold, dict) or "train" not in fold or "val" not in fold:
            issues.append(f"Fold {idx} missing 'train' or 'val' key or is not a dict")
            continue

        tr = fold["train"]
        va = fold["val"]

        if not tr:
            issues.append(f"Fold {idx} has empty training set")
        if not va:
            issues.append(f"Fold {idx} has empty validation set")

        # 1. Reject duplicate IDs within fold's train list
        if len(tr) != len(set(tr)):
            seen_tr = set()
            dup_tr = {x for x in tr if x in seen_tr or seen_tr.add(x)}
            issues.append(f"Fold {idx} training set contains duplicate IDs: {sorted(dup_tr)}")

        # 2. Reject duplicate IDs within fold's val list
        if len(va) != len(set(va)):
            seen_va = set()
            dup_va = {x for x in va if x in seen_va or seen_va.add(x)}
            issues.append(f"Fold {idx} validation set contains duplicate IDs: {sorted(dup_va)}")

        # 3. Disjointness within fold
        overlap = set(tr).intersection(set(va))
        if overlap:
            issues.append(f"Fold {idx} has overlapping cases in train and val: {sorted(overlap)}")

        for c in va:
            val_counts[c] = val_counts.get(c, 0) + 1

    # Check each validation case appears exactly once across folds
    for c, cnt in val_counts.items():
        if cnt != 1:
            issues.append(f"Case '{c}' appears in validation sets {cnt} times across folds (expected exactly 1)")

    if participants_tsv_path is not None:
        tsv_path = Path(participants_tsv_path).resolve()
        if tsv_path.is_file():
            is_valid, tsv_issues, rows = validate_participants_tsv(tsv_path)
            if not is_valid:
                issues.extend([f"participants.tsv invalid: {err}" for err in tsv_issues])
                return False, issues

            expected_train = {r["participant_id"].strip() for r in rows if r["split"].strip().lower() == "train"}
            expected_test = {r["participant_id"].strip() for r in rows if r["split"].strip().lower() == "test"}
            id_to_group = {r["participant_id"].strip(): r["group"].strip().lower() for r in rows}

            # 4. Leakage check
            for idx, fold in enumerate(splits):
                if "train" not in fold or "val" not in fold:
                    continue
                tr_leak = set(fold["train"]).intersection(expected_test)
                va_leak = set(fold["val"]).intersection(expected_test)
                if tr_leak:
                    issues.append(f"CRITICAL: Fold {idx} training set contains test cases: {sorted(tr_leak)}")
                if va_leak:
                    issues.append(f"CRITICAL: Fold {idx} validation set contains test cases: {sorted(va_leak)}")

            # 5. Exact fold training IDs check: must equal expected_train_ids minus that fold's validation IDs
            for idx, fold in enumerate(splits):
                if "train" not in fold or "val" not in fold:
                    continue
                tr_set = set(fold["train"])
                va_set = set(fold["val"])
                expected_fold_tr = expected_train - va_set

                if tr_set != expected_fold_tr:
                    missing_from_tr = expected_fold_tr - tr_set
                    unexpected_in_tr = tr_set - expected_fold_tr
                    if missing_from_tr:
                        issues.append(
                            f"Fold {idx} training set missing expected IDs (expected_train - fold_val): {sorted(missing_from_tr)}"
                        )
                    if unexpected_in_tr:
                        issues.append(
                            f"Fold {idx} training set contains unexpected IDs: {sorted(unexpected_in_tr)}"
                        )

            # 6. Check total validation set coverage against expected_train
            total_cv_cases = set(val_counts.keys())
            unexpected_va = total_cv_cases - expected_train
            if unexpected_va:
                issues.append(f"CV splits contain unexpected validation cases not in training cohort: {sorted(unexpected_va)}")

            missing_va = expected_train - total_cv_cases
            if missing_va:
                issues.append(f"Training cases missing from CV validation folds: {sorted(missing_va)}")

            # 7. Stratification check: verify balanced FCD and HC in every validation fold
            n_fcd_train = sum(1 for cid in expected_train if id_to_group.get(cid) == "fcd")
            n_hc_train = sum(1 for cid in expected_train if id_to_group.get(cid) == "hc")

            expected_fcd_per_fold = n_fcd_train / expected_n_splits
            expected_hc_per_fold = n_hc_train / expected_n_splits

            for idx, fold in enumerate(splits):
                if "val" not in fold:
                    continue
                va = fold["val"]
                fold_fcd = sum(1 for cid in va if id_to_group.get(cid) == "fcd")
                fold_hc = sum(1 for cid in va if id_to_group.get(cid) == "hc")

                if fold_fcd == 0:
                    issues.append(f"Fold {idx} validation set has 0 FCD cases (unstratified)")
                if fold_hc == 0:
                    issues.append(f"Fold {idx} validation set has 0 HC cases (unstratified)")

                if abs(fold_fcd - expected_fcd_per_fold) > 1.5:
                    issues.append(
                        f"Fold {idx} FCD count ({fold_fcd}) deviates significantly from expected ({expected_fcd_per_fold:.1f})"
                    )
                if abs(fold_hc - expected_hc_per_fold) > 1.5:
                    issues.append(
                        f"Fold {idx} HC count ({fold_hc}) deviates significantly from expected ({expected_hc_per_fold:.1f})"
                    )

    return len(issues) == 0, issues


def verify_dataset_splits_file(
    splits_file_path: Path | str,
    participants_tsv_path: Path | str,
) -> tuple[bool, list[str]]:
    """Verify that an existing splits_final.json file exists and is strictly valid."""
    path = Path(splits_file_path).resolve()
    if not path.is_file():
        return False, [f"Splits file does not exist: {path}"]

    try:
        with path.open("r", encoding="utf-8") as f:
            splits = json.load(f)
    except Exception as e:
        return False, [f"Splits file {path} is not valid JSON: {e}"]

    return validate_splits(splits, participants_tsv_path=participants_tsv_path)


def check_and_enforce_splits_for_training(
    dataset_names: list[str],
    preprocessed_dir: Path | str,
    participants_tsv_path: Path | str,
) -> None:
    """Pre-training guard: verifies splits_final.json for all datasets and checks consistency.

    Raises RuntimeError if any splits file is missing, invalid, stale, or if splits differ across datasets.
    """
    prep_dir = Path(preprocessed_dir).resolve()
    tsv_path = Path(participants_tsv_path).resolve()

    if not tsv_path.is_file():
        raise RuntimeError(f"participants.tsv not found at {tsv_path}. Cannot verify training splits.")

    splits_by_dataset: dict[str, list[dict[str, list[str]]]] = {}

    for d_name in dataset_names:
        split_file = prep_dir / d_name / "splits_final.json"
        is_valid, issues = verify_dataset_splits_file(split_file, tsv_path)
        if not is_valid:
            raise RuntimeError(
                f"Pre-training check failed for dataset '{d_name}': splits_final.json is missing or invalid!\n"
                f"Issues: {'; '.join(issues)}\n"
                f"Action: Run 'python scripts/train.py --split-folds --dataset all' to generate and export valid folds."
            )

        with split_file.open("r", encoding="utf-8") as f:
            splits_by_dataset[d_name] = json.load(f)

    # Check identical splits across all datasets
    if len(dataset_names) > 1:
        first_name = dataset_names[0]
        first_splits = splits_by_dataset[first_name]
        for other_name in dataset_names[1:]:
            other_splits = splits_by_dataset[other_name]
            if first_splits != other_splits:
                raise RuntimeError(
                    f"CRITICAL: Cross-validation splits mismatch between '{first_name}' and '{other_name}'!\n"
                    f"All models must be trained on identical subject splits.\n"
                    f"Action: Re-export splits with 'python scripts/train.py --split-folds --dataset all'."
                )

    logger.info("Verified splits_final.json across datasets %s: All folds valid and synchronized.", dataset_names)


def export_splits_to_nnunet(
    splits: list[dict[str, list[str]]],
    preprocessed_dir: Path | str,
    dataset_names: list[str] | None = None,
) -> list[Path]:
    """Export validated splits_final.json to nnUNet_preprocessed dataset directories."""
    if dataset_names is None:
        dataset_names = ["Dataset101_FLAIR", "Dataset102_T1", "Dataset103_Fusion"]

    prep_dir = Path(preprocessed_dir).resolve()
    written_files: list[Path] = []

    for d_name in dataset_names:
        target_dir = prep_dir / d_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "splits_final.json"

        with target_file.open("w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

        written_files.append(target_file)
        logger.info("Wrote splits_final.json for %s at %s", d_name, target_file)

    return written_files
