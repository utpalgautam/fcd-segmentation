from pathlib import Path

import pytest

from fcd_segmentation.training.cross_validation import (
    check_and_enforce_splits_for_training,
    export_splits_to_nnunet,
    generate_stratified_splits,
    validate_participants_tsv,
    validate_splits,
    verify_dataset_splits_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_PARTICIPANTS_TSV = PROJECT_ROOT / "data" / "raw" / "dataset" / "participants.tsv"


def create_mock_participants_tsv(
    path: Path,
    n_fcd_train: int = 6,
    n_hc_train: int = 4,
    n_fcd_test: int = 4,
    duplicate_id: bool = False,
    missing_group: bool = False,
    invalid_split: bool = False,
) -> Path:
    content = "participant_id\tgroup\tsplit\n"
    for i in range(1, n_fcd_train + 1):
        content += f"sub-fcd-tr-{i}\tfcd\ttrain\n"
    for i in range(1, n_hc_train + 1):
        content += f"sub-hc-tr-{i}\thc\ttrain\n"
    for i in range(1, n_fcd_test + 1):
        content += f"sub-fcd-ts-{i}\tfcd\ttest\n"

    if duplicate_id:
        content += "sub-fcd-tr-1\tfcd\ttrain\n"
    if missing_group:
        content += "sub-missing-grp\t\ttrain\n"
    if invalid_split:
        content += "sub-bad-split\tfcd\tunsupported_split\n"

    path.write_text(content, encoding="utf-8")
    return path


def test_validate_participants_tsv_valid(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv")
    is_valid, issues, rows = validate_participants_tsv(tsv)
    assert is_valid
    assert len(issues) == 0
    assert len(rows) == 14


def test_validate_participants_tsv_duplicate_id(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", duplicate_id=True)
    is_valid, issues, _ = validate_participants_tsv(tsv)
    assert not is_valid
    assert any("duplicate participant_id" in err for err in issues)


def test_validate_participants_tsv_missing_group(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", missing_group=True)
    is_valid, issues, _ = validate_participants_tsv(tsv)
    assert not is_valid
    assert any("missing group" in err for err in issues)


def test_validate_participants_tsv_invalid_split(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", invalid_split=True)
    is_valid, issues, _ = validate_participants_tsv(tsv)
    assert not is_valid
    assert any("unsupported split" in err for err in issues)


def test_generate_stratified_splits_synthetic(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", n_fcd_train=10, n_hc_train=10, n_fcd_test=4)
    splits = generate_stratified_splits(tsv, seed=42, n_splits=5)

    assert len(splits) == 5
    is_valid, issues = validate_splits(splits, participants_tsv_path=tsv, expected_n_splits=5)
    assert is_valid, f"Issues: {issues}"

    # Verify no test cases in train or val
    test_cases = {f"sub-fcd-ts-{i}" for i in range(1, 5)}
    for fold in splits:
        assert not set(fold["train"]).intersection(test_cases)
        assert not set(fold["val"]).intersection(test_cases)


def test_leakage_rejection(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", n_fcd_train=10, n_hc_train=10, n_fcd_test=4)
    splits = generate_stratified_splits(tsv, seed=42, n_splits=5)

    # Corrupt fold 0 by adding a test case into train
    splits[0]["train"].append("sub-fcd-ts-1")
    is_valid, issues = validate_splits(splits, participants_tsv_path=tsv, expected_n_splits=5)
    assert not is_valid
    assert any("contains test cases" in s for s in issues)


def test_duplicate_id_in_train_rejected(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", n_fcd_train=10, n_hc_train=10, n_fcd_test=4)
    splits = generate_stratified_splits(tsv, seed=42, n_splits=5)

    # Duplicate first train item
    splits[0]["train"].append(splits[0]["train"][0])
    is_valid, issues = validate_splits(splits, participants_tsv_path=tsv, expected_n_splits=5)
    assert not is_valid
    assert any("training set contains duplicate IDs" in s for s in issues)


def test_duplicate_id_in_val_rejected(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", n_fcd_train=10, n_hc_train=10, n_fcd_test=4)
    splits = generate_stratified_splits(tsv, seed=42, n_splits=5)

    # Duplicate first val item
    splits[0]["val"].append(splits[0]["val"][0])
    is_valid, issues = validate_splits(splits, participants_tsv_path=tsv, expected_n_splits=5)
    assert not is_valid
    assert any("validation set contains duplicate IDs" in s for s in issues)


def test_fold_train_ids_mismatch_rejected(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", n_fcd_train=10, n_hc_train=10, n_fcd_test=4)
    splits = generate_stratified_splits(tsv, seed=42, n_splits=5)

    # Remove one required training ID from fold 0
    splits[0]["train"].pop()
    is_valid, issues = validate_splits(splits, participants_tsv_path=tsv, expected_n_splits=5)
    assert not is_valid
    assert any("training set missing expected IDs" in s for s in issues)


def test_unexpected_id_in_train_rejected(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", n_fcd_train=10, n_hc_train=10, n_fcd_test=4)
    splits = generate_stratified_splits(tsv, seed=42, n_splits=5)

    # Add completely unknown ID
    splits[0]["train"].append("sub-unknown-ghost-case")
    is_valid, issues = validate_splits(splits, participants_tsv_path=tsv, expected_n_splits=5)
    assert not is_valid
    assert any("unexpected IDs" in s for s in issues)


def test_verify_dataset_splits_file_missing(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv")
    is_valid, issues = verify_dataset_splits_file(tmp_path / "missing_splits.json", tsv)
    assert not is_valid
    assert any("does not exist" in s for s in issues)


def test_check_and_enforce_splits_mismatch(tmp_path: Path):
    tsv = create_mock_participants_tsv(tmp_path / "participants.tsv", n_fcd_train=10, n_hc_train=10, n_fcd_test=4)
    splits1 = generate_stratified_splits(tsv, seed=42, n_splits=5)
    splits2 = generate_stratified_splits(tsv, seed=99, n_splits=5)

    prep_dir = tmp_path / "preprocessed"
    export_splits_to_nnunet(splits1, prep_dir, dataset_names=["Dataset101_FLAIR"])
    export_splits_to_nnunet(splits2, prep_dir, dataset_names=["Dataset102_T1"])

    with pytest.raises(RuntimeError, match="Cross-validation splits mismatch"):
        check_and_enforce_splits_for_training(
            ["Dataset101_FLAIR", "Dataset102_T1"],
            preprocessed_dir=prep_dir,
            participants_tsv_path=tsv,
        )


def test_real_participants_cross_validation():
    if not REAL_PARTICIPANTS_TSV.is_file():
        pytest.skip("Real participants.tsv not found")

    splits = generate_stratified_splits(REAL_PARTICIPANTS_TSV, seed=42, n_splits=5)
    assert len(splits) == 5

    is_valid, issues = validate_splits(splits, participants_tsv_path=REAL_PARTICIPANTS_TSV, expected_n_splits=5)
    assert is_valid, f"Validation issues: {issues}"

    # Verify all 142 training cases validated once
    all_val = []
    for f in splits:
        all_val.extend(f["val"])
    assert len(all_val) == 142
    assert len(set(all_val)) == 142
