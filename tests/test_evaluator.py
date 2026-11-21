import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.evaluation.evaluator import (
    DatasetEvaluator,
    compare_modalities,
    compute_summary_stats,
)


def create_nifti_file(path: Path, data: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0), affine=None):
    if affine is None:
        affine = np.diag((*spacing, 1.0))
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


def create_mock_participants_tsv(path: Path) -> Path:
    content = (
        "participant_id\tgroup\tsplit\n"
        "sub-01\tfcd\ttest\n"
        "sub-02\tfcd\ttest\n"
        "sub-03\tfcd\ttrain\n"
        "sub-04\tfcd\ttrain\n"
        "sub-05\thc\ttrain\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def create_mock_splits_json(path: Path) -> Path:
    splits = [
        {
            "train": ["sub-04", "sub-05"],
            "val": ["sub-03"],
        },
        {
            "train": ["sub-03", "sub-05"],
            "val": ["sub-04"],
        },
    ]
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return path


def test_compute_summary_stats_with_infinite():
    values = [1.5, 2.5, float("inf"), float("inf")]
    stats = compute_summary_stats(values)
    assert abs(stats["mean"] - 2.0) < 1e-4
    assert abs(stats["finite_only_mean"] - 2.0) < 1e-4
    assert stats["finite_count"] == 2.0
    assert stats["infinite_count"] == 2.0
    assert stats["infinite_percentage"] == 50.0
    assert stats["total_count"] == 4.0


def test_evaluator_predictions_dir_test_split_success(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gts"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    pred_dir.mkdir()
    gt_dir.mkdir()

    # Create expected test subjects sub-01 and sub-02
    data1 = np.zeros((10, 10, 10), dtype=np.uint8)
    data1[2:5, 2:5, 2:5] = 1
    create_nifti_file(pred_dir / "sub-01.nii.gz", data1)
    create_nifti_file(gt_dir / "sub-01.nii.gz", data1)

    data2 = np.zeros((10, 10, 10), dtype=np.uint8)
    data2[3:6, 2:5, 2:5] = 1
    create_nifti_file(pred_dir / "sub-02.nii.gz", data2)
    create_nifti_file(gt_dir / "sub-02.nii.gz", data2)

    report = DatasetEvaluator.evaluate_predictions_dir(
        name="Dataset101_FLAIR",
        predictions_dir=pred_dir,
        ground_truth_dir=gt_dir,
        participants_tsv=tsv_path,
        split="test",
    )

    assert report.num_cases == 2
    assert report.expected_num_cases == 2
    assert abs(report.subject_records[0].metrics["dice"] - 1.0) < 1e-5
    assert len(report.scientific_notes) > 0


def test_evaluator_validation_fold_success(tmp_path: Path):
    pred_dir = tmp_path / "preds_fold0"
    labels_tr = tmp_path / "labelsTr"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    splits_path = create_mock_splits_json(tmp_path / "splits_final.json")
    pred_dir.mkdir()
    labels_tr.mkdir()

    # In mock splits, fold 0 validation subject is ONLY sub-03
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 1
    create_nifti_file(pred_dir / "sub-03.nii.gz", data)
    create_nifti_file(labels_tr / "sub-03.nii.gz", data)

    report = DatasetEvaluator.evaluate_predictions_dir(
        name="Dataset101_FLAIR",
        predictions_dir=pred_dir,
        ground_truth_dir=labels_tr,
        participants_tsv=tsv_path,
        split="val",
        fold=0,
        splits_file=splits_path,
    )

    assert report.num_cases == 1
    assert report.expected_num_cases == 1
    assert report.fold == 0
    assert report.split == "val"
    assert report.subject_records[0].subject_id == "sub-03"


def test_evaluator_validation_missing_fold_rejected(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    labels_tr = tmp_path / "labelsTr"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    splits_path = create_mock_splits_json(tmp_path / "splits_final.json")
    pred_dir.mkdir()
    labels_tr.mkdir()

    with pytest.raises(ValueError, match="Fold index .* is required when evaluating validation split"):
        DatasetEvaluator.evaluate_predictions_dir(
            name="Dataset101_FLAIR",
            predictions_dir=pred_dir,
            ground_truth_dir=labels_tr,
            participants_tsv=tsv_path,
            split="val",
            fold=None,  # Missing fold index
            splits_file=splits_path,
        )


def test_evaluator_validation_extra_subjects_rejected(tmp_path: Path):
    pred_dir = tmp_path / "preds_fold0"
    labels_tr = tmp_path / "labelsTr"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    splits_path = create_mock_splits_json(tmp_path / "splits_final.json")
    pred_dir.mkdir()
    labels_tr.mkdir()

    # Fold 0 validation subject is sub-03. If sub-04 is also predicted, reject!
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    create_nifti_file(pred_dir / "sub-03.nii.gz", data)
    create_nifti_file(pred_dir / "sub-04.nii.gz", data)
    create_nifti_file(labels_tr / "sub-03.nii.gz", data)
    create_nifti_file(labels_tr / "sub-04.nii.gz", data)

    with pytest.raises(ValueError, match="outside the expected fold 0 validation cohort"):
        DatasetEvaluator.evaluate_predictions_dir(
            name="Dataset101_FLAIR",
            predictions_dir=pred_dir,
            ground_truth_dir=labels_tr,
            participants_tsv=tsv_path,
            split="val",
            fold=0,
            splits_file=splits_path,
        )


def test_evaluator_missing_prediction_rejected(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gts"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    pred_dir.mkdir()
    gt_dir.mkdir()

    # Only create sub-01, leaving sub-02 missing
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    create_nifti_file(pred_dir / "sub-01.nii.gz", data)
    create_nifti_file(gt_dir / "sub-01.nii.gz", data)
    create_nifti_file(gt_dir / "sub-02.nii.gz", data)

    with pytest.raises(ValueError, match="Incomplete prediction coverage"):
        DatasetEvaluator.evaluate_predictions_dir(
            name="Dataset101_FLAIR",
            predictions_dir=pred_dir,
            ground_truth_dir=gt_dir,
            participants_tsv=tsv_path,
            split="test",
        )


def test_evaluator_extra_prediction_rejected(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gts"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    pred_dir.mkdir()
    gt_dir.mkdir()

    # Create sub-01, sub-02, and sub-03 (sub-03 belongs to train split!)
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    for sid in ["sub-01", "sub-02", "sub-03"]:
        create_nifti_file(pred_dir / f"{sid}.nii.gz", data)
        create_nifti_file(gt_dir / f"{sid}.nii.gz", data)

    with pytest.raises(ValueError, match="outside the expected test split cohort"):
        DatasetEvaluator.evaluate_predictions_dir(
            name="Dataset101_FLAIR",
            predictions_dir=pred_dir,
            ground_truth_dir=gt_dir,
            participants_tsv=tsv_path,
            split="test",
        )


def test_evaluator_missing_ground_truth_rejected(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gts"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    pred_dir.mkdir()
    gt_dir.mkdir()

    data = np.zeros((10, 10, 10), dtype=np.uint8)
    create_nifti_file(pred_dir / "sub-01.nii.gz", data)
    create_nifti_file(pred_dir / "sub-02.nii.gz", data)
    create_nifti_file(gt_dir / "sub-01.nii.gz", data)

    with pytest.raises(FileNotFoundError, match="Ground-truth label file missing"):
        DatasetEvaluator.evaluate_predictions_dir(
            name="Dataset101_FLAIR",
            predictions_dir=pred_dir,
            ground_truth_dir=gt_dir,
            participants_tsv=tsv_path,
            split="test",
        )


def test_evaluator_affine_mismatch_rejected(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gts"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    pred_dir.mkdir()
    gt_dir.mkdir()

    data = np.zeros((10, 10, 10), dtype=np.uint8)
    create_nifti_file(pred_dir / "sub-01.nii.gz", data, affine=np.diag([1.0, 1.0, 1.0, 1.0]))
    create_nifti_file(gt_dir / "sub-01.nii.gz", data, affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    create_nifti_file(pred_dir / "sub-02.nii.gz", data)
    create_nifti_file(gt_dir / "sub-02.nii.gz", data)

    with pytest.raises(ValueError, match="Affine matrix mismatch"):
        DatasetEvaluator.evaluate_predictions_dir(
            name="Dataset101_FLAIR",
            predictions_dir=pred_dir,
            ground_truth_dir=gt_dir,
            participants_tsv=tsv_path,
            split="test",
        )


def test_evaluator_invalid_label_values_rejected(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gts"
    tsv_path = create_mock_participants_tsv(tmp_path / "participants.tsv")
    pred_dir.mkdir()
    gt_dir.mkdir()

    data_bad = np.zeros((10, 10, 10), dtype=np.int32)
    data_bad[2, 2, 2] = 5
    data_good = np.zeros((10, 10, 10), dtype=np.uint8)

    create_nifti_file(pred_dir / "sub-01.nii.gz", data_bad)
    create_nifti_file(gt_dir / "sub-01.nii.gz", data_good)
    create_nifti_file(pred_dir / "sub-02.nii.gz", data_good)
    create_nifti_file(gt_dir / "sub-02.nii.gz", data_good)

    with pytest.raises(ValueError, match="Invalid discrete prediction values"):
        DatasetEvaluator.evaluate_predictions_dir(
            name="Dataset101_FLAIR",
            predictions_dir=pred_dir,
            ground_truth_dir=gt_dir,
            participants_tsv=tsv_path,
            split="test",
        )


def test_compare_modalities(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gts"
    pred_dir.mkdir()
    gt_dir.mkdir()

    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[2:4, 2:4, 2:4] = 1
    create_nifti_file(pred_dir / "sub-01.nii.gz", data)
    create_nifti_file(gt_dir / "sub-01.nii.gz", data)

    rep1 = DatasetEvaluator.evaluate_predictions_dir("FLAIR", pred_dir, gt_dir, split="test")
    rep2 = DatasetEvaluator.evaluate_predictions_dir("T1", pred_dir, gt_dir, split="test")

    comp = compare_modalities([rep1, rep2])
    assert "modalities" in comp
    assert comp["modalities"] == ["FLAIR", "T1"]
    assert "dice" in comp["metrics_summary"]
    assert "finite_only_mean" in comp["metrics_summary"]["dice"]["FLAIR"]
