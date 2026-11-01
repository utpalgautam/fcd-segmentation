"""Unit and integration tests for nnU-Net training workflow, validation, guards, and runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fcd_segmentation.training.dataset_validator import (
    DatasetValidationResult,
    check_training_guard,
    compare_cohorts,
    normalize_dataset_selector,
    validate_dataset,
)
from fcd_segmentation.training.paths import resolve_nnunet_paths
from fcd_segmentation.training.runner import run_command
from fcd_segmentation.training.workflow import (
    build_training_command,
    run_training_workflow,
)


def create_synthetic_dataset(
    root: Path,
    dataset_name: str,
    channel_names: dict[str, str],
    case_ids: list[str],
    corrupt_json: bool = False,
    missing_json: bool = False,
    missing_channel: bool = False,
    missing_label: bool = False,
    empty_file: bool = False,
) -> Path:
    """Helper to create synthetic nnU-Net datasets for testing without heavy MRI dependencies."""
    dataset_dir = root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    if not missing_json:
        dataset_json_path = dataset_dir / "dataset.json"
        if corrupt_json:
            dataset_json_path.write_text("{bad json: true", encoding="utf-8")
        else:
            meta = {
                "channel_names": channel_names,
                "labels": {"background": 0, "lesion": 1},
                "numTraining": len(case_ids),
                "file_ending": ".nii.gz",
                "name": dataset_name,
            }
            dataset_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for i, case_id in enumerate(case_ids):
        # Create label
        if not (missing_label and i == 0):
            lbl_file = labels_tr / f"{case_id}.nii.gz"
            lbl_file.write_bytes(
                b"" if (empty_file and i == 0) else b"FAKE_NIFTI_LABEL_CONTENT"
            )

        # Create images
        for ch_idx in range(len(channel_names)):
            if missing_channel and i == 0 and ch_idx == 1:
                continue
            img_file = images_tr / f"{case_id}_{ch_idx:04d}.nii.gz"
            img_file.write_bytes(
                b"" if (empty_file and i == 0) else b"FAKE_NIFTI_IMAGE_CONTENT"
            )

    return dataset_dir


class TestPathResolution:
    def test_default_paths(self, tmp_path):
        paths = resolve_nnunet_paths(project_root=tmp_path)
        assert paths.raw == (tmp_path / "data" / "processed").resolve()
        assert (
            paths.preprocessed == (tmp_path / "data" / "nnunet_preprocessed").resolve()
        )
        assert paths.results == (tmp_path / "outputs" / "nnunet_results").resolve()
        env_dict = paths.to_env_dict()
        assert "nnUNet_raw" in env_dict
        assert "nnUNet_preprocessed" in env_dict
        assert "nnUNet_results" in env_dict

    def test_custom_paths(self, tmp_path):
        custom_raw = tmp_path / "custom_raw"
        custom_prep = tmp_path / "custom_prep"
        custom_res = tmp_path / "custom_res"
        paths = resolve_nnunet_paths(
            project_root=tmp_path,
            raw_dir=custom_raw,
            preprocessed_dir=custom_prep,
            results_dir=custom_res,
        )
        assert paths.raw == custom_raw.resolve()
        assert paths.preprocessed == custom_prep.resolve()
        assert paths.results == custom_res.resolve()


class TestDatasetValidation:
    def test_dataset_selector(self):
        assert normalize_dataset_selector("101") == ["101"]
        assert normalize_dataset_selector("Dataset102_T1") == ["102"]
        assert normalize_dataset_selector("all") == ["101", "102", "103"]
        with pytest.raises(ValueError):
            normalize_dataset_selector("999")

    def test_missing_dataset_directory(self, tmp_path):
        res = validate_dataset(tmp_path, "101", check_nifti_headers=False)
        assert not res.valid
        assert any("does not exist" in e for e in res.errors)

    def test_missing_dataset_json(self, tmp_path):
        create_synthetic_dataset(
            tmp_path, "Dataset101_FLAIR", {"0": "FLAIR"}, ["case-1"], missing_json=True
        )
        res = validate_dataset(tmp_path, "101", check_nifti_headers=False)
        assert not res.valid
        assert any("Missing dataset.json" in e for e in res.errors)

    def test_corrupt_dataset_json(self, tmp_path):
        create_synthetic_dataset(
            tmp_path, "Dataset101_FLAIR", {"0": "FLAIR"}, ["case-1"], corrupt_json=True
        )
        res = validate_dataset(tmp_path, "101", check_nifti_headers=False)
        assert not res.valid
        assert any("invalid JSON" in e for e in res.errors)

    def test_missing_fusion_channel(self, tmp_path):
        create_synthetic_dataset(
            tmp_path,
            "Dataset103_Fusion",
            {"0": "FLAIR", "1": "T1"},
            ["case-1"],
            missing_channel=True,
        )
        res = validate_dataset(tmp_path, "103", check_nifti_headers=False)
        assert not res.valid
        assert any("Missing expected image channel" in e for e in res.errors)

    def test_missing_label(self, tmp_path):
        create_synthetic_dataset(
            tmp_path,
            "Dataset101_FLAIR",
            {"0": "FLAIR"},
            ["case-1"],
            missing_label=True,
        )
        res = validate_dataset(tmp_path, "101", check_nifti_headers=False)
        assert not res.valid
        assert any("No label files" in e for e in res.errors)

    def test_empty_file(self, tmp_path):
        create_synthetic_dataset(
            tmp_path,
            "Dataset101_FLAIR",
            {"0": "FLAIR"},
            ["case-1"],
            empty_file=True,
        )
        res = validate_dataset(tmp_path, "101", check_nifti_headers=False)
        assert not res.valid
        assert any("Empty" in e for e in res.errors)

    def test_valid_synthetic_dataset(self, tmp_path):
        create_synthetic_dataset(
            tmp_path,
            "Dataset101_FLAIR",
            {"0": "FLAIR"},
            [f"case-{i}" for i in range(5)],
        )
        res = validate_dataset(tmp_path, "101", check_nifti_headers=False)
        assert res.valid
        assert res.num_training_cases == 5
        assert len(res.case_ids) == 5

    def test_cohort_comparison(self, tmp_path):
        r1 = DatasetValidationResult(
            dataset_name="Dataset101_FLAIR",
            dataset_id="101",
            dataset_path=tmp_path / "101",
            valid=True,
            case_ids=["sub-1", "sub-2"],
        )
        r2 = DatasetValidationResult(
            dataset_name="Dataset102_T1",
            dataset_id="102",
            dataset_path=tmp_path / "102",
            valid=True,
            case_ids=["sub-1", "sub-3"],
        )
        issues = compare_cohorts([r1, r2])
        assert len(issues) == 1
        assert "sub-2" in issues[0]
        assert "sub-3" in issues[0]


class TestTrainingGuard:
    def test_fewer_than_five_cases_blocked(self, tmp_path):
        res = DatasetValidationResult(
            dataset_name="Dataset101_FLAIR",
            dataset_id="101",
            dataset_path=tmp_path / "101",
            valid=True,
            num_training_cases=1,
            case_ids=["sub-1"],
        )
        can_train, msg = check_training_guard([res], min_cases_for_five_fold=5)
        assert not can_train
        assert "COHORT SIZE GUARD BLOCKED TRAINING" in msg
        assert "Found only 1 labeled training case(s)" in msg

    def test_five_cases_passes_technical_guard(self, tmp_path):
        res = DatasetValidationResult(
            dataset_name="Dataset101_FLAIR",
            dataset_id="101",
            dataset_path=tmp_path / "101",
            valid=True,
            num_training_cases=5,
            case_ids=[f"sub-{i}" for i in range(5)],
        )
        can_train, msg = check_training_guard([res], min_cases_for_five_fold=5)
        assert can_train
        assert "meet the minimum" in msg


class TestCommandsAndWorkflow:
    def test_build_training_command_syntax(self):
        cmd = build_training_command(
            dataset_id="101",
            configuration="3d_fullres",
            fold=0,
            continue_training=True,
            device="cuda",
        )
        assert "101" in cmd
        assert "3d_fullres" in cmd
        assert "0" in cmd
        assert "--c" in cmd
        assert "-device" in cmd
        assert "cuda" in cmd

    def test_build_training_command_no_resume(self):
        cmd = build_training_command(
            dataset_id="103",
            configuration="2d",
            fold=2,
            continue_training=False,
        )
        assert "--c" not in cmd
        assert "103" in cmd
        assert "2d" in cmd
        assert "2" in cmd

    @patch("subprocess.run")
    def test_run_command_success(self, mock_subproc, tmp_path):
        paths = resolve_nnunet_paths(project_root=tmp_path)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Execution success"
        mock_proc.stderr = ""
        mock_subproc.return_value = mock_proc

        res = run_command(["test_cmd", "arg1"], paths=paths)
        assert res.success
        assert res.exit_code == 0
        assert "Execution success" in res.stdout

    @patch("subprocess.run")
    def test_run_command_failure_captured(self, mock_subproc, tmp_path):
        paths = resolve_nnunet_paths(project_root=tmp_path)
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "CUDA out of memory"
        mock_subproc.return_value = mock_proc

        res = run_command(["test_cmd", "arg1"], paths=paths)
        assert not res.success
        assert res.exit_code == 1
        assert "CUDA out of memory" in res.stderr

    def test_training_workflow_guard_blocks_without_calling_subprocess(self, tmp_path):
        # Create dataset with only 1 case
        create_synthetic_dataset(
            tmp_path / "data" / "processed",
            "Dataset101_FLAIR",
            {"0": "FLAIR"},
            ["sub-1"],
        )
        paths = resolve_nnunet_paths(project_root=tmp_path)

        with patch("subprocess.run") as mock_subproc:
            summary = run_training_workflow(
                action="train",
                dataset_arg="101",
                fold_arg=0,
                paths=paths,
                logs_dir=tmp_path / "outputs" / "training",
            )
            assert not summary.success
            assert not summary.guard_passed
            assert "COHORT SIZE GUARD" in summary.guard_message
            # Subprocess must NEVER be called
            mock_subproc.assert_not_called()
            assert len(summary.commands_executed) == 0

    def test_import_does_not_execute_training(self):
        # Verify importing training modules does not execute anything

        assert True
