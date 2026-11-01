"""Unit tests for nnU-Net dataset generation and validation."""

import json

import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.preprocessing.dataset import (
    create_nnunet_dataset_json,
    validate_nnunet_subject_outputs,
    write_subject_to_nnunet,
)


def _make_dummy_image(shape=(12, 12, 12), affine=None, dtype=np.float32, val=1.0):
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    data = np.full(shape, val, dtype=dtype)
    return nib.Nifti1Image(data, affine)


def test_create_nnunet_dataset_json(tmp_path):
    p = create_nnunet_dataset_json(
        output_dir=tmp_path,
        channel_names={"0": "FLAIR", "1": "T1"},
        labels={"background": 0, "lesion": 1},
        num_training=10,
        dataset_name="Dataset103_Fusion",
    )
    assert p.is_file()
    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["channel_names"] == {"0": "FLAIR", "1": "T1"}
    assert meta["labels"] == {"background": 0, "lesion": 1}
    assert meta["numTraining"] == 10
    assert meta["file_ending"] == ".nii.gz"


def test_write_and_validate_subject_nnunet(tmp_path):
    flair_ds = tmp_path / "Dataset101_FLAIR"
    t1_ds = tmp_path / "Dataset102_T1"
    fusion_ds = tmp_path / "Dataset103_Fusion"

    flair_img = _make_dummy_image(val=2.5)
    t1_img = _make_dummy_image(val=1.8)
    roi_data = np.zeros((12, 12, 12), dtype=np.uint8)
    roi_data[3:6, 3:6, 3:6] = 1
    roi_img = nib.Nifti1Image(roi_data, np.eye(4))

    res = write_subject_to_nnunet(
        subject_id="sub-001",
        flair_image=flair_img,
        t1_image=t1_img,
        roi_image=roi_img,
        split="train",
        flair_dataset_dir=flair_ds,
        t1_dataset_dir=t1_ds,
        fusion_dataset_dir=fusion_ds,
        overwrite=True,
    )
    assert res["fusion_geometry_match"] is True

    # Validate written outputs
    val_res = validate_nnunet_subject_outputs(
        subject_id="sub-001",
        flair_dataset_dir=flair_ds,
        t1_dataset_dir=t1_ds,
        fusion_dataset_dir=fusion_ds,
        split="train",
    )
    assert val_res["status"] == "PASS"
    assert val_res["flair_dataset_valid"] is True
    assert val_res["t1_dataset_valid"] is True
    assert val_res["fusion_dataset_valid"] is True
    assert val_res["fusion_geometry_match"] is True
    assert val_res["lesion_voxels"] == 27
    assert val_res["unique_labels"] == [0, 1]


def test_write_subject_fails_on_grid_mismatch(tmp_path):
    flair_img = _make_dummy_image(shape=(12, 12, 12))
    # T1 with different shape
    t1_img = _make_dummy_image(shape=(14, 14, 14))
    roi_img = _make_dummy_image(shape=(12, 12, 12), dtype=np.uint8, val=0)

    with pytest.raises(ValueError, match="Fusion channels FLAIR and T1 do not share the exact same grid"):
        write_subject_to_nnunet(
            subject_id="sub-bad",
            flair_image=flair_img,
            t1_image=t1_img,
            roi_image=roi_img,
            split="train",
            flair_dataset_dir=tmp_path / "flair",
            t1_dataset_dir=tmp_path / "t1",
            fusion_dataset_dir=tmp_path / "fusion",
        )
