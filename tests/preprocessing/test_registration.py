"""Integration test for registration on real sub-00001 data if available."""

from pathlib import Path

import pytest

from fcd_segmentation.preprocessing.registration import register_to_reference


def test_registration_sub00001(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    t1_path = (
        project_root
        / "data"
        / "raw"
        / "dataset"
        / "sub-00001"
        / "anat"
        / "sub-00001_acq-iso08_T1w.nii.gz"
    )
    flair_path = (
        project_root
        / "data"
        / "raw"
        / "dataset"
        / "sub-00001"
        / "anat"
        / "sub-00001_acq-T2sel_FLAIR.nii.gz"
    )

    if not (t1_path.is_file() and flair_path.is_file()):
        pytest.skip("sub-00001 raw MRI data not available")

    output_path = tmp_path / "sub-00001_T1_registered.nii.gz"
    transform_path = tmp_path / "t1_to_flair.tfm"

    registered = register_to_reference(
        moving_path=t1_path,
        reference_path=flair_path,
        output_path=output_path,
        method="rigid",
        transform_output_path=transform_path,
        number_of_iterations=10,
        number_of_bins=16,
        sampling_percentage=0.2,
    )

    assert output_path.is_file()
    assert transform_path.is_file()
    assert registered["final_metric"] is not None