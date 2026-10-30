from pathlib import Path

import pytest

from fcd_segmentation.data.discovery import discover_flair_files, discover_roi_files


def test_discovery_excludes_roi_and_supports_suffixes(tmp_path: Path):
    anat = tmp_path / "sub-00001" / "anat"
    anat.mkdir(parents=True)
    flair_gz = anat / "sub-00001_acq-a_FLAIR.nii.gz"
    flair_nii = anat / "sub-00001_acq-b_FLAIR.nii"
    roi = anat / "sub-00001_acq-a_FLAIR_roi.nii.gz"
    for path in (flair_gz, flair_nii, roi):
        path.touch()
    assert discover_flair_files(tmp_path) == sorted([flair_gz, flair_nii])
    assert discover_roi_files(anat) == [roi]


def test_missing_root_raises(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        discover_flair_files(tmp_path / "missing")


def test_no_flair_raises(tmp_path: Path):
    (tmp_path / "sub-00001" / "anat").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        discover_flair_files(tmp_path)
