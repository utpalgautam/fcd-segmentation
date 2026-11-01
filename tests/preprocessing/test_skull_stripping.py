"""Read-only integration tests against the real FCD dataset.

These tests inspect real NIfTI files. They do not run HD-BET,
modify source images, or create output inside the raw dataset.

Optional environment variable:
    FCD_DATA_ROOT=/path/to/dataset

Default:
    data/raw/dataset
"""

import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.preprocessing import skull_stripping as ss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "FCD_DATA_ROOT",
        PROJECT_ROOT / "data" / "raw" / "dataset",
    )
).expanduser().resolve()


def _find_flair_files(root: Path) -> list[Path]:
    """Find FLAIR images using the same public discovery function."""
    return ss.discover_flair_files(root)


def _find_roi_candidates(flair_path: Path) -> list[Path]:
    """Find likely ROI files associated with a FLAIR image."""
    name = flair_path.name

    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        return []

    candidates = [
        flair_path.with_name(f"{stem}_roi.nii.gz"),
        flair_path.with_name(f"{stem}_roi.nii"),
    ]

    # Also check the common subject-level ROI naming convention.
    subject_id = flair_path.parent.parent.name
    candidates.extend(
        [
            flair_path.parent / f"{subject_id}_FLAIR_roi.nii.gz",
            flair_path.parent / f"{subject_id}_FLAIR_roi.nii",
        ]
    )

    return list(dict.fromkeys(path for path in candidates if path.is_file()))


@pytest.fixture(scope="module")
def real_flair_files():
    """Discover actual FLAIR files; fail clearly if the dataset is absent."""
    if not DATA_ROOT.is_dir():
        pytest.skip(
            f"Real MRI dataset not available at {DATA_ROOT}. "
            "Skipping real-dataset integration tests."
        )

    files = _find_flair_files(DATA_ROOT)

    if not files:
        pytest.fail(
            f"No FLAIR files were discovered under {DATA_ROOT}. "
            "Check the dataset layout and FLAIR filename convention."
        )

    return files


def test_real_dataset_contains_discoverable_flair_files(real_flair_files):
    """The discovery function must find actual dataset images."""
    assert real_flair_files
    assert all(path.is_file() for path in real_flair_files)

    for path in real_flair_files:
        assert path.name.endswith(("_FLAIR.nii", "_FLAIR.nii.gz"))
        assert "anat" in path.parts


def test_real_flair_paths_are_unique(real_flair_files):
    """Discovery must not return duplicate paths."""
    resolved = [path.resolve() for path in real_flair_files]
    assert len(resolved) == len(set(resolved))


def test_real_flair_images_load_and_have_3d_data(real_flair_files):
    """Every discovered FLAIR image must be readable as a 3D NIfTI."""
    unreadable = []
    invalid_shape = []

    for path in real_flair_files:
        try:
            image = nib.load(str(path))
            if len(image.shape) != 3:
                invalid_shape.append((str(path), image.shape))
                continue

            if any(size <= 0 for size in image.shape):
                invalid_shape.append((str(path), image.shape))

        except Exception as exc:
            unreadable.append((str(path), repr(exc)))

    assert not unreadable, (
        "Some real FLAIR images could not be loaded:\n"
        + "\n".join(f"{path}: {error}" for path, error in unreadable[:20])
    )
    assert not invalid_shape, (
        "Some real FLAIR images are not non-empty 3D volumes:\n"
        + "\n".join(f"{path}: {shape}" for path, shape in invalid_shape[:20])
    )


def test_real_flair_images_have_valid_affines(real_flair_files):
    """Check that actual images have finite, nonsingular spatial geometry."""
    invalid = []

    for path in real_flair_files:
        image = nib.load(str(path))
        affine = np.asarray(image.affine, dtype=np.float64)

        if affine.shape != (4, 4):
            invalid.append((str(path), "affine is not 4x4"))
            continue

        if not np.isfinite(affine).all():
            invalid.append((str(path), "affine contains non-finite values"))
            continue

        if abs(np.linalg.det(affine[:3, :3])) < 1e-10:
            invalid.append((str(path), "affine is singular"))

    assert not invalid, (
        "Invalid image geometry found:\n"
        + "\n".join(f"{path}: {reason}" for path, reason in invalid[:20])
    )


def test_real_flair_voxel_data_are_finite_and_nonempty(real_flair_files):
    """Check voxel data without modifying the source images."""
    invalid = []

    for path in real_flair_files:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)

        if not np.isfinite(data).all():
            invalid.append((str(path), "contains NaN or infinity"))
            continue

        if not np.any(data != 0):
            invalid.append((str(path), "all voxels are zero"))

    assert not invalid, (
        "Unexpected voxel data found:\n"
        + "\n".join(f"{path}: {reason}" for path, reason in invalid[:20])
    )


def test_real_roi_masks_are_readable_and_binary(real_flair_files):
    """Validate existing ROI masks, without assuming every subject has one."""
    roi_files = sorted(
        {
            roi
            for flair_path in real_flair_files
            for roi in _find_roi_candidates(flair_path)
        }
    )

    if not roi_files:
        pytest.skip(
            "No ROI masks matching the supported naming conventions "
            "were found next to the discovered FLAIR images."
        )

    invalid = []

    for path in roi_files:
        try:
            image = nib.load(str(path))
            data = np.asarray(image.dataobj)

            if len(image.shape) != 3:
                invalid.append((str(path), f"not 3D: {image.shape}"))
                continue

            if not np.isfinite(data).all():
                invalid.append((str(path), "contains NaN or infinity"))
                continue

            values = np.unique(data)
            if not np.isin(values, [0, 1]).all():
                invalid.append(
                    (str(path), f"not binary; sample values: {values[:10]}")
                )

        except Exception as exc:
            invalid.append((str(path), repr(exc)))

    assert not invalid, (
        "Invalid ROI masks found:\n"
        + "\n".join(f"{path}: {reason}" for path, reason in invalid[:20])
    )


def test_real_roi_masks_can_be_aligned_to_flair_geometry_when_available(
    real_flair_files,
):
    """Check ROI content and verify that preprocessing aligns it to FLAIR."""
    checked = 0

    for flair_path in real_flair_files:
        roi_files = _find_roi_candidates(flair_path)

        # Only test explicitly associated ROI files.
        for roi_path in roi_files:
            flair = nib.load(str(flair_path))
            roi = nib.load(str(roi_path))
            checked += 1

            assert len(flair.shape) == 3, f"FLAIR is not 3D: {flair_path}"
            assert len(roi.shape) == 3, f"ROI is not 3D: {roi_path}"

            roi_data = np.asarray(roi.dataobj)

            assert np.isfinite(roi_data).all(), (
                f"ROI contains non-finite values: {roi_path}"
            )
            assert set(np.unique(roi_data)).issubset({0, 1}), (
                f"ROI is not binary: {roi_path}"
            )
            assert np.count_nonzero(roi_data) > 0, (
                f"ROI is empty: {roi_path}"
            )

            # Test the actual preprocessing alignment behavior instead of
            # requiring the source files to have identical affine matrices.
            aligned_roi = ss._roi_on_reference_grid(roi_path, flair)

            assert aligned_roi.shape == flair.shape, (
                f"Aligned ROI shape does not match FLAIR: {roi_path}"
            )
            assert aligned_roi.dtype == np.bool_, (
                f"Aligned ROI should be boolean: {roi_path}"
            )
            assert aligned_roi.any(), (
                f"Aligned ROI is empty: {roi_path}"
            )

    if checked == 0:
        pytest.skip(
            "No associated ROI masks found; alignment could not be tested."
        )


def test_real_dataset_discovery_is_deterministic():
    """Repeated discovery should return the same ordered paths."""
    if not DATA_ROOT.is_dir():
        pytest.skip(
            f"Real MRI dataset not available at {DATA_ROOT}. "
            "Skipping real-dataset integration test."
        )

    first = _find_flair_files(DATA_ROOT)
    second = _find_flair_files(DATA_ROOT)

    assert first == second


# ---------------------------------------------------------------------------
# Unit tests: image validation, geometry, paths, and file copying
# ---------------------------------------------------------------------------

def _make_test_image(
    path: Path,
    data: np.ndarray | None = None,
    affine: np.ndarray | None = None,
) -> Path:
    """Create a small synthetic 3D NIfTI image for unit tests."""
    if data is None:
        data = np.ones((8, 9, 10), dtype=np.float32)

    if affine is None:
        affine = np.eye(4, dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    nib.save(image, str(path))
    return path


def test_find_hd_bet_executable_returns_path(monkeypatch):
    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/bin/hd-bet")

    assert ss._find_hd_bet_executable() == "/usr/bin/hd-bet"


def test_find_hd_bet_executable_raises_when_missing(monkeypatch):
    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    # Also block the venv-candidate fallback so RuntimeError is guaranteed
    monkeypatch.setattr(ss.os, "access", lambda path, mode: False)

    with pytest.raises(RuntimeError):
        ss._find_hd_bet_executable()


def test_load_image_accepts_valid_3d_nifti(tmp_path):
    path = _make_test_image(tmp_path / "valid.nii.gz")

    image = ss._load_image(path)

    assert image.shape == (8, 9, 10)


def test_load_image_rejects_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        ss._load_image(tmp_path / "missing.nii.gz")


def test_load_image_rejects_4d_nifti(tmp_path):
    path = _make_test_image(
        tmp_path / "four_dimensional.nii.gz",
        data=np.ones((8, 9, 10, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="3D|three-dimensional|dimension"):
        ss._load_image(path)


def test_image_data_returns_requested_dtype():
    image = nib.Nifti1Image(
        np.arange(24, dtype=np.int16).reshape(2, 3, 4),
        np.eye(4),
    )

    data = ss._image_data(image, dtype=np.float32)

    assert data.shape == (2, 3, 4)
    assert data.dtype == np.float32


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_image_data_rejects_nonfinite_values(bad_value):
    values = np.ones((3, 3, 3), dtype=np.float32)
    values[0, 0, 0] = bad_value
    image = nib.Nifti1Image(values, np.eye(4))

    with pytest.raises(ValueError):
        ss._image_data(image)


def test_same_geometry_accepts_identical_images():
    first = nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4))
    second = nib.Nifti1Image(np.zeros((4, 5, 6)), np.eye(4))

    assert ss._same_geometry(first, second)


def test_same_geometry_rejects_different_shapes():
    first = nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4))
    second = nib.Nifti1Image(np.ones((4, 5, 7)), np.eye(4))

    assert not ss._same_geometry(first, second)


def test_same_geometry_rejects_affine_difference():
    first = nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4))

    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 0.1
    second = nib.Nifti1Image(np.ones((4, 5, 6)), shifted_affine)

    assert not ss._same_geometry(first, second)


def test_validate_binary_mask_accepts_nonempty_binary_mask():
    values = np.zeros((4, 5, 6), dtype=np.uint8)
    values[1:3, 1:4, 2:5] = 1
    image = nib.Nifti1Image(values, np.eye(4))

    mask = ss._validate_binary_mask(image)

    assert mask.dtype == np.bool_
    assert mask.shape == values.shape
    assert mask.sum() == np.count_nonzero(values)


def test_validate_binary_mask_rejects_empty_mask():
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.uint8), np.eye(4))

    with pytest.raises(ValueError):
        ss._validate_binary_mask(image)


@pytest.mark.parametrize("invalid_value", [2, -1, 0.5])
def test_validate_binary_mask_rejects_nonbinary_values(invalid_value):
    values = np.zeros((4, 5, 6), dtype=np.float32)
    values[1, 1, 1] = 1
    values[2, 2, 2] = invalid_value
    image = nib.Nifti1Image(values, np.eye(4))

    with pytest.raises(ValueError):
        ss._validate_binary_mask(image)


def test_mask_path_from_output_accepts_nii_gz(tmp_path):
    output_path = tmp_path / "subject.nii.gz"

    assert ss._mask_path_from_output(output_path) == (
        tmp_path / "subject_bet.nii.gz"
    )


def test_mask_path_from_output_rejects_unsupported_extension(tmp_path):
    with pytest.raises(ValueError):
        ss._mask_path_from_output(tmp_path / "subject.nii")


def test_copy_nifti_copies_file(tmp_path):
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "nested" / "copy.nii.gz"
    source.write_bytes(b"synthetic NIfTI file")

    result = ss._copy_nifti(source, destination, overwrite=False)

    assert result == destination
    assert destination.read_bytes() == source.read_bytes()


def test_copy_nifti_protects_existing_destination(tmp_path):
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "destination.nii.gz"
    source.write_bytes(b"new content")
    destination.write_bytes(b"original content")

    with pytest.raises(FileExistsError):
        ss._copy_nifti(source, destination, overwrite=False)

    assert destination.read_bytes() == b"original content"


def test_copy_nifti_overwrites_when_requested(tmp_path):
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "destination.nii.gz"
    source.write_bytes(b"new content")
    destination.write_bytes(b"original content")

    ss._copy_nifti(source, destination, overwrite=True)

    assert destination.read_bytes() == b"new content"

# ---------------------------------------------------------------------------
# Unit tests: mocked HD-BET execution
# ---------------------------------------------------------------------------

def _make_hdbet_input(tmp_path):
    """Create a valid synthetic FLAIR image and its expected subject ID."""
    subject_dir = tmp_path / "sub-test" / "anat"
    subject_dir.mkdir(parents=True, exist_ok=True)

    input_path = subject_dir / "sub-test_FLAIR.nii.gz"

    data = np.zeros((12, 12, 12), dtype=np.float32)
    data[2:10, 2:10, 2:10] = 5.0

    _make_test_image(input_path, data=data)
    return input_path


def _mock_hdbet_success(
    command,
    *,
    check,
    text,
    output_data=None,
    mask_data=None,
    output_affine=None,
    mask_affine=None,
    create_output=True,
    create_mask=True,
    returncode=0,
):
    """Mock HD-BET by writing synthetic output files."""
    output_path = Path(command[command.index("-o") + 1])
    mask_path = ss._mask_path_from_output(output_path)

    if output_data is None:
        output_data = np.zeros((12, 12, 12), dtype=np.float32)
        output_data[3:9, 3:9, 3:9] = 5.0

    if mask_data is None:
        mask_data = np.zeros((12, 12, 12), dtype=np.uint8)
        mask_data[3:9, 3:9, 3:9] = 1

    if output_affine is None:
        output_affine = np.eye(4)

    if mask_affine is None:
        mask_affine = output_affine

    if create_output:
        nib.save(
            nib.Nifti1Image(output_data, output_affine),
            str(output_path),
        )

    if create_mask:
        nib.save(
            nib.Nifti1Image(mask_data, mask_affine),
            str(mask_path),
        )

    return SimpleNamespace(returncode=returncode)


def test_run_hd_bet_success(monkeypatch, tmp_path):
    input_path = _make_hdbet_input(tmp_path)
    captured = {}

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _mock_hdbet_success(command, **kwargs)

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    output_dir = tmp_path / "outputs"

    result = ss.run_hd_bet(
        input_path,
        output_dir,
        device="cpu",
        tta=True,
    )

    assert result["status"] == "success"
    assert result["subject_id"] == "sub-test"
    assert result["device"] == "cpu"

    stripped_path = Path(result["skullstripped_path"])
    mask_path = Path(result["brain_mask_path"])

    assert stripped_path.is_file()
    assert mask_path.is_file()

    assert captured["kwargs"] == {
        "check": False,
        "text": True,
    }

    command = captured["command"]
    assert command[0] == "/fake/hd-bet"
    assert "-i" in command
    assert "-o" in command
    assert "-device" in command
    assert "cpu" in command
    assert "--save_bet_mask" in command
    assert "--verbose" in command
    assert "--disable_tta" not in command


def test_run_hd_bet_disables_tta_when_requested(monkeypatch, tmp_path):
    input_path = _make_hdbet_input(tmp_path)
    captured = {}

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        captured["command"] = command
        return _mock_hdbet_success(command, **kwargs)

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    ss.run_hd_bet(
        input_path,
        tmp_path / "outputs",
        device="cpu",
        tta=False,
    )

    assert "--disable_tta" in captured["command"]


def test_run_hd_bet_rejects_unsupported_device(monkeypatch, tmp_path):
    input_path = _make_hdbet_input(tmp_path)

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    with pytest.raises(ValueError, match="Unsupported device"):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="mps",
        )


def test_run_hd_bet_protects_existing_output(monkeypatch, tmp_path):
    input_path = _make_hdbet_input(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    existing = output_dir / "sub-test_FLAIR_skullstripped.nii.gz"
    existing.write_bytes(b"preserve this")

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    with pytest.raises(FileExistsError):
        ss.run_hd_bet(
            input_path,
            output_dir,
            device="cpu",
            overwrite=False,
        )

    assert existing.read_bytes() == b"preserve this"


def test_run_hd_bet_raises_on_subprocess_failure(monkeypatch, tmp_path):
    input_path = _make_hdbet_input(tmp_path)

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    monkeypatch.setattr(
        ss.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError, match="HD-BET failed"):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="cpu",
        )


@pytest.mark.parametrize(
    ("create_output", "create_mask", "message"),
    [
        (False, True, "expected output image"),
        (True, False, "expected brain mask"),
    ],
)
def test_run_hd_bet_rejects_missing_outputs(
    monkeypatch,
    tmp_path,
    create_output,
    create_mask,
    message,
):
    input_path = _make_hdbet_input(tmp_path)

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        return _mock_hdbet_success(
            command,
            create_output=create_output,
            create_mask=create_mask,
            **kwargs,
        )

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=message):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="cpu",
        )


def test_run_hd_bet_rejects_empty_brain_mask(monkeypatch, tmp_path):
    input_path = _make_hdbet_input(tmp_path)

    empty_mask = np.zeros((12, 12, 12), dtype=np.uint8)

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        return _mock_hdbet_success(
            command,
            mask_data=empty_mask,
            **kwargs,
        )

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="empty brain mask"):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="cpu",
        )


def test_run_hd_bet_rejects_nonbinary_brain_mask(monkeypatch, tmp_path):
    input_path = _make_hdbet_input(tmp_path)

    invalid_mask = np.zeros((12, 12, 12), dtype=np.float32)
    invalid_mask[3:9, 3:9, 3:9] = 2.0

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        return _mock_hdbet_success(
            command,
            mask_data=invalid_mask,
            **kwargs,
        )

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="other than 0 and 1"):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="cpu",
        )


def test_run_hd_bet_rejects_output_geometry_mismatch(
    monkeypatch,
    tmp_path,
):
    input_path = _make_hdbet_input(tmp_path)

    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 2.0

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        return _mock_hdbet_success(
            command,
            output_affine=shifted_affine,
            **kwargs,
        )

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    with pytest.raises(
        ValueError,
        match="output geometry does not match the input",
    ):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="cpu",
        )


def test_run_hd_bet_rejects_mask_geometry_mismatch(
    monkeypatch,
    tmp_path,
):
    input_path = _make_hdbet_input(tmp_path)

    shifted_affine = np.eye(4)
    shifted_affine[1, 3] = 2.0

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        return _mock_hdbet_success(
            command,
            mask_affine=shifted_affine,
            **kwargs,
        )

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    with pytest.raises(
        ValueError,
        match="brain-mask geometry does not match",
    ):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="cpu",
        )


def test_run_hd_bet_rejects_nonzero_voxels_outside_mask(
    monkeypatch,
    tmp_path,
):
    input_path = _make_hdbet_input(tmp_path)

    output_data = np.zeros((12, 12, 12), dtype=np.float32)
    output_data[3:9, 3:9, 3:9] = 5.0
    output_data[0, 0, 0] = 2.0

    monkeypatch.setattr(
        ss,
        "_find_hd_bet_executable",
        lambda: "/fake/hd-bet",
    )

    def fake_run(command, **kwargs):
        return _mock_hdbet_success(
            command,
            output_data=output_data,
            **kwargs,
        )

    monkeypatch.setattr(ss.subprocess, "run", fake_run)

    with pytest.raises(
        ValueError,
        match="nonzero voxels outside",
    ):
        ss.run_hd_bet(
            input_path,
            tmp_path / "outputs",
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Unit tests: visual QC report generation
# ---------------------------------------------------------------------------

def _make_qc_test_images(tmp_path, include_roi=True):
    """Create compatible synthetic images for visual QC tests."""
    shape = (20, 20, 20)

    original = np.zeros(shape, dtype=np.float32)
    original[5:15, 5:15, 5:15] = 10.0

    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:15, 5:15, 5:15] = 1

    stripped = original.copy()
    stripped[mask == 0] = 0.0

    original_path = _make_test_image(
        tmp_path / "original_flair.nii.gz",
        data=original,
    )
    stripped_path = _make_test_image(
        tmp_path / "stripped_flair.nii.gz",
        data=stripped,
    )
    mask_path = _make_test_image(
        tmp_path / "brain_mask.nii.gz",
        data=mask,
    )

    roi_path = None
    if include_roi:
        roi = np.zeros(shape, dtype=np.uint8)
        roi[8:10, 8:10, 8:10] = 1

        roi_path = _make_test_image(
            tmp_path / "lesion_roi.nii.gz",
            data=roi,
        )

    return original_path, stripped_path, mask_path, roi_path


def test_create_visual_qc_generates_png_and_correct_metrics(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)
    output_png = tmp_path / "qc" / "subject-001.png"

    result = ss.create_visual_qc(
        original,
        stripped,
        mask,
        output_png,
        roi_path=roi,
        subject_id="subject-001",
    )

    assert output_png.is_file()
    assert output_png.stat().st_size > 0

    assert result["subject_id"] == "subject-001"
    assert result["qc_image"] == str(output_png)

    # The mask covers 10 * 10 * 10 voxels out of 20 * 20 * 20.
    assert result["brain_voxels"] == 1000
    assert result["brain_fraction"] == pytest.approx(0.125)

    # All stripped-image voxels outside the brain mask are zero.
    assert result["stripped_nonzero_outside_mask"] == 0

    # The ROI covers 2 * 2 * 2 voxels, all inside the brain mask.
    assert result["roi_voxels"] == 8
    assert result["roi_inside_brain_voxels"] == 8
    assert result["roi_inside_brain_fraction"] == pytest.approx(1.0)

    assert result["review_flags"] == ""
    assert result["qc_status"] == "PASS"


def test_create_visual_qc_flags_nonzero_voxels_outside_mask(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)

    # Introduce one nonzero voxel outside the brain mask.
    stripped_image = nib.load(str(stripped))
    stripped_data = np.asarray(stripped_image.dataobj).copy()
    stripped_data[0, 0, 0] = 5.0

    _make_test_image(
        stripped,
        data=stripped_data,
        affine=stripped_image.affine,
    )

    result = ss.create_visual_qc(
        original,
        stripped,
        mask,
        tmp_path / "qc_outside_mask.png",
        roi_path=roi,
    )

    assert result["stripped_nonzero_outside_mask"] == 1
    assert "REVIEW_NONZERO_OUTSIDE_MASK" in result["review_flags"]
    assert result["qc_status"] == "REVIEW"


def test_create_visual_qc_flags_low_brain_fraction(tmp_path):
    shape = (20, 20, 20)
    original_data = np.ones(shape, dtype=np.float32)

    # Only 4 voxels are in the mask: 4 / 8000 = 0.0005.
    mask_data = np.zeros(shape, dtype=np.uint8)
    mask_data[10:12, 10:12, 10:11] = 1

    stripped_data = original_data.copy()
    stripped_data[mask_data == 0] = 0.0

    original = _make_test_image(
        tmp_path / "low_original.nii.gz",
        data=original_data,
    )
    stripped = _make_test_image(
        tmp_path / "low_stripped.nii.gz",
        data=stripped_data,
    )
    mask = _make_test_image(
        tmp_path / "low_mask.nii.gz",
        data=mask_data,
    )

    result = ss.create_visual_qc(
        original,
        stripped,
        mask,
        tmp_path / "low_brain_fraction.png",
    )

    assert result["brain_fraction"] == pytest.approx(4 / 8000)
    assert "REVIEW_BRAIN_FRACTION" in result["review_flags"]
    assert result["qc_status"] == "REVIEW"


def test_create_visual_qc_rejects_original_geometry_mismatch(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)

    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 2.0

    mismatched_stripped = _make_test_image(
        tmp_path / "mismatched_stripped.nii.gz",
        data=np.zeros((20, 20, 20), dtype=np.float32),
        affine=shifted_affine,
    )

    with pytest.raises(
        ValueError,
        match="Original and skull-stripped FLAIR geometry do not match",
    ):
        ss.create_visual_qc(
            original,
            mismatched_stripped,
            mask,
            tmp_path / "should_not_exist.png",
            roi_path=roi,
        )


def test_create_visual_qc_rejects_mask_geometry_mismatch(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)

    shifted_affine = np.eye(4)
    shifted_affine[1, 3] = 2.0

    mismatched_mask = _make_test_image(
        tmp_path / "mismatched_mask.nii.gz",
        data=np.ones((20, 20, 20), dtype=np.uint8),
        affine=shifted_affine,
    )

    with pytest.raises(
        ValueError,
        match="Skull-stripped image and brain-mask geometry do not match",
    ):
        ss.create_visual_qc(
            original,
            stripped,
            mismatched_mask,
            tmp_path / "should_not_exist.png",
            roi_path=roi,
        )


def test_create_visual_qc_rejects_empty_brain_mask(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)

    empty_mask = _make_test_image(
        tmp_path / "empty_mask.nii.gz",
        data=np.zeros((20, 20, 20), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="empty brain mask"):
        ss.create_visual_qc(
            original,
            stripped,
            empty_mask,
            tmp_path / "empty_mask_qc.png",
            roi_path=roi,
        )


def test_create_visual_qc_rejects_nonbinary_brain_mask(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)

    invalid_mask_data = np.zeros((20, 20, 20), dtype=np.uint8)
    invalid_mask_data[5:15, 5:15, 5:15] = 2

    invalid_mask = _make_test_image(
        tmp_path / "invalid_mask.nii.gz",
        data=invalid_mask_data,
    )

    with pytest.raises(ValueError, match="other than 0 and 1"):
        ss.create_visual_qc(
            original,
            stripped,
            invalid_mask,
            tmp_path / "invalid_mask_qc.png",
            roi_path=roi,
        )


def test_create_visual_qc_works_without_roi(tmp_path):
    original, stripped, mask, _ = _make_qc_test_images(
        tmp_path,
        include_roi=False,
    )

    result = ss.create_visual_qc(
        original,
        stripped,
        mask,
        tmp_path / "without_roi.png",
    )

    assert result["roi_voxels"] == 0
    assert result["roi_inside_brain_voxels"] == 0
    assert result["roi_inside_brain_fraction"] is None
    assert result["qc_status"] == "PASS"


def test_create_visual_qc_flags_roi_outside_brain(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)

    roi_data = np.zeros((20, 20, 20), dtype=np.uint8)
    roi_data[0:2, 0:2, 0:2] = 1

    outside_roi = _make_test_image(
        tmp_path / "outside_roi.nii.gz",
        data=roi_data,
    )

    result = ss.create_visual_qc(
        original,
        stripped,
        mask,
        tmp_path / "outside_roi_qc.png",
        roi_path=outside_roi,
    )

    assert result["roi_voxels"] == 8
    assert result["roi_inside_brain_voxels"] == 0
    assert result["roi_inside_brain_fraction"] == pytest.approx(0.0)
    assert "REVIEW_ROI_MASK_OVERLAP" in result["review_flags"]
    assert result["qc_status"] == "REVIEW"


def test_create_visual_qc_flags_empty_roi(tmp_path):
    original, stripped, mask, roi = _make_qc_test_images(tmp_path)

    empty_roi = _make_test_image(
        tmp_path / "empty_roi.nii.gz",
        data=np.zeros((20, 20, 20), dtype=np.uint8),
    )

    result = ss.create_visual_qc(
        original,
        stripped,
        mask,
        tmp_path / "empty_roi_qc.png",
        roi_path=empty_roi,
    )

    assert result["roi_voxels"] == 0
    assert "REVIEW_EMPTY_ROI" in result["review_flags"]
    assert result["qc_status"] == "REVIEW"


# ---------------------------------------------------------------------------
# Unit tests: QC HTML and CSV report generation
# ---------------------------------------------------------------------------

def test_generate_qc_index_creates_html_and_csv(tmp_path):
    output_dir = tmp_path / "qc_reports"
    qc_image = output_dir / "sub-001" / "sub-001_qc.png"

    records = [
        {
            "subject_id": "sub-001",
            "status": "success",
            "qc_status": "PASS",
            "qc_image": str(qc_image),
            "brain_voxels": 1000,
            "brain_fraction": 0.125,
            "stripped_nonzero_outside_mask": 0,
            "roi_voxels": 8,
            "roi_inside_brain_voxels": 8,
            "roi_inside_brain_fraction": 1.0,
            "review_flags": "",
            "error": "",
        }
    ]

    index_path, csv_path = ss.generate_qc_index(records, output_dir)

    assert index_path.is_file()
    assert csv_path.is_file()
    # The index generator links QC images; it does not create the images.
    assert not (output_dir / "images").exists()

    html_content = index_path.read_text(encoding="utf-8")
    csv_content = csv_path.read_text(encoding="utf-8")

    assert "sub-001" in html_content
    assert "PASS" in html_content
    assert "sub-001/sub-001_qc.png" in html_content
    assert "sub-001" in csv_content
    assert "brain_fraction" in csv_content
    assert "0.125" in csv_content


def test_generate_qc_index_includes_failed_records(tmp_path):
    output_dir = tmp_path / "qc_reports"

    records = [
        {
            "subject_id": "sub-failed",
            "status": "failed",
            "error": "Input MRI file is missing.",
        }
    ]

    index_path, csv_path = ss.generate_qc_index(records, output_dir)

    html_content = index_path.read_text(encoding="utf-8")
    csv_content = csv_path.read_text(encoding="utf-8")

    assert "sub-failed" in html_content
    assert "FAILED" in html_content
    assert "Input MRI file is missing." in html_content
    assert "sub-failed" in csv_content
    assert "Input MRI file is missing." in csv_content


def test_generate_qc_index_escapes_html_content(tmp_path):
    output_dir = tmp_path / "qc_reports"

    records = [
        {
            "subject_id": "<script>alert(1)</script>",
            "status": "failed",
            "error": "<img src=x onerror=alert(1)>",
        }
    ]

    index_path, _ = ss.generate_qc_index(records, output_dir)
    html_content = index_path.read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in html_content
    assert "&lt;script&gt;" in html_content
    assert "<img src=x onerror=alert(1)>" not in html_content
    assert "&lt;img" in html_content