"""
Comprehensive test suite for T1-to-FLAIR registration and lesion ROI alignment.

Verifies:
1. FLAIR and ROI with identical geometry.
2. ROI with different spacing but same physical location.
3. ROI with different origin.
4. ROI with different orientation.
5. Correct nearest-neighbor resampling.
6. Binary label preservation.
7. ROI voxel count preservation within reasonable tolerance.
8. Correct registered T1 reference geometry.
9. Correct transform direction.
10. No accidental application of T1 registration transform to the ROI.
11. HC zero-mask generation.
12. Final nnU-Net label geometry synchronization across datasets.
13. QC generation with ROI over registered T1 and off-center lesion.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from fcd_segmentation.preprocessing.dataset import write_subject_to_nnunet
from fcd_segmentation.preprocessing.geometry import (
    compute_roi_physical_centroid,
    same_grid,
    validate_subject_geometry_and_roi,
)
from fcd_segmentation.preprocessing.registration import (
    create_registration_qc_visuals,
    register_to_reference,
)
from fcd_segmentation.preprocessing.resampling import resample_to_reference


def _create_sphere(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    dtype: type = np.float32,
    max_val: float = 100.0,
) -> np.ndarray:
    """Create a 3D sphere volume."""
    vol = np.zeros(shape, dtype=dtype)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    mask = dist <= radius
    if np.issubdtype(dtype, np.integer):
        vol[mask] = 1
    else:
        vol[mask] = max_val - dist[mask] * (max_val / (radius + 1e-5))
    return vol


# ---------------------------------------------------------------------------
# Test 1: FLAIR and ROI with identical geometry
# ---------------------------------------------------------------------------
def test_flair_and_roi_identical_geometry(tmp_path: Path):
    shape = (32, 32, 32)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-16.0, -16.0, -16.0]

    flair_data = _create_sphere(shape, (16, 16, 16), 12.0)
    roi_data = _create_sphere(shape, (22, 20, 18), 4.0, dtype=np.uint8)

    flair_img = nib.Nifti1Image(flair_data, affine)
    roi_img = nib.Nifti1Image(roi_data, affine)
    t1_img = nib.Nifti1Image(flair_data * 1.2, affine)

    res = validate_subject_geometry_and_roi(
        flair_img=flair_img,
        t1_img=t1_img,
        roi_img=roi_img,
        raw_flair_img=flair_img,
        raw_roi_img=roi_img,
        subject_id="sub-test01",
        is_hc=False,
    )

    assert res["status"] == "PASS"
    assert res["max_geometry_diff_mm"] == 0.0
    assert res["roi_centroid_distance_mm"] == 0.0
    assert res["roi_voxels_before"] == res["roi_voxels_after"]
    assert res["roi_voxels_after"] > 0
    assert len(res["errors"]) == 0


# ---------------------------------------------------------------------------
# Test 2: ROI with different spacing but the same physical location
# ---------------------------------------------------------------------------
def test_roi_different_spacing_same_physical_location(tmp_path: Path):
    # Reference FLAIR: spacing 1.0 mm, shape (40, 40, 40)
    flair_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    flair_affine[:3, 3] = [-20.0, -20.0, -20.0]
    flair_data = _create_sphere((40, 40, 40), (20, 20, 20), 15.0)
    flair_img = nib.Nifti1Image(flair_data, flair_affine)

    # ROI acquired at spacing 0.8 mm, shape (50, 50, 50)
    # Lesion physically centered at world coordinates (0, 0, 0)
    roi_affine = np.diag([0.8, 0.8, 0.8, 1.0])
    roi_affine[:3, 3] = [-20.0, -20.0, -20.0]  # world center is at voxel (25, 25, 25)
    roi_data = _create_sphere((50, 50, 50), (25, 25, 25), 5.0, dtype=np.uint8)
    roi_raw_img = nib.Nifti1Image(roi_data, roi_affine)

    # Resample ROI onto reference FLAIR grid using nearest-neighbor
    resampled_roi = resample_to_reference(roi_raw_img, flair_img, is_mask=True)

    # Physical centroid must match between raw and resampled ROI
    raw_c = compute_roi_physical_centroid(roi_data, roi_affine)
    res_c = compute_roi_physical_centroid(np.asarray(resampled_roi.dataobj), resampled_roi.affine)
    assert raw_c is not None and res_c is not None
    dist = np.linalg.norm(np.array(raw_c) - np.array(res_c))
    assert dist < 0.2  # less than 0.2 mm shift

    t1_img = nib.Nifti1Image(flair_data, flair_affine)
    res = validate_subject_geometry_and_roi(
        flair_img=flair_img,
        t1_img=t1_img,
        roi_img=resampled_roi,
        raw_flair_img=flair_img,
        raw_roi_img=roi_raw_img,
        subject_id="sub-test02",
        is_hc=False,
    )
    assert res["status"] == "PASS"


# ---------------------------------------------------------------------------
# Test 3: ROI with different origin
# ---------------------------------------------------------------------------
def test_roi_different_origin(tmp_path: Path):
    flair_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    flair_affine[:3, 3] = [-10.0, -10.0, -10.0]
    flair_img = nib.Nifti1Image(_create_sphere((30, 30, 30), (15, 15, 15), 10.0), flair_affine)

    # ROI has a shifted origin of (-15, -15, -15) and larger shape
    roi_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    roi_affine[:3, 3] = [-15.0, -15.0, -15.0]
    # Physical lesion at world (2, 2, 2) -> in ROI voxel space (17, 17, 17)
    roi_data = _create_sphere((40, 40, 40), (17, 17, 17), 3.0, dtype=np.uint8)
    roi_raw_img = nib.Nifti1Image(roi_data, roi_affine)

    resampled_roi = resample_to_reference(roi_raw_img, flair_img, is_mask=True)
    raw_c = compute_roi_physical_centroid(roi_data, roi_affine)
    res_c = compute_roi_physical_centroid(np.asarray(resampled_roi.dataobj), resampled_roi.affine)
    assert raw_c is not None and res_c is not None
    assert np.linalg.norm(np.array(raw_c) - np.array(res_c)) < 0.05


# ---------------------------------------------------------------------------
# Test 4: ROI with different orientation
# ---------------------------------------------------------------------------
def test_roi_different_orientation(tmp_path: Path):
    # FLAIR in standard RAS
    flair_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    flair_img = nib.Nifti1Image(_create_sphere((30, 30, 30), (15, 15, 15), 10.0), flair_affine)

    # ROI in LAS (X flipped)
    roi_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    roi_affine[0, 3] = 30.0  # offset for positive voxel indices
    # Physical sphere at world (10, 15, 15) -> voxel coords: 30 - 10 = 20
    roi_data = _create_sphere((30, 30, 30), (15, 15, 20), 4.0, dtype=np.uint8)
    roi_raw_img = nib.Nifti1Image(roi_data, roi_affine)

    resampled_roi = resample_to_reference(roi_raw_img, flair_img, is_mask=True)
    raw_c = compute_roi_physical_centroid(roi_data, roi_affine)
    res_c = compute_roi_physical_centroid(np.asarray(resampled_roi.dataobj), resampled_roi.affine)
    assert raw_c is not None and res_c is not None
    assert np.linalg.norm(np.array(raw_c) - np.array(res_c)) < 0.2


# ---------------------------------------------------------------------------
# Test 5: Correct nearest-neighbor resampling
# ---------------------------------------------------------------------------
def test_correct_nearest_neighbor_resampling():
    shape = (20, 20, 20)
    data = np.zeros(shape, dtype=np.uint8)
    data[8:12, 8:12, 8:12] = 1
    src_img = nib.Nifti1Image(data, np.eye(4))

    # Target grid with arbitrary floating spacing and origin
    tgt_affine = np.diag([1.37, 1.37, 1.37, 1.0])
    tgt_affine[:3, 3] = [0.45, -0.22, 1.1]
    tgt_img = nib.Nifti1Image(np.zeros((15, 15, 15)), tgt_affine)

    resampled = resample_to_reference(src_img, tgt_img, is_mask=True)
    res_data = np.asarray(resampled.dataobj)
    # Output must strictly contain discrete integer labels only
    unique_vals = set(np.unique(res_data))
    assert unique_vals.issubset({0, 1})
    assert np.count_nonzero(res_data) > 0


# ---------------------------------------------------------------------------
# Test 6: Binary label preservation
# ---------------------------------------------------------------------------
def test_binary_label_preservation():
    shape = (20, 20, 20)
    affine = np.eye(4)
    # Invalid mask containing value 2
    bad_data = np.zeros(shape, dtype=np.uint8)
    bad_data[5:8, 5:8, 5:8] = 2
    bad_img = nib.Nifti1Image(bad_data, affine)
    flair_img = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)

    res = validate_subject_geometry_and_roi(
        flair_img=flair_img,
        t1_img=flair_img,
        roi_img=bad_img,
        subject_id="sub-invalid",
        is_hc=False,
    )
    assert res["status"] == "FAIL"
    assert any("binary 0 and 1" in e for e in res["errors"])


# ---------------------------------------------------------------------------
# Test 7: ROI voxel count preservation within reasonable tolerance
# ---------------------------------------------------------------------------
def test_roi_voxel_count_preservation():
    shape = (30, 30, 30)
    affine = np.eye(4)
    roi_data = _create_sphere(shape, (15, 15, 15), 5.0, dtype=np.uint8)
    orig_count = np.count_nonzero(roi_data)

    flair_img = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    roi_img = nib.Nifti1Image(roi_data, affine)

    # Identical grid must have exactly identical voxel count
    res = validate_subject_geometry_and_roi(
        flair_img=flair_img,
        t1_img=flair_img,
        roi_img=roi_img,
        raw_flair_img=flair_img,
        raw_roi_img=roi_img,
        subject_id="sub-count",
        is_hc=False,
    )
    assert res["status"] == "PASS"
    assert res["roi_voxels_before"] == orig_count
    assert res["roi_voxels_after"] == orig_count


# ---------------------------------------------------------------------------
# Test 8: Correct registered T1 reference geometry
# ---------------------------------------------------------------------------
def test_registered_t1_reference_geometry(tmp_path: Path):
    flair_shape = (24, 28, 28)
    flair_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    flair_affine[:3, 3] = [-12.0, -14.0, -14.0]
    flair_p = tmp_path / "flair.nii.gz"
    nib.save(nib.Nifti1Image(_create_sphere(flair_shape, (12, 14, 14), 8.0), flair_affine), str(flair_p))

    t1_shape = (30, 35, 35)
    t1_affine = np.diag([0.8, 0.8, 0.8, 1.0])
    t1_affine[:3, 3] = [-12.0, -14.0, -14.0]
    t1_p = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(_create_sphere(t1_shape, (15, 17, 17), 10.0), t1_affine), str(t1_p))

    out_p = tmp_path / "reg_t1.nii.gz"
    register_to_reference(
        moving_path=t1_p,
        reference_path=flair_p,
        output_path=out_p,
        method="rigid",
        number_of_iterations=15,
        number_of_bins=16,
        sampling_percentage=0.5,
    )

    reg_nib = nib.load(str(out_p))
    flair_nib = nib.load(str(flair_p))
    assert reg_nib.shape == flair_shape
    assert np.allclose(reg_nib.affine, flair_affine, atol=1e-4)
    assert same_grid(flair_nib, reg_nib, tolerance_mm=0.01)


# ---------------------------------------------------------------------------
# Test 9: Correct transform direction
# ---------------------------------------------------------------------------
def test_correct_transform_direction(tmp_path: Path):
    # Reference FLAIR: sphere centered at (12, 12, 12)
    flair_shape = (24, 24, 24)
    flair_affine = np.eye(4)
    flair_data = _create_sphere(flair_shape, (12, 12, 12), 6.0)
    flair_p = tmp_path / "flair_fixed.nii.gz"
    nib.save(nib.Nifti1Image(flair_data, flair_affine), str(flair_p))

    # Moving T1: sphere centered at (15, 12, 12) (+3 mm in z)
    t1_data = _create_sphere(flair_shape, (15, 12, 12), 6.0)
    t1_p = tmp_path / "t1_moving.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, flair_affine), str(t1_p))

    out_p = tmp_path / "t1_aligned.nii.gz"
    register_to_reference(
        moving_path=t1_p,
        reference_path=flair_p,
        output_path=out_p,
        method="rigid",
        number_of_iterations=40,
        number_of_bins=20,
        sampling_percentage=0.5,
    )

    res_arr = nib.load(str(out_p)).get_fdata()
    center_res = np.round(np.argwhere(res_arr > 50).mean(axis=0))
    # Resampled sphere must align with fixed FLAIR center (12, 12, 12), not (15, 12, 12)
    assert np.allclose(center_res, [12, 12, 12], atol=1.0)


# ---------------------------------------------------------------------------
# Test 10: No accidental application of T1 transform to the ROI
# ---------------------------------------------------------------------------
def test_no_accidental_application_of_t1_transform_to_roi():
    # Synthetic scenario: T1 has an offset, but ROI is defined in FLAIR space.
    # The ROI must NOT be moved by any registration transform.
    shape = (30, 30, 30)
    affine = np.eye(4)
    flair_img = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    roi_data = _create_sphere(shape, (10, 15, 20), 4.0, dtype=np.uint8)
    roi_img = nib.Nifti1Image(roi_data, affine)

    # Suppose an erroneous pipeline applied a 5mm shift to the ROI
    shifted_affine = affine.copy()
    shifted_affine[0, 3] += 5.0
    erroneous_roi_img = nib.Nifti1Image(roi_data, shifted_affine)

    # Validation must detect this centroid shift and fail
    res_err = validate_subject_geometry_and_roi(
        flair_img=flair_img,
        t1_img=flair_img,
        roi_img=erroneous_roi_img,
        raw_flair_img=flair_img,
        raw_roi_img=roi_img,
        subject_id="sub-shift-test",
        is_hc=False,
    )
    assert res_err["status"] == "FAIL"
    assert any("shifted by" in e for e in res_err["errors"])


# ---------------------------------------------------------------------------
# Test 11: HC zero-mask generation
# ---------------------------------------------------------------------------
def test_hc_zero_mask_generation():
    shape = (25, 25, 25)
    affine = np.eye(4)
    flair_img = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    t1_img = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    zero_roi_img = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)

    res = validate_subject_geometry_and_roi(
        flair_img=flair_img,
        t1_img=t1_img,
        roi_img=zero_roi_img,
        subject_id="sub-hc01",
        is_hc=True,
    )
    assert res["status"] == "PASS"
    assert res["roi_voxels_after"] == 0
    assert len(res["errors"]) == 0


# ---------------------------------------------------------------------------
# Test 12: Final nnU-Net label geometry across datasets
# ---------------------------------------------------------------------------
def test_final_nnunet_label_geometry(tmp_path: Path):
    shape = (20, 20, 20)
    affine = np.eye(4)
    flair_img = nib.Nifti1Image(_create_sphere(shape, (10, 10, 10), 6.0), affine)
    t1_img = nib.Nifti1Image(_create_sphere(shape, (10, 10, 10), 6.0), affine)
    roi_img = nib.Nifti1Image(_create_sphere(shape, (10, 10, 10), 3.0, dtype=np.uint8), affine)

    flair_dir = tmp_path / "Dataset101_FLAIR"
    t1_dir = tmp_path / "Dataset102_T1"
    fusion_dir = tmp_path / "Dataset103_Fusion"

    write_subject_to_nnunet(
        subject_id="sub-001",
        flair_image=flair_img,
        t1_image=t1_img,
        roi_image=roi_img,
        split="train",
        flair_dataset_dir=flair_dir,
        t1_dataset_dir=t1_dir,
        fusion_dataset_dir=fusion_dir,
    )

    f_lbl = nib.load(str(flair_dir / "labelsTr" / "sub-001.nii.gz"))
    t_lbl = nib.load(str(t1_dir / "labelsTr" / "sub-001.nii.gz"))
    fus_lbl = nib.load(str(fusion_dir / "labelsTr" / "sub-001.nii.gz"))

    assert same_grid(f_lbl, t_lbl, tolerance_mm=0.001)
    assert same_grid(f_lbl, fus_lbl, tolerance_mm=0.001)
    assert np.array_equal(np.asarray(f_lbl.dataobj), np.asarray(t_lbl.dataobj))
    assert np.array_equal(np.asarray(f_lbl.dataobj), np.asarray(fus_lbl.dataobj))


# ---------------------------------------------------------------------------
# Test 13: QC generation with ROI over registered T1 and off-center lesion
# ---------------------------------------------------------------------------
def test_qc_generation_with_roi_over_registered_t1(tmp_path: Path):
    # Lesion far off-center at (5, 6, 7), while brain center is at (15, 15, 15)
    shape = (30, 30, 30)
    affine = np.eye(4)
    brain_data = _create_sphere(shape, (15, 15, 15), 12.0)
    roi_data = _create_sphere(shape, (5, 6, 7), 3.0, dtype=np.uint8)

    flair_p = tmp_path / "fixed_flair.nii.gz"
    t1_p = tmp_path / "reg_t1.nii.gz"
    roi_p = tmp_path / "roi.nii.gz"

    nib.save(nib.Nifti1Image(brain_data, affine), str(flair_p))
    nib.save(nib.Nifti1Image(brain_data * 1.1, affine), str(t1_p))
    nib.save(nib.Nifti1Image(roi_data, affine), str(roi_p))

    png_out = tmp_path / "qc_reg.png"
    res = create_registration_qc_visuals(
        fixed_path=flair_p,
        resampled_moving_path=t1_p,
        subject_id="sub-test13",
        output_png_path=png_out,
        return_base64=True,
        roi_path=roi_p,
        is_hc=False,
        raw_flair_path=flair_p,
        raw_roi_path=roi_p,
    )

    assert res["status"] == "PASS"
    assert res["roi_overlay"] is True
    assert res["roi_lesion_voxels"] == np.count_nonzero(roi_data)
    assert png_out.is_file()
    assert png_out.stat().st_size > 1000
    assert res["base64_image"].startswith("data:image/png;base64,")
