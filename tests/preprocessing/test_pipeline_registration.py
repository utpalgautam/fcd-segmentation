"""Unit tests for registration module."""

import nibabel as nib
import numpy as np

from fcd_segmentation.preprocessing.geometry import same_grid
from fcd_segmentation.preprocessing.registration import (
    create_registration_qc_visuals,
    register_to_reference,
)


def _create_sphere_volume(shape, center, radius):
    vol = np.zeros(shape, dtype=np.float32)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    mask = dist <= radius
    vol[mask] = 100.0 - dist[mask] * 5.0
    return vol


def test_registration_direction_and_resampling(tmp_path):
    # Reference (FLAIR): (20, 24, 24), spacing 1.0
    flair_shape = (20, 24, 24)
    flair_data = _create_sphere_volume(flair_shape, (10, 12, 12), 6.0)
    flair_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    flair_img = nib.Nifti1Image(flair_data, flair_affine)
    flair_path = tmp_path / "flair.nii.gz"
    nib.save(flair_img, str(flair_path))

    # Moving (T1): (25, 30, 30), spacing 0.8, slightly shifted
    t1_shape = (25, 30, 30)
    t1_data = _create_sphere_volume(t1_shape, (12, 15, 15), 7.5)
    t1_affine = np.diag([0.8, 0.8, 0.8, 1.0])
    t1_img = nib.Nifti1Image(t1_data, t1_affine)
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(t1_img, str(t1_path))

    out_resampled = tmp_path / "t1_resampled.nii.gz"
    out_tfm = tmp_path / "transform.tfm"

    res = register_to_reference(
        moving_path=t1_path,
        reference_path=flair_path,
        output_path=out_resampled,
        transform_output_path=out_tfm,
        method="rigid",
        number_of_iterations=30,
        number_of_bins=20,
        sampling_percentage=0.5,
    )

    assert out_resampled.is_file()
    assert out_tfm.is_file()
    assert res["final_metric"] < 0

    # Verify resampled T1 is on exact FLAIR grid
    resampled_nib = nib.load(str(out_resampled))
    assert resampled_nib.shape == flair_shape
    assert same_grid(flair_img, resampled_nib, tolerance_mm=0.01)


def test_create_registration_qc_visuals(tmp_path):
    shape = (16, 16, 16)
    data1 = _create_sphere_volume(shape, (8, 8, 8), 5.0)
    data2 = _create_sphere_volume(shape, (8, 8, 8), 5.0)

    p1 = tmp_path / "fixed.nii.gz"
    p2 = tmp_path / "resampled.nii.gz"
    nib.save(nib.Nifti1Image(data1, np.eye(4)), str(p1))
    nib.save(nib.Nifti1Image(data2, np.eye(4)), str(p2))

    qc_res = create_registration_qc_visuals(
        fixed_path=p1,
        resampled_moving_path=p2,
        subject_id="sub-test",
        return_base64=True,
    )

    assert qc_res["status"] == "PASS"
    assert qc_res["size_match"] is True
    assert qc_res["spacing_match"] is True
    assert qc_res["base64_image"].startswith("data:image/png;base64,")


def test_create_registration_qc_visuals_with_roi(tmp_path):
    shape = (16, 16, 16)
    data1 = _create_sphere_volume(shape, (8, 8, 8), 5.0)
    data2 = _create_sphere_volume(shape, (8, 8, 8), 5.0)
    roi_data = np.zeros(shape, dtype=np.uint8)
    roi_data[6:10, 6:10, 6:10] = 1

    p1 = tmp_path / "fixed.nii.gz"
    p2 = tmp_path / "resampled.nii.gz"
    roi_p = tmp_path / "roi.nii.gz"
    nib.save(nib.Nifti1Image(data1, np.eye(4)), str(p1))
    nib.save(nib.Nifti1Image(data2, np.eye(4)), str(p2))
    nib.save(nib.Nifti1Image(roi_data, np.eye(4)), str(roi_p))

    qc_res = create_registration_qc_visuals(
        fixed_path=p1,
        resampled_moving_path=p2,
        subject_id="sub-test-roi",
        return_base64=True,
        roi_path=roi_p,
    )

    assert qc_res["status"] == "PASS"
    assert qc_res["roi_overlay"] is True
    assert qc_res["roi_lesion_voxels"] == 64  # 4*4*4
    assert qc_res["base64_image"].startswith("data:image/png;base64,")
