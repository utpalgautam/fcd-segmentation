"""Reference-grid resampling; masks use nearest-neighbor interpolation."""

from pathlib import Path


def resample_to_reference(image, reference, *, is_mask=False):
    try:
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise RuntimeError(
            "Install with: python -m pip install -e '.[imaging]'"
        ) from exc
    return resample_from_to(
        image, (reference.shape, reference.affine), order=0 if is_mask else 1
    )


def resample_file_to_reference(
    input_path, reference_path, output_path, *, is_mask=False, overwrite=False
):
    from fcd_segmentation.preprocessing.io import load_nifti, save_nifti

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    image = load_nifti(input_path)
    reference = load_nifti(reference_path)
    return save_nifti(
        resample_to_reference(image, reference, is_mask=is_mask),
        output_path,
        overwrite=overwrite,
    )
