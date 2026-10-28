"""Basic NIfTI geometry checks."""
import numpy as np

def same_grid(first, second, *, tolerance_mm=0.01):
    if first.shape != second.shape:
        return False
    return bool(np.allclose(first.affine, second.affine, atol=tolerance_mm, rtol=0))

def validate_3d_image(image):
    if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
        raise ValueError(f"Expected a non-empty 3D image, got {image.shape}")
    if not np.isfinite(np.asarray(image.dataobj)).all():
        raise ValueError("Image contains non-finite voxel values")
