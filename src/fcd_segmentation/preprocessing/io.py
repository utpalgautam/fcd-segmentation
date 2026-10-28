"""Minimal NIfTI I/O; requires the imaging extra."""
from pathlib import Path

def load_nifti(path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("Install with: python -m pip install -e '.[imaging]'") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D image, got {image.shape}")
    return image

def save_nifti(image, path, *, overwrite=False):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("Install with: python -m pip install -e '.[imaging]'") from exc
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
    return path
