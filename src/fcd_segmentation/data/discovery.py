"""Portable discovery for BIDS-like subject/anat folders."""
from pathlib import Path

def discover_flair_files(data_root):
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    candidates = list(root.glob("sub-*/anat/*_FLAIR.nii"))
    candidates += list(root.glob("sub-*/anat/*_FLAIR.nii.gz"))
    found = sorted(p for p in set(candidates)
                   if not p.name.endswith("_FLAIR_roi.nii")
                   and not p.name.endswith("_FLAIR_roi.nii.gz"))
    if not found:
        raise FileNotFoundError(f"No FLAIR NIfTI images found under {root}/sub-*/anat/")
    return found

def discover_roi_files(subject_anat_dir):
    folder = Path(subject_anat_dir)
    candidates = list(folder.glob("*_FLAIR_roi.nii"))
    candidates += list(folder.glob("*_FLAIR_roi.nii.gz"))
    return sorted(set(p for p in candidates if p.is_file()))
