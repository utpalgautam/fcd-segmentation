from pathlib import Path

import nibabel as nib

dataset_dir = Path("data/raw/dataset")

t1_files = sorted(dataset_dir.glob("sub-*/anat/*_T1w.nii.gz"))

for t1_path in t1_files:
    subject_dir = t1_path.parent
    subject_id = t1_path.parts[-3]

    flair_files = list(subject_dir.glob("*_FLAIR.nii.gz"))

    if not flair_files:
        continue

    flair_path = flair_files[0]

    try:
        t1 = nib.load(str(t1_path))
        flair = nib.load(str(flair_path))

        print(f"\nSubject: {subject_id}")
        print(f"T1: {t1_path}")
        print(f"FLAIR: {flair_path}")
        print(f"T1 shape: {t1.shape}")
        print(f"FLAIR shape: {flair.shape}")
        print(f"T1 affine:\n{t1.affine}")
        print(f"FLAIR affine:\n{flair.affine}")

        break

    except Exception as exc:
        print(f"Could not read {subject_id}: {exc}")