from pathlib import Path

from fcd_segmentation.preprocessing.registration import (
    register_to_reference,
)

# Project root
project_root = Path(__file__).resolve().parent

# Subject sub-00001
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

# Output directory
output_dir = (
    project_root
    / "data"
    / "processed"
    / "registration"
    / "sub-00001"
)

output_path = output_dir / "sub-00001_FLAIR_registered.nii.gz"

transform_path = output_dir / "flair_to_t1.tfm"

print("Starting FLAIR-to-T1 registration...")
print(f"T1 reference: {t1_path}")
print(f"FLAIR moving image: {flair_path}")

registered_path = register_to_reference(
    moving_path=flair_path,
    reference_path=t1_path,
    output_path=output_path,
    method="rigid",
    transform_output_path=transform_path,
)

print("\nRegistration test completed.")
print(f"Registered FLAIR: {registered_path}")
print(f"Saved transformation: {transform_path}")