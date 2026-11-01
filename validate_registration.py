from pathlib import Path
import numpy as np
import SimpleITK as sitk

project_root = Path(__file__).resolve().parent

t1_path = (
    project_root
    / "data"
    / "raw"
    / "dataset"
    / "sub-00001"
    / "anat"
    / "sub-00001_acq-iso08_T1w.nii.gz"
)

registered_flair_path = (
    project_root
    / "data"
    / "processed"
    / "registration"
    / "sub-00001"
    / "sub-00001_FLAIR_registered.nii.gz"
)

if not t1_path.is_file():
    raise FileNotFoundError(f"T1 reference not found: {t1_path}")

if not registered_flair_path.is_file():
    raise FileNotFoundError(
        f"Registered FLAIR not found: {registered_flair_path}"
    )

t1 = sitk.ReadImage(str(t1_path))
flair = sitk.ReadImage(str(registered_flair_path))

print("T1 size:", t1.GetSize())
print("Registered FLAIR size:", flair.GetSize())

print("\nT1 spacing:", t1.GetSpacing())
print("Registered FLAIR spacing:", flair.GetSpacing())

print("\nT1 origin:", t1.GetOrigin())
print("Registered FLAIR origin:", flair.GetOrigin())

print("\nT1 direction:", t1.GetDirection())
print("Registered FLAIR direction:", flair.GetDirection())

checks = {
"Size": t1.GetSize() == flair.GetSize(),
"Spacing": np.allclose(
t1.GetSpacing(), flair.GetSpacing(), rtol=0, atol=1e-6
),
"Origin": np.allclose(
t1.GetOrigin(), flair.GetOrigin(), rtol=0, atol=1e-5
),
"Direction": np.allclose(
t1.GetDirection(), flair.GetDirection(), rtol=0, atol=1e-6
),
}

print("\n--- Geometry validation ---")

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

if not all(checks.values()):
    raise AssertionError(
        "The registered FLAIR geometry does not match the T1 reference."
    )

print("\nAll geometry checks passed.")