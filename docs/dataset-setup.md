# Dataset Setup and Audit

## Important privacy rule

Do not commit original MRI scans, masks, participant IDs, clinical records, or derived volumes to GitHub. Keep data on approved storage and follow the dataset provider's access, consent, and redistribution conditions. `.gitignore` is a safeguard, not a substitute for checking `git status` before every commit.

## Local directories

The repository reserves:

```text
data/
├── raw/        # Original, unmodified data
└── processed/  # Derived data created by preprocessing
outputs/        # Logs, checkpoints, predictions, metrics, visualizations
```

The directories contain `.gitkeep` files so Git can preserve the empty folder layout. Put actual data locally; do not replace `.gitkeep` with data files in a commit.

## Do not assume a dataset naming convention

The exact dataset has not yet been specified in this documentation. Before using real data, the team must record:
- Dataset name/version and approved access location.
- Whether images are NIfTI (`.nii` or `.nii.gz`), BIDS-organized, or in another format.
- Subject and session naming rules.
- How FLAIR and T1-weighted images are identified.
- How segmentation masks are named and encoded.
- Whether each subject has every modality.
- Whether images and masks share the same voxel grid and affine.
- Which label values represent background, lesion, and any other class.
- Which files are excluded and why.

Do not rename files or invent mask labels based on assumptions. Derive the dataset contract from the dataset documentation and project report.

## Suggested organization (only if compatible with the real dataset)

The following is an illustrative layout, not a mandatory format:

```text
data/raw/dataset/
├── sub-001/
│   ├── flair.nii.gz
│   ├── t1.nii.gz
│   └── lesion_mask.nii.gz
├── sub-002/
│   ├── flair.nii.gz
│   ├── t1.nii.gz
│   └── lesion_mask.nii.gz
└── ...
```

If the dataset uses BIDS or another existing standard, preserve its native structure instead of forcing it into this example. Implement discovery rules to match the actual dataset.

## Current audit command

From the repository root with the virtual environment activated:

```bash
python scripts/audit_dataset.py --help
python scripts/audit_dataset.py --data-root data/raw/dataset
```

The audit currently inventories candidate FLAIR files. It is **not** a complete BIDS validator and does not certify:
- that every subject has all expected modalities;
- that masks exist and use correct label values;
- that images and labels are spatially aligned;
- that registration succeeded;
- that image intensities are valid;
- that the data are suitable for training.

Read the script's `--help` output for actual supported options.

## Dataset audit checklist

Before preprocessing or training, the team should confirm:

- [ ] Data access and research permissions are documented.
- [ ] A stable subject ID mapping exists locally and is protected if sensitive.
- [ ] All expected images are present and readable.
- [ ] Required modalities are paired with the correct subject/session.
- [ ] Mask files exist where expected.
- [ ] Image and mask dimensions, affine matrices, orientations, and voxel spacings have been inspected.
- [ ] Mask label values match the dataset documentation.
- [ ] Empty masks and positive cases have been counted and reviewed.
- [ ] Missing modalities and corrupted files are explicitly reported.
- [ ] Duplicate scans and multiple sessions are handled consistently.
- [ ] Train/validation/test splits are subject-level and recorded.
- [ ] No subject or derived patch leaks across splits.
- [ ] Raw data remain unchanged and untracked by Git.

## Spatial metadata matters

Two images with equal array dimensions are not necessarily spatially aligned. Compare affine matrices, orientations, spacing, and coordinate conventions. Registration and resampling must be explicit, tested, and logged. Use label-safe interpolation for segmentation masks; continuous interpolation can create invalid fractional class labels.

Do not overwrite raw files. Store prepared images in `data/processed/` and retain enough metadata to reproduce the transformation.

## Train/validation/test split policy

Split by participant, not by slice or patch. If multiple sessions belong to one participant, keep that participant in one split unless the research protocol explicitly defines a different leakage-safe strategy. Select models and hyperparameters using validation data; reserve the test set for final evaluation.

## Data inventory reporting

The audit report should avoid including names or paths that disclose sensitive participant information when shared externally. Share summary counts and validation errors with the team; keep any identity mapping separate and access-controlled.
