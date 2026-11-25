# Dataset Setup & Preprocessing Guide

This document details the dataset specifications, download procedure, cohort integrity rules, and preprocessing transformations for the FCD (Focal Cortical Dysplasia) lesion segmentation pipeline.

---

## 1. Dataset Overview

The project uses the open-access Focal Cortical Dysplasia MRI benchmark from Zenodo:

* **Training Cohort:** 142 total subjects
  * **57 FCD Patients** (confirmed Type II FCD lesions with manual expert segmentations)
  * **85 Healthy Controls** (clean brain scans with all-zero background masks)
* **Held-Out Test Cohort:** 28 subjects (strictly FCD patients for independent external evaluation)
* **Total Scans:** 170 subjects $\times$ 2 modalities = 340 3D MRI volumes.

---

## 2. Directory Layout & Data Contracts

Raw and processed data follow standardized structures:

```text
data/
├── raw/
│   └── dataset/
│       ├── participants.tsv           # Clinical metadata, demographics, group, split assignments
│       ├── sub-00001/                 # Per-subject BIDS hierarchy
│       │   ├── anat/
│       │   │   ├── sub-00001_FLAIR.nii.gz
│       │   │   └── sub-00001_T1w.nii.gz
│       │   └── mask/
│       │       └── sub-00001_mask.nii.gz
│       └── ...
└── processed/
    ├── Dataset101_FLAIR/              # Single-modality FLAIR benchmark
    │   ├── dataset.json
    │   ├── imagesTr/ (sub-XXXXX_0000.nii.gz)
    │   ├── labelsTr/ (sub-XXXXX.nii.gz)
    │   ├── imagesTs/
    │   └── labelsTs/
    ├── Dataset102_T1/                 # Single-modality T1w (registered to FLAIR space)
    └── Dataset103_Fusion/             # Multimodal 2-channel paired (Channel 0: FLAIR, Channel 1: T1)
```

---

## 3. Dataset Acquisition & Download

Download and automatically unpack the benchmark dataset:

```bash
python scripts/download_dataset.py
```

Options:
- `--extract-dir <path>`: Destination directory (defaults to `data/raw/dataset`).
- `--record-id <zenodo_id>`: Override Zenodo deposit identifier.

---

## 4. Preprocessing Pipeline

Run end-to-end preprocessing to construct Datasets 101, 102, and 103:

```bash
python scripts/run_preprocessing.py --all-modalities
```

### Preprocessing Operations:
1. **HD-BET Skull Stripping:** Extracts intracranial brain masks and removes non-brain tissue (skull, scalp, meninges) using deep learning brain extraction.
2. **T1-to-FLAIR Rigid Registration:** Aligns the high-resolution T1 volume to the native FLAIR geometry using affine mutual information registration, guaranteeing identical voxel grids and voxel-to-voxel spatial correspondence.
3. **Discrete Binary Mask Standardization:** Normalizes all lesion segmentations to discrete binary labels (`0` = background, `1` = FCD lesion).
4. **Group-Stratified 5-Fold Splitting:** Generates reproducible cross-validation folds (`splits_final.json`) preserving the exact FCD:HC clinical distribution across all five folds.

---

## 5. Dataset Integrity & Cohort Verification

Run automated pre-training integrity verification:

```bash
# Verify cohort case counts, label binarization, and volume headers
python scripts/train.py --validate --dataset all
```

Expected output:
```text
Workflow action 'validate' finished. Status: SUCCESS
Selected Datasets: ['Dataset101_FLAIR', 'Dataset102_T1', 'Dataset103_Fusion']
  [VALID] Dataset101_FLAIR (142 case(s)): data/processed/Dataset101_FLAIR
  [VALID] Dataset102_T1 (142 case(s)): data/processed/Dataset102_T1
  [VALID] Dataset103_Fusion (142 case(s)): data/processed/Dataset103_Fusion
Cohort Guard: PASSED
```
