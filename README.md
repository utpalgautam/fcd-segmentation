# FCD MRI Segmentation: Complete Preprocessing, Training & Evaluation Pipeline

[![Tests](https://img.shields.io/badge/tests-243%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-orange.svg)]()

Cross-platform research pipeline for focal cortical dysplasia (FCD) lesion segmentation using multimodal structural MRI (T1-weighted and FLAIR).

This repository provides an automated, auditable, and fully validated end-to-end medical image computing pipeline that covers the entire lifecycle:
1. **Raw BIDS Data Ingestion**: Automated OpenNeuro dataset retrieval (`ds004199`) and participant manifest validation.
2. **Preprocessing & Geometric Alignment**: Brain extraction via HD-BET, rigid registration of T1 onto FLAIR reference space, discrete lesion ROI alignment, and masked intensity normalization.
3. **nnU-Net v2 Dataset Packaging**: Assembly of three standardized datasets:
   - **`Dataset101_FLAIR`** — Skull-stripped, normalized FLAIR images (1 channel: `_0000.nii.gz`).
   - **`Dataset102_T1`** — Skull-stripped, rigid-registered, and resampled T1w images mapped onto FLAIR voxel grid (1 channel: `_0000.nii.gz`).
   - **`Dataset103_Fusion`** — Multi-channel paired inputs on the exact same anatomical grid (`channel 0000` = FLAIR, `channel 0001` = T1).
4. **Leak-Free 5-Fold Stratified Cross-Validation**: Deterministic cohort partitioning (stratified by FCD vs. HC) with strict held-out test cohort isolation.
5. **Lab-Ready Training & Preflight**: Preflight system diagnostics, custom calibrated 100-epoch nnU-Net trainer (`FCDnnUNetTrainer_100epochs`), and 5-fold training execution.
6. **Inference & Scientific Evaluation**: Fold-aware validation, held-out test inference, and robust metric reporting (Dice, IoU, Sensitivity, Specificity, Precision, and finite/infinite HD95 statistics).

---

## Table of Contents

- [Benchmark Results & Multi-Modality Outcomes](#benchmark-results--multi-modality-outcomes)
- [Pipeline Overview & Architecture](#pipeline-overview--architecture)
- [Dataset Summary & Cohorts](#dataset-summary--cohorts)
- [Project Status & Implemented Features](#project-status--implemented-features)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Step-by-Step Pipeline Execution](#step-by-step-pipeline-execution)
  - [Step 1: Download Raw OpenNeuro Dataset](#step-1-download-raw-openneuro-dataset)
  - [Step 2: Run Preprocessing Pipeline](#step-2-run-preprocessing-pipeline)
  - [Step 3: Preflight System & Environment Check](#step-3-preflight-system--environment-check)
  - [Step 4: Generate Stratified 5-Fold Cross-Validation Splits](#step-4-generate-stratified-5-fold-cross-validation-splits)
  - [Step 5: nnU-Net Experiment Planning & Preprocessing](#step-5-nnu-net-experiment-planning--preprocessing)
  - [Step 6: Train Models](#step-6-train-models)
  - [Step 7: Model Inference / Predictions](#step-7-model-inference--predictions)
  - [Step 8: Metric Evaluation & Scientific Reporting](#step-8-metric-evaluation--scientific-reporting)
  - [Step 9: Automated Post-Training Pipeline & Visual Reporting](#step-9-automated-post-training-pipeline--visual-reporting)
- [Documentation Index](#documentation-index)
- [Quality Control (QC) Reports](#quality-control-qc-reports)
- [Testing & Validation](#testing--validation)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)
- [Dataset Privacy & Handling](#dataset-privacy--handling)

---

## Benchmark Results & Multi-Modality Outcomes

Full-resolution 3D models trained for 100 epochs (`FCDnnUNetTrainer_100epochs`, `3d_fullres`, `nnUNetPlans`, $128 \times 160 \times 112$ patch size) evaluated on the Fold 0 validation cohort (29 cases: 12 FCD patients + 17 Healthy Controls):

| Modality / Dataset | FCD-Only Dice | Overall Cohort Dice | Sensitivity | Specificity | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset 103 (Multimodal Fusion)** | **51.04%** | 45.26% | 45.73% | **99.99%** | Virtually eliminated |
| **Dataset 101 (FLAIR Baseline)** | **49.72%** | **61.95%** | **63.52%** | **100.00%** | Zero on controls |
| **Dataset 102 (T1w Baseline)** | **27.25%** | 45.76% | 44.64% | **100.00%** | Zero on controls |

> [!TIP]
> **Complete Scientific Report:** For in-depth patient-by-patient metrics, cross-modal synergy discussion, and statistical breakdown, see the [Comprehensive Benchmark Report](docs/benchmark_report.md).

### Visual Segmentation Reports (Interactive HTML)
Open these reports directly in any browser to inspect slice-by-slice 4-panel overlays (MRI, Ground Truth, Prediction, and Green/Red Overlay):
- **FLAIR Report:** [outputs/segmentation_report_Dataset101_FLAIR_fold0.html](outputs/segmentation_report_Dataset101_FLAIR_fold0.html) (4.7 MB)
- **T1 Report:** [outputs/segmentation_report_Dataset102_T1_fold0.html](outputs/segmentation_report_Dataset102_T1_fold0.html) (3.9 MB)
- **Fusion Report:** [outputs/segmentation_report_Dataset103_Fusion_fold0.html](outputs/segmentation_report_Dataset103_Fusion_fold0.html) (4.8 MB)

### Top-Performing Validation Patients
- **`sub-00003`:** **84.60% Dice** (Sensitivity: 91.7%, Precision: 78.5%)
- **`sub-00123`:** **79.41% Dice** (Sensitivity: 73.7%, Precision: 86.0%)
- **`sub-00073`:** **74.00% Dice** (*Synergistic boost:* FLAIR: 57.1%, T1: 72.2% &rarr; Fusion: **74.0%**)
- **`sub-00055`:** **71.46% Dice** (Sensitivity: 66.5%, Precision: 77.3%)
- **`sub-00050`:** **65.22% Dice** (*Synergistic boost:* FLAIR: 48.8%, T1: 59.9% &rarr; Fusion: **65.2%**)
- **`sub-00040`:** **53.10% Dice** (*Breakthrough:* FLAIR: 6.5%, T1: 0.0% &rarr; Fusion: **53.1%**)

---

## Pipeline Overview & Architecture

```
Raw Scans (data/raw/dataset/sub-XXXXX/anat/)
├── T1-weighted (moving)
├── FLAIR (reference)
└── FLAIR_roi (ground-truth lesion mask, FCD only)
             │
             ▼
   [ 1. Discovery & Validation ]
   • Paired modality inspection & geometry verification
   • BIDS/cohort inventory & manifest generation
             │
             ▼
   [ 2. HD-BET Skull Stripping ]
   • FLAIR skull stripping -> FLAIR_brainmask.nii.gz
   • T1w skull stripping -> T1_brainmask.nii.gz
             │
             ▼
   [ 3. Rigid Registration (T1 -> FLAIR Reference Grid) ]
   • SimpleITK Euler3D / Rigid registration with Mattes Mutual Information
   • Resampling moving T1 directly into FLAIR grid space
   • Preserving discrete integer labels (FLAIR ROI is untouched)
             │
             ▼
   [ 4. Intensity Normalization ]
   • Brain-masked foreground z-score normalization ((x - μ) / σ)
   • Intensity clipping between 0.5% and 99.5% percentiles
   • Background strictly preserved as zero
             │
             ▼
   [ 5. nnU-Net v2 Dataset Packaging & Atomic Delivery ]
   • Dataset101_FLAIR   (FLAIR-only)
   • Dataset102_T1      (Registered T1-only)
   • Dataset103_Fusion  (Channel 0: FLAIR, Channel 1: T1)
   • 142 Training cases (57 FCD + 85 HC) in imagesTr / labelsTr
   • 28 Held-out Test cases (28 FCD) in imagesTs / labelsTs
             │
             ▼
   [ 6. Model Training, Inference & Evaluation ]
   • Stratified 5-fold cross-validation with exact complement invariants
   • FCDnnUNetTrainer_100epochs (calibrated 100-epoch custom trainer)
   • Fold-aware validation evaluation (labelsTr) & held-out test evaluation (labelsTs)
   • Scientific HD95 reporting (finite-only mean + explicit infinite-case counts)
```

### Critical Geometric Reference Contract
- **FLAIR is the definitive spatial reference.** Ground-truth lesion masks (`*_FLAIR_roi.nii.gz`) are delineated natively on FLAIR space.
- T1w is rigidly registered and resampled into the FLAIR reference geometry so both modalities share identical dimensions, voxel spacing, origin, and coordinate axes.
- **The lesion ROI is NEVER transformed** by the T1-to-FLAIR registration matrix, completely eliminating spatial misalignment artifacts.
- Healthy Control (HC) subjects with no lesion masks are correctly ingested with verified zero-filled background masks.

---

## Dataset Summary & Cohorts

The cohort comprises **170 subjects** from OpenNeuro dataset `ds004199`:

| Cohort | Total Cases | Training / CV Set (`imagesTr`) | Held-out Test Set (`imagesTs`) | Lesion Label |
|---|---|---|---|---|
| **FCD (Focal Cortical Dysplasia)** | 85 | 57 | 28 | Delineated binary mask (1 = lesion, 0 = background) |
| **Healthy Controls (HC)** | 85 | 85 | 0 | Blank background mask (all zeros) |
| **Total** | **170** | **142** | **28** | — |

- **Training cohort**: 142 subjects (57 FCD + 85 HC), partitioned into 5 stratified cross-validation folds (each fold contains 113–114 train cases and 28–29 validation cases).
- **Test cohort**: 28 FCD subjects held out strictly for independent final evaluation; never exposed during cross-validation or training.

---

## Project Status & Implemented Features

| Component | Status | Verification & Capabilities |
|---|---|---|
| **Dataset Downloader** | **Complete** | Automated download and participant check of OpenNeuro `ds004199` |
| **BIDS Discovery & Pairing** | **Complete** | Validates paired T1w, FLAIR, and ROI masks; handles both FCD and HC cohorts |
| **HD-BET Skull Stripping** | **Complete** | GPU/CPU skull stripping with brain-mask geometry verification and auto-fallback |
| **T1-to-FLAIR Registration** | **Complete** | SimpleITK rigid registration with mutual information; resamples T1 to FLAIR grid |
| **Foreground Normalization** | **Complete** | Masked z-score normalization with percentile clipping ($0.5\% - 99.5\%$) |
| **nnU-Net v2 Dataset Assembly** | **Complete** | Generates `Dataset101_FLAIR`, `Dataset102_T1`, and `Dataset103_Fusion` |
| **Cross-Validation Splits** | **Complete** | Stratified 5-fold CV (`seed=42`) with duplicate rejection & complement invariants |
| **Preflight System Check** | **Complete** | Comprehensive environment, GPU/CUDA, dependency, and split verification |
| **Custom nnU-Net Trainer** | **Complete** | `FCDnnUNetTrainer_100epochs` with resolution check preventing collisions |
| **Model Training Runner** | **Complete** | Orchestration for single-fold, all-fold, and multi-dataset cross-validation |
| **Inference CLI** | **Complete** | Supports validation fold (`--split val --fold {0..4}`) and test set (`--split test`) |
| **Evaluation & HD95 Metrics** | **Complete** | Fold-aware validation; reports finite-only mean with explicit infinite counts |
| **Interactive HTML QC Reports** | **Complete** | 5 standalone HTML reports with embedded Base64 slice visuals & red ROI contours |
| **Automated Test Suite** | **Complete** | **243 passing unit & integration tests** (`pytest tests/`) |

---

## Repository Layout

```text
fcd-segmentation/
├── configs/
│   ├── preprocessing_config.json # Preprocessing pipeline settings (paths, registration, device)
│   ├── training.yaml             # Training workflow configuration (trainer, batch size, epochs)
│   └── evaluation.yaml           # Metric evaluation and thresholding parameters
├── data/
│   ├── raw/
│   │   ├── dataset/              # Raw BIDS scans (sub-*/anat/ scans; never modified)
│   │   └── participants.tsv      # Subject metadata with clinical group and split assignments
│   ├── processed/                # Preprocessed nnU-Net v2 datasets (nnUNet_raw)
│   │   ├── Dataset101_FLAIR/
│   │   ├── Dataset102_T1/
│   │   └── Dataset103_Fusion/
│   └── nnunet_preprocessed/      # nnU-Net preprocessed data & splits_final.json
├── docs/
│   └── training.md               # In-depth training and cluster deployment guide
├── outputs/
│   ├── predictions/              # Inferred segmentation masks
│   │   └── Dataset101_FLAIR/
│   │       ├── fold_0/           # Fold validation predictions
│   │       └── test/             # Held-out test predictions
│   ├── evaluation/               # Metrics CSV and JSON evaluation reports
│   └── preprocessing/
│       ├── qc/                   # Standalone HTML QC reports
│       └── logs/                 # Preprocessing run manifests and logs
├── scripts/
│   ├── download_raw_dataset.py   # Downloads raw OpenNeuro ds004199 dataset
│   ├── prepare_dataset.py        # Primary CLI entry point for preprocessing pipeline
│   ├── preflight_check.py        # Pre-training environment & dataset diagnostics
│   ├── train.py                  # Model training orchestrator (nnU-Net v2)
│   ├── predict.py                # Model inference CLI (val folds and test cohort)
│   └── evaluate.py               # Metric evaluation CLI (Dice, IoU, HD95, etc.)
├── src/fcd_segmentation/
│   ├── config.py                 # Pydantic schema validation for configs
│   ├── evaluation/
│   │   ├── evaluator.py          # Dataset evaluation and cohort boundary checker
│   │   └── metrics.py            # Multi-metric computation engine (finite/infinite HD95)
│   ├── inference/
│   │   └── predict.py            # nnU-Net model prediction backend
│   ├── preprocessing/
│   │   ├── discovery.py          # Subject and scan pairing discovery
│   │   ├── geometry.py           # Voxel grid, affine, and dimension validators
│   │   ├── skull_stripping.py    # HD-BET integration and mask validation
│   │   ├── registration.py       # SimpleITK registration & QC image generation
│   │   ├── resampling.py         # Label-safe interpolation and grid alignment
│   │   ├── normalization.py      # Foreground z-score normalization
│   │   ├── dataset.py            # nnU-Net v2 dataset builder & metadata writer
│   │   ├── qc_reports.py         # HTML QC reports with embedded Base64 visuals
│   │   └── pipeline.py           # Subject orchestrator with safe cleanup
│   └── training/
│       ├── cross_validation.py   # Stratified 5-fold cross-validation split engine
│       ├── dataset_validator.py  # Strict nnU-Net dataset integrity checker
│       ├── paths.py              # Environment variable and path resolver
│       ├── runner.py             # Subprocess execution and logging wrapper
│       ├── trainers.py           # FCDnnUNetTrainer_100epochs and trainer resolver
│       └── workflow.py           # Training workflow guard and coordinator
├── tests/                        # Comprehensive test suite (243 tests)
│   ├── preprocessing/            # Preprocessing, geometry, and QC report tests
│   ├── test_cross_validation.py  # Cross-validation integrity & leakage tests
│   ├── test_evaluator.py         # Evaluator and cohort validation tests
│   ├── test_metrics.py           # Metric accuracy and infinite HD95 tests
│   ├── test_trainer_discovery.py # Trainer resolution tests
│   └── test_training.py          # Training command and workflow guard tests
└── pyproject.toml                # Project packaging and dependencies
```

---

## Prerequisites

- **Operating System**: Linux (Ubuntu 22.04+ recommended) or Windows 10/11 (64-bit)
- **Python**: `>=3.11, <3.13` (Python 3.12 recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended for HD-BET skull stripping and nnU-Net 3D full-resolution training; CPU execution is supported for testing via `--device cpu`.
- **Disk Space**: ~150 GB free disk space recommended for raw scans, preprocessed volumes, and nnU-Net training plans.

---

## Installation Guide

### Linux Installation

```bash
# 1. Clone repository and navigate inside
git clone https://github.com/utpalgautam/fcd-segmentation.git
cd fcd-segmentation

# 2. Create virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package in editable mode with all dependencies
pip install --upgrade pip
pip install -e ".[dev,imaging]"

# 4. Install PyTorch with CUDA support (adjust cu121/cu124 according to your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install HD-BET for brain skull stripping
pip install hd-bet==2.0.1

# 6. Install openneuro-py for dataset downloading
pip install openneuro-py

# 7. Verify installation and run test suite
pytest tests/ -v
```

### Windows Installation

Open **PowerShell**:

```powershell
# 1. Clone repository and navigate inside
git clone https://github.com/utpalgautam/fcd-segmentation.git
Set-Location fcd-segmentation

# 2. Create virtual environment with Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install package in editable mode
python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"

# 4. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install HD-BET and openneuro-py
pip install hd-bet==2.0.1 openneuro-py

# 6. Verify installation and run test suite
pytest tests\ -v
```

---

## Step-by-Step Pipeline Execution

Follow these steps sequentially to run the entire pipeline from raw data acquisition to model evaluation.

### Step 1: Download Raw OpenNeuro Dataset

Download the public raw OpenNeuro BIDS dataset `ds004199` (170 subjects). No API key or account login is required:

```bash
# Download raw dataset into data/raw/dataset
python scripts/download_raw_dataset.py \
    --dataset ds004199 \
    --output data/raw/dataset
```

*Flags*:
- `--dataset`: OpenNeuro dataset accession (default: `ds004199`).
- `--output`: Destination directory (default: `data/raw/dataset`).
- `--skip-verification`: Skip verifying subject count against `participants.tsv`.

---

### Step 2: Run Preprocessing Pipeline

Run skull stripping, T1-to-FLAIR rigid registration, masked z-score normalization, and nnU-Net dataset packaging:

#### A. Quick Dry Run (Inventory Only)
Inspect discovered subjects, check scan pairs, and produce `input_inventory.html` without running any heavy processing:
```bash
python scripts/prepare_dataset.py --inventory-only
```

#### B. Smoke Test on a Subset (e.g. 2 subjects)
```bash
python scripts/prepare_dataset.py \
    --limit 2 \
    --reports outputs/preprocessing \
    --device cuda
```

#### C. Full Preprocessing Execution
```bash
python scripts/prepare_dataset.py \
    --input data/raw/dataset \
    --output data/processed \
    --reports outputs/preprocessing \
    --config configs/preprocessing_config.json \
    --device cuda \
    --resume
```

*Key Preprocessing CLI Flags*:
| Argument | Description | Default |
|---|---|---|
| `--input` | Path to raw cohort directory containing `sub-*/anat/` | `data/raw/dataset` |
| `--output` | Destination directory for processed nnU-Net v2 datasets | `data/processed` |
| `--reports` | Destination directory for HTML QC reports and logs | `outputs/preprocessing` |
| `--config` | Path to JSON pipeline configuration file | `configs/preprocessing_config.json` |
| `--device` | Compute device for HD-BET skull stripping (`cuda` or `cpu`) | `cuda` |
| `--resume` | Skip subjects with already verified complete outputs | `False` |
| `--overwrite` | Force re-processing of all subjects from scratch | `False` |
| `--limit N` | Process only first `N` subjects (for debugging) | `None` (all) |

---

### Step 3: Preflight System & Environment Check

Before launching training on a GPU workstation or cluster, verify environment readiness:

```bash
python scripts/preflight_check.py --full --device cuda
```

The preflight check inspects:
- Python version, PyTorch version, and CUDA acceleration.
- nnU-Net v2 CLI commands (`nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`, `nnUNetv2_predict`).
- Dynamic discovery of custom trainer `FCDnnUNetTrainer_100epochs`.
- Integrity of `Dataset101_FLAIR`, `Dataset102_T1`, and `Dataset103_Fusion`.
- Presence and consistency of cross-validation splits.

---

### Step 4: Generate Stratified 5-Fold Cross-Validation Splits

Generate leak-free, deterministic 5-fold cross-validation splits stratified by clinical group (`fcd` vs. `hc`), with random seed `42`:

```bash
# Generate splits for all three datasets simultaneously:
python scripts/train.py --split-folds --dataset all

# Alternatively, generate explicitly for each dataset directory:
python -m fcd_segmentation.training.cross_validation \
    --participants data/raw/participants.tsv \
    --output-dir data/nnunet_preprocessed/Dataset101_FLAIR/splits_final.json

python -m fcd_segmentation.training.cross_validation \
    --participants data/raw/participants.tsv \
    --output-dir data/nnunet_preprocessed/Dataset102_T1/splits_final.json

python -m fcd_segmentation.training.cross_validation \
    --participants data/raw/participants.tsv \
    --output-dir data/nnunet_preprocessed/Dataset103_Fusion/splits_final.json
```

**Cross-Validation Guarantees**:
- Exactly 142 training subjects (57 FCD, 85 HC).
- All 28 test FCD subjects are strictly excluded (zero data leakage).
- For every fold: `train = expected_train - fold_val` with zero overlaps and no duplicates.

---

### Step 5: nnU-Net Experiment Planning & Preprocessing

Run dataset integrity verification, fingerprint extraction, and planning:

```bash
# Using project runner:
python scripts/train.py --prepare --dataset 101
python scripts/train.py --prepare --dataset 102
python scripts/train.py --prepare --dataset 103

# Or directly with nnU-Net v2 CLI:
nnUNetv2_plan_and_preprocess -d 101 102 103 -c 3d_fullres --verify_dataset_integrity
```

---

### Step 6: Train Models

Train models using the calibrated 100-epoch custom trainer (`FCDnnUNetTrainer_100epochs`):

```bash
# Train Fold 0 of FLAIR model (Dataset 101)
python scripts/train.py --config configs/training.yaml --dataset 101 --fold 0 --device cuda

# Train Fold 0 of Multimodal Fusion model (Dataset 103)
python scripts/train.py --config configs/training.yaml --dataset 103 --fold 0 --device cuda

# Train all 5 folds sequentially for a dataset
python scripts/train.py --config configs/training.yaml --dataset 101 --all-folds --device cuda

# Resume an interrupted training run from latest checkpoint
python scripts/train.py --config configs/training.yaml --dataset 101 --fold 0 --continue --device cuda
```

*Key Training CLI Flags*:
| Argument | Description | Default |
|---|---|---|
| `--config` | Path to YAML training configuration file | `configs/training.yaml` |
| `--dataset` | Target dataset: `101`, `102`, `103`, or `all` | `101` |
| `--fold` | Target fold index: `0`, `1`, `2`, `3`, `4`, or `all` | `0` |
| `--all-folds` | Train all 5 folds sequentially | `False` |
| `--trainer` | Custom trainer class name | `FCDnnUNetTrainer_100epochs` |
| `--configuration`, `-c` | nnU-Net model configuration (e.g. `3d_fullres`) | Config default |
| `--continue` | Resume training from latest checkpoint | `False` |
| `--device` | Compute device (`cuda`, `cpu`) | `cuda` |

---

### Step 7: Model Inference / Predictions

Generate segmentation masks on either cross-validation folds or the independent held-out test cohort:

```bash
# 1. Predict on validation cohort of Fold 0 (outputs to outputs/predictions/Dataset101_FLAIR/fold_0)
python scripts/predict.py \
    --dataset 101 \
    --fold 0 \
    --split val \
    --device cuda

# 2. Predict on held-out test cohort (outputs to outputs/predictions/Dataset101_FLAIR/test)
python scripts/predict.py \
    --dataset 101 \
    --split test \
    --device cuda

# 3. Predict with ensembling across all 5 folds on the test set
python scripts/predict.py \
    --dataset 101 \
    --fold all \
    --split test \
    --device cuda
```

*Inference Output Paths*:
- Validation Fold: `outputs/predictions/{dataset_name}/fold_{fold}/`
- Held-Out Test Set: `outputs/predictions/{dataset_name}/test/`

---

### Step 8: Metric Evaluation & Scientific Reporting

Evaluate predicted segmentations against ground truth labels:

```bash
# 1. Evaluate Fold 0 validation predictions (against labelsTr)
python scripts/evaluate.py \
    --dataset 101 \
    --split val \
    --fold 0

# 2. Evaluate held-out test predictions (against labelsTs, 28 FCD cases)
python scripts/evaluate.py \
    --dataset 101 \
    --split test

# 3. Multi-modality benchmark comparison (compares FLAIR, T1, and Fusion on test set)
python scripts/evaluate.py \
    --dataset all \
    --split test
```

#### Metrics Reported:
- **Dice Similarity Coefficient (DSC)** and **Intersection over Union (IoU / Jaccard)**
- **Sensitivity (Recall)**, **Specificity**, and **Precision (PPV)**
- **95% Hausdorff Distance (HD95)**:
  - `finite_only_mean`: Average distance across successful segmentations.
  - `finite_count` / `infinite_count`: Number of cases with finite vs. failed segmentations.
  - `infinite_percentage`: Exact failure rate percentage (reported transparently; never silently dropped).

Output files are stored under `outputs/evaluation/{dataset_name}/`:
- `evaluation_summary_{split}.json`
- `evaluation_per_subject_{split}.csv`
- `modality_comparison_{split}.json` (when `--dataset all` is evaluated)

---

### Step 9: Automated Post-Training Pipeline & Visual Reporting

After training completes on one or more datasets, run the consolidated post-training automation script:

```bash
python scripts/run_post_training_pipeline.py
```

This single command automatically executes:
1. **Validation Prediction Completeness Check:** Confirms all 29 validation cases have predictions in `outputs/predictions/` (generating any missing subjects on GPU).
2. **Ground Truth Staging:** Stages compressed validation labels in `outputs/evaluation_data/`.
3. **Formal Metric Evaluation:** Computes Dice, Sensitivity, Specificity, Precision, and HD95 stratified by clinical cohort (FCD vs Healthy Controls) using `scripts/evaluate.py`.
4. **HTML Visual Segmentation Reports:** Renders 4-panel slice-by-slice overlays (MRI, Ground Truth, Prediction, and Green/Red Overlay) at the slice of maximum lesion volume into interactive standalone HTML files:
   - `outputs/segmentation_report_Dataset101_FLAIR_fold0.html` (4.7 MB)
   - `outputs/segmentation_report_Dataset102_T1_fold0.html` (3.9 MB)
   - `outputs/segmentation_report_Dataset103_Fusion_fold0.html` (4.8 MB)
5. **Cross-Modality Benchmark Summary:** Prints the consolidated comparative table in terminal.

---

## Documentation Index

Detailed technical guides and benchmark records are maintained in the [`docs/`](docs/) directory:

| Document | Description |
| :--- | :--- |
| **[Benchmark Outcomes Report](docs/benchmark_report.md)** | Full scientific report of Fold 0 validation outcomes across FLAIR, T1, and Multimodal Fusion. |
| **[Installation & Server Setup](docs/installation.md)** | Step-by-step setup on Linux workstations, Windows PowerShell, and Docker GPU servers (NVIDIA H200 / A100). |
| **[Dataset Setup & Preprocessing](docs/dataset-setup.md)** | Zenodo dataset acquisition, cohort contracts, HD-BET skull stripping, and rigid registration. |
| **[Training & Workflow Guide](docs/training.md)** | nnU-Net v2 model training, custom 100-epoch trainers, 5-fold cross-validation, and inference. |
| **[Experiment Protocol](docs/experiments.md)** | Experimental tracks, statistical reporting standards, and metric formulas. |
| **[Troubleshooting Guide](docs/troubleshooting.md)** | Solutions for common GPU VRAM, compilation, CUDA, and environment errors. |
| **[Architecture & Modules](docs/architecture.md)** | Source code package layout and module design principles. |

---

## Quality Control (QC) Reports

The preprocessing pipeline automatically generates 5 self-contained, interactive HTML reports with embedded Base64 imagery in `outputs/preprocessing/qc/`:

1. **`input_inventory.html`**:
   - Total subjects discovered, FCD vs. Healthy Control counts.
   - Scan pairing completeness (T1w, FLAIR, ROI) and exclusion flags.
2. **`skull_stripping_FLAIR.html`**:
   - Axial, coronal, and sagittal orthogonal views.
   - Brain mask outline (cyan) and lesion ROI contour (red) overlay.
   - Foreground volume and Dice metrics.
3. **`skull_stripping_T1.html`**:
   - T1 native space skull-stripping views and brain mask verification.
4. **`registration_T1_to_FLAIR.html`**:
   - Side-by-side Fixed FLAIR and Resampled T1 views.
   - **Color Overlay**: FLAIR displayed in Green, T1 in Magenta, and ground-truth lesion ROI in **Red contour**.
   - **Checkerboard View**: Alternate 4x4x4 grid blocks to visually assess anatomical contour continuity across sulci and ventricles.
5. **`final_dataset_qc.html`**:
   - Comprehensive audit of the 3 nnU-Net datasets (`Dataset101_FLAIR`, `Dataset102_T1`, `Dataset103_Fusion`).
   - Verifies channel geometry matching, target voxel grids, lesion voxel counts, and intensity distributions.

---

## Testing & Validation

The codebase includes an extensive automated test suite covering unit tests, integration tests, geometry verification, and cross-validation integrity:

```bash
# Run the complete test suite (243 tests)
pytest tests/ -v

# Run only training and evaluation tests
pytest tests/test_cross_validation.py tests/test_evaluator.py tests/test_training.py -v

# Run only preprocessing and registration tests
pytest tests/preprocessing/ -v

# Run code style and lint checks
ruff check .
```

---

## Troubleshooting & Common Issues

### 1. `openneuro-py: command not found`
Install `openneuro-py` in your active virtual environment:
```bash
pip install openneuro-py
```

### 2. `hd-bet: command not found`
Ensure `hd-bet` is installed in your active virtual environment:
```bash
pip install hd-bet==2.0.1
which hd-bet          # Linux
Get-Command hd-bet    # Windows PowerShell
```

### 3. CUDA Out of Memory during Skull Stripping
If GPU VRAM is constrained (< 4 GB), run skull stripping with `--device cpu`:
```bash
python scripts/prepare_dataset.py --device cpu --resume
```

### 4. Custom Trainer Resolution Error
If nnU-Net reports that it cannot find `FCDnnUNetTrainer_100epochs`, ensure the project root is in `PYTHONPATH` or that the runner is used:
```bash
python scripts/preflight_check.py --full
```
The runner automatically sets `nnUNet_extTrainer=fcd_segmentation.training.trainers` in the environment.

### 5. Resuming Interrupted Processing or Training
- **Preprocessing**: Pass `--resume` to `prepare_dataset.py` to skip already completed subjects.
- **Training**: Pass `--continue` to `train.py` to resume model training from the latest checkpoint (`checkpoint_latest.pth`).

---

## Dataset Privacy & Handling

> [!CAUTION]
> **CRITICAL REMINDER**:
> Never commit patient imaging files, NIfTI volumes (`*.nii`, `*.nii.gz`), segmentation labels, clinical tables, or personal health identifiers to Git. The `.gitignore` is pre-configured to exclude `data/raw/`, `data/processed/`, `data/nnunet_preprocessed/`, and `outputs/`. Always verify using `git status` prior to committing.
