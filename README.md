# FCD MRI Segmentation: Complete Preprocessing Pipeline & nnU-Net Datasets

Cross-platform research pipeline for focal cortical dysplasia (FCD) lesion segmentation using multi-modal structural MRI (T1-weighted and FLAIR).

This repository provides an automated, auditable, and fully validated end-to-end preprocessing pipeline that converts raw BIDS/cohort MRI scans into three standardized **nnU-Net v2** datasets:
1. **`Dataset101_FLAIR`** — Skull-stripped, normalized FLAIR images (1 channel: `_0000.nii.gz`).
2. **`Dataset102_T1`** — Skull-stripped, rigid-registered, and resampled T1w images mapped onto the FLAIR voxel grid (1 channel: `_0000.nii.gz`).
3. **`Dataset103_Fusion`** — Multi-channel paired inputs on the exact same anatomical grid (`channel 0000` = FLAIR, `channel 0001` = T1).

---

## Table of Contents

- [Pipeline Overview & Architecture](#pipeline-overview--architecture)
- [Project Status](#project-status)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
  - [Linux (Ubuntu / Debian)](#linux-installation)
  - [Windows (PowerShell / Command Prompt)](#windows-installation)
- [Running the Preprocessing Pipeline](#running-the-preprocessing-pipeline)
  - [Quick Dry Run (Inventory Only)](#1-quick-dry-run-inventory-only)
  - [Smoke Test on a Subset](#2-smoke-test-on-a-subset)
  - [Full Pipeline Execution (Linux)](#3-full-pipeline-execution-linux)
  - [Full Pipeline Execution (Windows)](#4-full-pipeline-execution-windows)
  - [CLI Flags & Options Reference](#cli-flags--options-reference)
- [Model Training Workflow (nnU-Net v2)](#model-training-workflow-nnu-net-v2)
  - [Environment & Dataset Check](#1-environment--dataset-check)
  - [Experiment Preparation & Fingerprinting](#2-experiment-preparation--fingerprinting)
  - [Single-Fold & 5-Fold Training](#3-single-fold--5-fold-training)
  - [Resuming Training](#4-resuming-training)
- [Quality Control (QC) Reports](#quality-control-qc-reports)
- [Testing & Validation](#testing--validation)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)
- [Dataset Privacy & Handling](#dataset-privacy--handling)

---

## Pipeline Overview & Architecture

```
Raw Scans (data/raw/dataset/sub-XXXXX/anat/)
├── T1-weighted (moving)
├── FLAIR (reference)
└── FLAIR_roi (ground truth lesion mask)
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
   [ 3. Rigid Registration (T1 -> FLAIR Grid) ]
   • SimpleITK Euler3D / Rigid registration
   • Mutual information metric optimization
   • Resampling moving T1 directly into FLAIR grid space
   • Preservation of discrete integer labels
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
   • Dataset103_Fusion  (Channel 0: FLAclearIR, Channel 1: T1)
   • Shared subject splits (subject_splits.json) across all 3 datasets
             │
             ▼
   [ 6. Self-Contained HTML QC Reports ]
   • input_inventory.html
   • skull_stripping_FLAIR.html
   • skull_stripping_T1.html
   • registration_T1_to_FLAIR.html (with red ROI contour overlay)
   • final_dataset_qc.html
```

### Critical Geometric Reference Contract
- **FLAIR is the definitive reference space.** Ground-truth lesion masks (`*_FLAIR_roi.nii.gz`) are delineated on FLAIR space.
- T1w is registered to FLAIR so that both modalities share identical dimensions, voxel spacing, origin, and coordinate axes.
- Healthy Control (HC) subjects with no lesion masks are correctly discovered and included in training splits with zero-filled background masks.

---

## Project Status

| Component | Status | Verification / Description |
|---|---|---|
| **Discovery & Inventory** | **Complete** | Discovers paired T1w, FLAIR, and ROI masks; handles both FCD and HC cohorts |
| **HD-BET Skull Stripping** | **Complete** | Processes FLAIR and T1 with brain-mask geometry validation and auto-fallback |
| **T1-to-FLAIR Registration** | **Complete** | SimpleITK rigid registration with mutual information; resamples T1 to FLAIR grid |
| **Foreground Normalization** | **Complete** | Robust masked z-score normalization with percentile clipping ($0.5\% - 99.5\%$) |
| **nnU-Net v2 Dataset Assembly** | **Complete** | Generates `Dataset101_FLAIR`, `Dataset102_T1`, and `Dataset103_Fusion` |
| **nnU-Net v2 Model Training** | **Complete** | Environment checks, validation, cohort guards, preparation, and 5-fold CV orchestration |
| **Interactive HTML QC Reports** | **Complete** | 5 standalone HTML reports with embedded Base64 slice visuals & red ROI contours |
| **Automated Test Suite** | **Complete** | **161 unit & integration tests passing** (`pytest tests/`) |

---

## Repository Layout

```text
fcd-segmentation/
├── configs/
│   ├── preprocessing_config.json # Pipeline configuration (paths, registration, device)
│   ├── data.yaml                 # Legacy path rules
│   └── models/                   # Model experiment configurations
├── data/
│   ├── raw/                      # Raw dataset root (sub-*/anat/ scans; never modified)
│   └── processed/                # Output directory for nnU-Net v2 datasets
│       ├── Dataset101_FLAIR/
│       ├── Dataset102_T1/
│       ├── Dataset103_Fusion/
│       └── subject_splits.json   # Synchronized 5-fold cross-validation splits
├── outputs/
│   └── preprocessing/
│       ├── qc/                   # Self-contained HTML QC reports
│       └── logs/                 # Pipeline manifests and run logs
├── scripts/
│   ├── prepare_dataset.py        # Primary CLI entry point for preprocessing
│   ├── check_environment.py     # Environment and CUDA diagnostic tool
│   ├── audit_dataset.py          # Quick inventory of input files
│   ├── train.py                  # Model training orchestrator
│   ├── predict.py                # Model inference script
│   └── evaluate.py               # Evaluation and metrics computation
├── src/fcd_segmentation/
│   ├── cli.py                    # Package command-line endpoints
│   └── preprocessing/
│       ├── discovery.py          # Subject and scan pairing discovery
│       ├── skull_stripping.py    # HD-BET integration and mask validation
│       ├── registration.py       # SimpleITK registration & QC image generation
│       ├── resampling.py         # Label-safe interpolation and grid alignment
│       ├── normalization.py      # Foreground z-score normalization
│       ├── dataset.py            # nnU-Net v2 dataset builder & metadata writer
│       ├── qc_reports.py         # HTML QC reports with embedded Base64 visuals
│       └── pipeline.py           # Subject orchestrator with safe cleanup
├── tests/                        # Comprehensive test suite (161 tests)
│   ├── preprocessing/            # Unit & integration tests for preprocessing modules
│   └── conftest.py
└── pyproject.toml                # Project packaging and dependencies
```

---

## Prerequisites

- **Python**: `>=3.12, <3.13`
- **Operating System**: Linux (Ubuntu 22.04+ recommended) or Windows 10/11 (64-bit)
- **CUDA / GPU**: NVIDIA GPU with 4GB+ VRAM recommended for fast HD-BET skull stripping; CPU execution is fully supported via `--device cpu`.
- **HD-BET**: External brain extraction tool (`hd-bet==2.0.1`).

---

## Installation Guide

### Linux Installation

```bash
# 1. Clone repository and navigate inside
git clone <YOUR_REPOSITORY_URL>
cd fcd-segmentation

# 2. Create virtual environment using Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package with imaging & dev dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"

# 4. Install PyTorch with CUDA (adjust cuda version to match system driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Install HD-BET for skull stripping
pip install hd-bet==2.0.1

# 6. Verify environment and test suite
fcd-check-environment
pytest tests/
```

### Windows Installation

Open **PowerShell** as standard user:

```powershell
# 1. Clone repository and navigate inside
git clone <YOUR_REPOSITORY_URL>
Set-Location fcd-segmentation

# 2. Create virtual environment with Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install package
python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"

# 4. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Install HD-BET
pip install hd-bet==2.0.1

# 6. Verify environment and test suite
fcd-check-environment
pytest tests/
```

*(If using Windows Command Prompt `cmd.exe`, activate using `.venv\Scripts\activate.bat` instead).*

---

## Running the Preprocessing Pipeline

All commands should be executed from the repository root with the virtual environment activated.

### 1. Quick Dry Run (Inventory Only)
Inspect discovered subjects, check scan pairs, and produce `input_inventory.html` without running any heavy processing:

**Linux:**
```bash
python scripts/prepare_dataset.py --inventory-only
```

**Windows (PowerShell):**
```powershell
python scripts\prepare_dataset.py --inventory-only
```

---

### 2. Smoke Test on a Subset
Run the complete pipeline on just 2 subjects to verify HD-BET, registration, and dataset output without waiting for the full cohort:

**Linux:**
```bash
python scripts/prepare_dataset.py --limit 2 --reports outputs/preprocessing
```

**Windows (PowerShell):**
```powershell
python scripts\prepare_dataset.py --limit 2 --reports outputs\preprocessing
```

---

### 3. Full Pipeline Execution (Linux)

Run the end-to-end preprocessing pipeline on the entire cohort using GPU acceleration and automatic resume for already completed subjects:

```bash
# Standard complete run with CUDA acceleration and resume capability
python scripts/prepare_dataset.py \
    --input data/raw/dataset \
    --output data/processed \
    --reports outputs/preprocessing \
    --config configs/preprocessing_config.json \
    --device cuda \
    --resume

# CPU-only execution (if NVIDIA GPU / CUDA is not available):
python scripts/prepare_dataset.py \
    --input data/raw/dataset \
    --output data/processed \
    --reports outputs/preprocessing \
    --device cpu \
    --resume

# Force re-processing of all subjects from scratch:
python scripts/prepare_dataset.py \
    --input data/raw/dataset \
    --output data/processed \
    --reports outputs/preprocessing \
    --overwrite
```

---

### 4. Full Pipeline Execution (Windows)

**PowerShell:**
```powershell
# Standard complete run with CUDA acceleration and resume capability
python scripts\prepare_dataset.py `
    --input data\raw\dataset `
    --output data\processed `
    --reports outputs\preprocessing `
    --config configs\preprocessing_config.json `
    --device cuda `
    --resume

# CPU-only execution:
python scripts\prepare_dataset.py `
    --input data\raw\dataset `
    --output data\processed `
    --reports outputs\preprocessing `
    --device cpu `
    --resume

# Force re-processing of all subjects from scratch:
python scripts\prepare_dataset.py `
    --input data\raw\dataset `
    --output data\processed `
    --reports outputs\preprocessing `
    --overwrite
```

**Command Prompt (`cmd.exe`):**
```cmd
python scripts\prepare_dataset.py --input data\raw\dataset --output data\processed --reports outputs\preprocessing --config configs\preprocessing_config.json --device cuda --resume
```

---

### CLI Flags & Options Reference

| Argument | Description | Default |
|---|---|---|
| `--input`, `--data-root` | Path to raw cohort directory containing `sub-*/anat/` | `data/raw/dataset` |
| `--output` | Destination directory for processed nnU-Net v2 datasets | `data/processed` |
| `--reports` | Destination directory for HTML QC reports, manifests, and logs | `outputs/preprocessing` |
| `--config` | Path to JSON pipeline configuration file | `configs/preprocessing_config.json` |
| `--inventory-only` | Run cohort discovery and input report without processing MRI data | `False` |
| `--limit N` | Process only first `N` subjects (ideal for debugging/testing) | `None` (all) |
| `--resume` | Skip subjects that already have verified outputs matching inputs | `False` |
| `--overwrite` | Force re-run and overwrite all existing outputs and QC reports | `False` |
| `--device {cuda,cpu}` | Device selector for HD-BET skull stripping | Config default (`cuda`) |
| `--fail-fast` | Immediately halt pipeline on first error instead of logging & continuing | `False` |
| `--keep-failed-intermediates` | Preserve temporary directory when a subject fails (for debugging) | `False` |

---

## Model Training Workflow (nnU-Net v2)

The model training workflow provides end-to-end orchestration for `Dataset101_FLAIR`, `Dataset102_T1`, and `Dataset103_Fusion` models using nnU-Net v2. Detailed documentation is available in [docs/training.md](docs/training.md).

> [!IMPORTANT]
> The current dataset was preprocessed with `--limit 1` for smoke testing. The training workflow enforces a **hard cohort size guard** (minimum 5 valid cases) and safely blocks 5-fold cross-validation until the full dataset is preprocessed.

### 1. Environment & Dataset Check

Validate Python, PyTorch, CUDA, nnU-Net package status, path resolution, and dataset metadata without running training:

```powershell
python .\scripts\train.py --check
python .\scripts\train.py --check --dataset all
```

### 2. Experiment Preparation & Fingerprinting

Extract dataset fingerprints and run nnU-Net experiment planning after completing full preprocessing:

```powershell
python .\scripts\train.py --prepare --dataset 101
python .\scripts\train.py --prepare --dataset all
```

### 3. Single-Fold & 5-Fold Training

Train models sequentially on available GPU hardware. By default, training runs with the calibrated **100-epoch trainer** (`nnUNetTrainer_100epochs` configured in `configs/training.yaml`):

```powershell
# Train Fold 0 of FLAIR model (runs for 100 epochs by default)
python .\scripts\train.py --dataset 101 --fold 0

# Train 5-fold cross validation on FLAIR model
python .\scripts\train.py --dataset 101 --all-folds

# Train 5-fold cross validation across all three datasets (15 runs total)
python .\scripts\train.py --dataset all --all-folds

# Optional: override trainer if desired
python .\scripts\train.py --dataset 101 --fold 0 --trainer nnUNetTrainer
```

### 4. Resuming Training

Resume interrupted training from the latest checkpoint (`--c` flag):

```powershell
python .\scripts\train.py --dataset 101 --fold 0 --continue
```

---

## Quality Control (QC) Reports

The pipeline produces 5 self-contained, interactive HTML reports with embedded Base64 imagery:

1. **`input_inventory.html`**:
   - Total subjects discovered, FCD vs. Healthy Control counts.
   - Scan pairing completeness (T1w, FLAIR, ROI) and exclusion flags.
2. **`skull_stripping_FLAIR.html`**:
   - Axial, coronal, and sagittal orthogonal views.
   - Brain mask outline (cyan) and lesion ROI contour (red) overlay.
   - Dice overlap coefficient and brain volume metrics.
3. **`skull_stripping_T1.html`**:
   - T1 native space skull-stripping views and brain mask verification.
4. **`registration_T1_to_FLAIR.html`**:
   - Side-by-side Fixed FLAIR and Resampled T1 views.
   - **Color Overlay**: FLAIR displayed in Green, T1 in Magenta, and ground-truth lesion ROI in **Red contour**.
   - **Checkerboard View**: Alternate 4x4x4 grid blocks to visually assess anatomical contour continuity across sulci and ventricles.
   - Quantitative foreground correlation metrics.
5. **`final_dataset_qc.html`**:
   - Comprehensive audit of the 3 nnU-Net datasets (`Dataset101_FLAIR`, `Dataset102_T1`, `Dataset103_Fusion`).
   - Verifies channel geometry matching, target voxel grids, lesion voxel counts, and intensity distributions.

---

## Testing & Validation

The codebase includes comprehensive unit, integration, and geometry tests. Run them at any time:

```bash
# Run all tests (161 tests)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run only preprocessing unit & integration tests
pytest tests/preprocessing/ -v

# Run code style and lint checks
ruff check .
```

---

## Troubleshooting & Common Issues

### 1. `hd-bet: command not found`
Ensure `hd-bet` is installed in your active virtual environment:
```bash
pip install hd-bet==2.0.1
# Verify:
which hd-bet          # Linux
Get-Command hd-bet    # Windows PowerShell
```

### 2. CUDA Out of Memory during Skull Stripping
If your GPU VRAM is constrained (< 4 GB), run with `--device cpu`:
```bash
python scripts/prepare_dataset.py --device cpu --resume
```

### 3. Missing ROI on Healthy Control (HC) subjects
Healthy control subjects do not have FCD lesions. The pipeline automatically assigns them a verified blank mask (all zeros) in `Dataset101_FLAIR`, `Dataset102_T1`, and `Dataset103_Fusion`, and labels their QC overlap as `N/A (HC)`.

### 4. Resuming Interrupted Processing
If a run is stopped mid-way, simply re-run with `--resume`. The orchestrator inspects the generated output files and logs, skipping completed subjects without redundant computation.

---

## Dataset Privacy & Handling

> **CRITICAL REMINDER**:
> Never commit patient imaging files, NIfTI volumes (`*.nii`, `*.nii.gz`), segmentation labels, clinical tables, or personal health identifiers to Git. The `.gitignore` is pre-configured to exclude `data/raw/`, `data/processed/`, and `outputs/`. Always verify using `git status` prior to committing.
