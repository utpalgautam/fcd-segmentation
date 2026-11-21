# Model Training & Evaluation Guide — nnU-Net v2 Workflow

This guide details how to validate, prepare, train, and evaluate nnU-Net v2 models for FCD lesion segmentation on Windows PowerShell, Linux workstations, and remote laboratory GPU clusters.

---

## 1. Overview & Experimental Design

The project establishes a rigorous benchmark comparing three standardized representations in `data/processed`:

| Dataset ID | Folder Name | Modality | Channels | Training Cohort | Held-out Test Cohort |
|---|---|---|---|---|---|
| **101** | `Dataset101_FLAIR` | FLAIR-only | `0`: FLAIR | 142 cases (57 FCD + 85 HC) | 28 cases (All FCD) |
| **102** | `Dataset102_T1` | T1-only | `0`: T1 (rigid-registered to FLAIR) | 142 cases (57 FCD + 85 HC) | 28 cases (All FCD) |
| **103** | `Dataset103_Fusion` | Multimodal Paired | `0`: FLAIR, `1`: registered T1 | 142 cases (57 FCD + 85 HC) | 28 cases (All FCD) |

### Key Experimental Guarantees
1. **Group-Stratified 5-Fold Cross-Validation**:
   - Training cohort (142 cases: 57 FCD, 85 HC) is partitioned into 5 folds preserving the FCD:HC ratio across each fold (~11-12 FCD, 17 HC per validation fold).
   - Random seed is fixed (`seed = 42`).
   - Identical `splits_final.json` files are deployed to all three datasets, ensuring strict subject-level correspondence across models.
2. **Strict Test Cohort Isolation**:
   - 28 FCD subjects designated in `participants.tsv` (`split == 'test'`) are strictly isolated in `imagesTs` / `labelsTs`.
   - Test cases are never used during fingerprint extraction, plan generation, training, or cross-validation.
3. **Custom 100-Epoch Trainer**:
   - `nnUNetTrainer_100epochs` / `FCDnnUNetTrainer_100epochs` sets `num_epochs = 100` and adjusts polynomial learning rate decay and checkpointing (`save_every = 25`).

---

## 2. Remote Lab Preflight Check

Before starting training or preprocessing on the remote lab workstation or cluster, run the automated diagnostic script:

```bash
# On Linux / macOS
.venv/bin/python scripts/preflight_check.py

# On Windows PowerShell
.\.venv\Scripts\python.exe scripts\preflight_check.py
```

This diagnostic script validates:
1. Python version (3.11+) and installed packages (`torch`, `nnunetv2`, `nibabel`, `scipy`, `scikit-learn`, `pandas`, `pydantic`, `PyYAML`).
2. CUDA GPU device name, count, and VRAM.
3. nnU-Net v2 CLI tools (`nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`, `nnUNetv2_predict`).
4. Custom trainer discovery via `nnUNet_extTrainer`.
5. Free storage space on the outputs disk volume (recommends ≥ 30 GB).
6. Cohort integrity and presence of 142 training and 28 test cases.

---

## 3. Environment & Effective Paths

nnU-Net v2 requires three primary directories configured via environment variables:
- `nnUNet_raw`: Directory containing `Dataset101_FLAIR`, `Dataset102_T1`, `Dataset103_Fusion`.
- `nnUNet_preprocessed`: Target for extracted fingerprints, plans, resampled arrays, and `splits_final.json`.
- `nnUNet_results`: Output directory for checkpoints, progress plots, and validation logs.

### Portable Defaults
The repository resolves relative defaults automatically without needing system-wide environment modifications:
- `nnUNet_raw` &rarr; `<project-root>/data/processed`
- `nnUNet_preprocessed` &rarr; `<project-root>/data/nnunet_preprocessed`
- `nnUNet_results` &rarr; `<project-root>/outputs/nnunet_results`

### Custom Mounts on Cluster Storage
If your lab machine uses high-speed NVMe or scratch disks:

```bash
# Linux Bash:
export nnUNet_raw="/scratch/fcd-segmentation/data/processed"
export nnUNet_preprocessed="/scratch/fcd-segmentation/data/nnunet_preprocessed"
export nnUNet_results="/scratch/fcd-segmentation/outputs/nnunet_results"

# Windows PowerShell:
$env:nnUNet_raw = "D:\Datasets\fcd-segmentation\data\processed"
$env:nnUNet_preprocessed = "D:\nnunet_preprocessed"
$env:nnUNet_results = "D:\nnunet_results"
```

Or pass flags directly to `scripts/train.py`:
`--raw-dir`, `--preprocessed-dir`, `--results-dir`.

---

## 4. Dataset Validation & Cross-Validation Splitting

### Step 4.1: Dataset Integrity & Cohort Check
Verify dataset structure, channel order, NIfTI headers, and discrete labels without running training:

```bash
# Check all three datasets
python scripts/train.py --check --dataset all

# Strict validation check
python scripts/train.py --validate --dataset all
```

### Step 4.2: Export Stratified Cross-Validation Splits
Generate reproducible group-stratified 5-fold splits and export `splits_final.json` to `nnUNet_preprocessed` for all three datasets:

```bash
python scripts/train.py --split-folds --dataset all
```

---

## 5. Experiment Preparation (`nnUNetv2_plan_and_preprocess`)

Run planning and preprocessing on the GPU machine:

```bash
# Prepare single dataset (FLAIR)
python scripts/train.py --prepare --dataset 101

# Prepare all three datasets (101, 102, 103)
python scripts/train.py --prepare --dataset all

# Clean existing fingerprints and re-plan
python scripts/train.py --prepare --dataset all --clean
```

---

## 6. Model Training Commands

### Train Single Fold
Train Fold 0 with default `3d_fullres` and 100-epoch trainer:

```bash
# Train FLAIR model on Fold 0
python scripts/train.py --dataset 101 --fold 0

# Train T1 model on Fold 0
python scripts/train.py --dataset 102 --fold 0

# Train Multimodal Fusion model on Fold 0
python scripts/train.py --dataset 103 --fold 0
```

### 5-Fold Cross-Validation
Train all 5 folds sequentially:

```bash
# 5-fold CV for FLAIR
python scripts/train.py --dataset 101 --all-folds

# 5-fold CV for Fusion
python scripts/train.py --dataset 103 --all-folds
```

### Benchmark All Datasets Across All Folds (15 Fold Runs)
```bash
python scripts/train.py --dataset all --all-folds --keep-going
```

### Resuming Interrupted Training
Resume training from the latest checkpoint (`checkpoint_latest.pth`):

```bash
python scripts/train.py --dataset 101 --fold 0 --continue
```

---

## 7. Model Inference & Evaluation

### Step 7.1: Predict on Held-Out Test Set (28 Subjects)
Generate segmentations for test cases in `imagesTs`:

```bash
# Inference with FLAIR model (Fold 0)
python scripts/predict.py --dataset 101 --fold 0

# Inference with 5-fold ensemble (all folds)
python scripts/predict.py --dataset 101 --fold all

# Inference for Multimodal Fusion model
python scripts/predict.py --dataset 103 --fold all
```

### Step 7.2: Evaluate Predictions & Generate Reports
Compute Dice similarity, Sensitivity/Recall, Specificity, Precision, and HD95:

```bash
# Evaluate test split predictions for FLAIR
python scripts/evaluate.py --dataset 101 --split test

# Evaluate and compare all three modalities (FLAIR vs T1 vs Fusion)
python scripts/evaluate.py --compare-all --split test
```

### Step 7.3: Artifact Outputs
- **Per-Subject Metrics CSV**: `outputs/evaluation/metrics_Dataset101_FLAIR_test.csv`
- **Summary Statistics JSON**: `outputs/evaluation/summary_Dataset101_FLAIR_test.json`
- **Cross-Modality Benchmark JSON**: `outputs/evaluation/modality_comparison_test.json`
- **Training Logs**: `outputs/training/train_<timestamp>.log`
- **Model Checkpoints**: `outputs/nnunet_results/Dataset<ID>_<Name>/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_<fold>/`
  - `checkpoint_latest.pth`
  - `checkpoint_best.pth`
  - `checkpoint_final.pth`
  - `progress.png`
