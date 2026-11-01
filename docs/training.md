# Model Training Guide — nnU-Net v2 Workflow

This guide details how to validate, prepare, and train nnU-Net v2 models for FCD lesion segmentation on Windows PowerShell and remote GPU environments.

---

## 1. Overview & Datasets

The project supports three standardized nnU-Net v2 dataset variants located in `data/processed`:

| Dataset ID | Folder Name | Channels | Description |
|---|---|---|---|
| **101** | `Dataset101_FLAIR` | `0`: FLAIR | FLAIR-only baseline |
| **102** | `Dataset102_T1` | `0`: T1 | T1-only registered to FLAIR grid |
| **103** | `Dataset103_Fusion` | `0`: FLAIR, `1`: T1 | Multimodal paired fusion |

---

## 2. Environment & Effective Paths

nnU-Net v2 requires three environment variables pointing to absolute paths:
- `nnUNet_raw`: Raw dataset folders containing `imagesTr`, `labelsTr`, and `dataset.json`.
- `nnUNet_preprocessed`: Target folder for extracted fingerprints, plans, and resampled training data.
- `nnUNet_results`: Output directory for model checkpoints, logs, and validation summaries.

### Repository-Relative Defaults

The training CLI automatically resolves portable defaults relative to the repository root:
- `nnUNet_raw` &rarr; `<project-root>\data\processed`
- `nnUNet_preprocessed` &rarr; `<project-root>\data\nnunet_preprocessed`
- `nnUNet_results` &rarr; `<project-root>\outputs\nnunet_results`

You do not need to configure machine-specific paths or modify your system variables.

### Setting Custom Paths on a Professor's GPU Machine

If running on an external server or cluster with separate storage disks:

```powershell
# PowerShell (session-local):
$env:nnUNet_raw = "D:\Datasets\fcd-segmentation\data\processed"
$env:nnUNet_preprocessed = "D:\nnunet_preprocessed"
$env:nnUNet_results = "D:\nnunet_results"
```

Or pass them directly to `scripts/train.py`:

```powershell
python .\scripts\train.py --check --raw-dir "D:\Datasets\data\processed" --preprocessed-dir "D:\nnunet_preprocessed" --results-dir "D:\nnunet_results"
```

---

## 3. Environment & Dataset Health Check

Run `--check` to verify your Python environment, CUDA availability, installed packages, path resolution, and dataset validity:

```powershell
# Activate your virtual environment first
.\.venv\Scripts\Activate.ps1

# Check environment and default dataset (101)
python .\scripts\train.py --check

# Check all three datasets
python .\scripts\train.py --check --dataset all
```

This command will:
1. Print Python version, PyTorch version, CUDA GPU device name, and nnU-Net v2 package version.
2. Confirm the effective environment variables passed to child processes.
3. Validate `dataset.json`, channel assignments, image/label pairings, non-empty files, and label discrete integers.
4. Check whether the cohort passes the 5-fold cross-validation guard.
5. Save a structured JSON summary to `outputs/training/check_<timestamp>_summary.json`.

---

## 4. Cohort Size Guard & Smoke-Testing Note

> [!IMPORTANT]
> The current dataset was preprocessed with `--limit 1` as a smoke test. It contains only one subject (`sub-00001`).
>
> **The training workflow enforces a hard cohort size guard:**
> At least **5 valid labeled cases** are strictly required before 5-fold cross-validation training (`nnUNetv2_train`) can execute.
> If fewer than 5 cases exist, training will be safely blocked, and no training process will start.
>
> To prepare for actual training, first execute the full preprocessing pipeline on the complete cohort:
> ```powershell
> python .\scripts\prepare_dataset.py --overwrite
> ```

---

## 5. Experiment Preparation (`nnUNetv2_plan_and_preprocess`)

Once the full dataset is preprocessed and validated, run `--prepare` to extract dataset fingerprints, determine target spacings and patch sizes, and preprocess data:

```powershell
# Prepare Dataset 101 (FLAIR)
python .\scripts\train.py --prepare --dataset 101

# Prepare all three datasets (101, 102, 103)
python .\scripts\train.py --prepare --dataset all

# Force re-running planning and clean existing fingerprints
python .\scripts\train.py --prepare --dataset all --clean
```

---

## 6. Model Training Commands

### Train Single Fold

Train Fold 0 of a specific dataset with `3d_fullres` (default):

```powershell
# Train FLAIR-only model on fold 0
python .\scripts\train.py --dataset 101 --fold 0

# Train T1-only model on fold 0
python .\scripts\train.py --dataset 102 --fold 0

# Train Multimodal Fusion model on fold 0
python .\scripts\train.py --dataset 103 --fold 0
```

### 5-Fold Cross-Validation

Train all 5 folds (`0, 1, 2, 3, 4`) sequentially to avoid GPU memory exhaustion:

```powershell
# 5-fold CV for FLAIR
python .\scripts\train.py --dataset 101 --all-folds

# 5-fold CV for Fusion
python .\scripts\train.py --dataset 103 --all-folds
```

### Full Benchmark: All Datasets, All Folds

To train all three representations across all 5 folds (15 fold runs total):

```powershell
python .\scripts\train.py --dataset all --all-folds
```

Use `--keep-going` if you wish the pipeline to log any fold error and proceed to the next experiment rather than stopping immediately.

### Resuming Interrupted Training

If a training job was stopped or interrupted, resume from the latest checkpoint using `--continue` (which passes `--c` to `nnUNetv2_train`):

```powershell
python .\scripts\train.py --dataset 101 --fold 0 --continue
```

---

## 7. Artifact Locations & Logs

All training operations produce structured, audit-ready logs:

- **CLI Run Logs**: `outputs/training/train_<timestamp>.log`
- **Machine-Readable Summaries**: `outputs/training/train_<timestamp>_summary.json`
- **nnU-Net Preprocessed Plans**: `data/nnunet_preprocessed/Dataset<ID>_<Name>/`
- **Model Checkpoints & Weights**: `outputs/nnunet_results/Dataset<ID>_<Name>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_<fold>/`
  - `checkpoint_latest.pth`
  - `checkpoint_best.pth`
  - `progress.png`
  - `training_log.txt`

