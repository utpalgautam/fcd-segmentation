# FCD Segmentation

Cross-platform research codebase for focal cortical dysplasia (FCD) MRI segmentation.

> **Project status:** This repository is under active development. The environment checks, FLAIR file inventory, preprocessing utilities, and tests are available, but the end-to-end preprocessing pipeline, model training, prediction, and final evaluation workflows are not yet complete. Do not treat placeholder scripts as implemented functionality.

## Contents

- [Project status](#project-status)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation on Linux](#installation-on-linux)
- [Installation on Windows](#installation-on-windows)
- [Dataset setup and privacy](#dataset-setup-and-privacy)
- [Configuration](#configuration)
- [Commands you can run now](#commands-you-can-run-now)
- [Preprocessing utilities](#preprocessing-utilities)
- [Testing and linting](#testing-and-linting)
- [Troubleshooting](#troubleshooting)
- [Development workflow](#development-workflow)

## Project status

| Component | Status | Notes |
|---|---|---|
| Python package and configuration | Available | Python `>=3.12,<3.13` |
| Environment check | Available | Reports Python, platform, optional packages, and PyTorch CUDA status |
| FLAIR dataset inventory | Available | Writes a CSV of discovered candidate FLAIR files |
| Image I/O, geometry validation, resampling, normalization | Utilities and tests available | These are preprocessing building blocks, not yet one complete pipeline command |
| HD-BET skull stripping | Utility available | Requires the external `hd-bet` executable and compatible dependencies; see below |
| Dataset preparation pipeline | Placeholder | `scripts/prepare_dataset.py` intentionally exits with “Not implemented yet” |
| Model training and prediction | Not yet complete | Do not assume `scripts/train.py` or `scripts/predict.py` is a complete workflow |
| Final evaluation workflow | Not yet complete | Verify each script before relying on it for experiments |

Update this table as components become implemented and validated.

## Repository layout

```text
fcd-segmentation/
├── configs/
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── models/
├── data/
│   ├── raw/
│   │   └── dataset/        # Private input dataset; not committed
│   └── processed/           # Derived data; not committed
├── docs/                    # Supporting design and team notes
├── outputs/                 # Local reports, QC, and experiment outputs
├── scripts/
│   ├── audit_dataset.py
│   ├── check_environment.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── src/fcd_segmentation/
│   ├── cli.py
│   └── preprocessing/
├── tests/
└── pyproject.toml
```

The exact dataset structure depends on the approved dataset. The example tree above shows the repository's intended local data locations, not a required subject naming scheme.

## Prerequisites

- Git
- Python **3.12.x** (the package requires `>=3.12,<3.13`)
- Internet access for installing Python dependencies
- Sufficient local storage for MRI files and derived outputs
- For HD-BET skull stripping: the `hd-bet` command and its compatible dependencies
- For future GPU model training: compatible NVIDIA hardware/driver and a suitable PyTorch build (GPU is not required for ordinary utility tests)

Do not copy a virtual environment between Linux and Windows. Create one separately on each operating system.

## Installation on Linux

Commands below use Ubuntu and assume the repository directory is named `fcd-segmentation`.

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv python3-pip
```

If Python 3.12 is not available from your distribution's configured repositories, install Python 3.12 using a trusted method for your OS.

### 2. Clone the repository

Replace the placeholder with the team's actual Git URL:

```bash
git clone <YOUR_REPOSITORY_URL>
cd fcd-segmentation
```

If you already have the repository, simply enter it and pull the team's current branch:

```bash
cd ~/fcd-segmentation
git pull
```

Use the branch your team has agreed to use.

### 3. Create and activate the environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python --version
which python
```

The version should be Python 3.12.x and the executable path should point inside this repository's `.venv`.

### 4. Install the package and development/imaging dependencies

```bash
python -m pip install -e ".[dev,imaging]"
```

This installs the project in editable mode plus its development and imaging extras. PyTorch is a separate optional extra; install a suitable build only if you need it. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a compatible CPU or CUDA build.

### 5. Verify the environment

```bash
fcd-check-environment
python -m pytest -q
ruff check .
```

The test suite may skip integration tests that require the private MRI dataset if that data is not present. Read the pytest summary to distinguish skips from failures.

## Installation on Windows

Use **Windows PowerShell** and Python 3.12.x. Install Git for Windows and Python 3.12 first; ensure the Python launcher `py` is available.

### 1. Clone the repository

Replace the placeholder with the team's actual Git URL:

```powershell
git clone <YOUR_REPOSITORY_URL>
Set-Location fcd-segmentation
```

For an existing checkout, enter the repository and update the agreed branch:

```powershell
Set-Location path\to\fcd-segmentation
git pull
```

### 2. Create and activate the environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python --version
python -m pip --version
```

The active interpreter and pip should point inside this repository's `.venv`.

If PowerShell blocks virtual-environment activation, use **Command Prompt**:

```bat
.venv\Scripts\activate.bat
```

Avoid changing machine-wide PowerShell execution policy just to activate the environment.

### 3. Install the package and development/imaging dependencies

```powershell
python -m pip install -e ".[dev,imaging]"
```

For optional PyTorch support, select the appropriate CPU or CUDA build using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). GPU availability depends on hardware, drivers, and the selected PyTorch package.

### 4. Verify the environment

```powershell
fcd-check-environment
python -m pytest -q
ruff check .
```

If `ruff` is not found, verify that installation completed in the activated `.venv`; then try `python -m pip show ruff` and reinstall the documented development extra if needed.

## Dataset setup and privacy

### Privacy and access

**Never commit original MRI scans, segmentation masks, participant IDs, clinical records, or derived medical volumes to GitHub.** Follow the dataset's access, consent, and redistribution requirements. `.gitignore` is only a safeguard: always inspect `git status` before staging files.

### Local locations

The configuration currently sets:

```yaml
# configs/data.yaml
data_root: data/raw/dataset
processed_root: data/processed
output_root: outputs
```

Keep original data under `data/raw/` and derived data under `data/processed/`. Reports and quality-control outputs can be stored under `outputs/`. Do not overwrite raw scans.

Put the approved dataset in `data/raw/dataset/` unless you pass another path to a command that explicitly supports it. The actual dataset's directory layout and filename conventions must be checked against its documentation; do not rename files or assume every subject has every modality.

The configuration currently lists these filename patterns:

```yaml
filename_rules:
  flair_glob: "*_FLAIR.nii*"
  t1_glob: "*_T1w.nii*"
  roi_glob: "*_FLAIR_roi.nii*"
```

These are project configuration values; they do not by themselves guarantee every script uses all patterns or validates every modality. The currently available audit command inventories candidate FLAIR files only.

### Recommended dataset checks

Before using real data, record the dataset name/version and approved storage location, image and mask naming rules, modality availability, mask label meanings, spatial metadata, and any excluded cases. Check for missing/corrupted scans and confirm that images and masks are aligned. Split participants—not individual slices or patches—between training, validation, and test sets to avoid leakage.

## Configuration

Important configuration files:

| File | Purpose |
|---|---|
| `configs/data.yaml` | Data, processed-data, output paths and filename patterns |
| `configs/preprocessing.yaml` | Provisional preprocessing settings |
| `configs/models/flair.yaml` | FLAIR model configuration |
| `configs/models/t1.yaml` | T1 model configuration |
| `configs/models/fusion.yaml` | Fusion model configuration |
| `configs/training.yaml` | Training settings |
| `configs/evaluation.yaml` | Evaluation settings |

The current `configs/preprocessing.yaml` contains provisional values:

```yaml
reference_modality: flair
target_spacing_mm: null
orientation: null
registration:
  enabled: false
  backend: null
normalization:
  method: foreground_zscore
  clip_percentiles: [0.5, 99.5]
skull_stripping:
  enabled: false
  backend: null
resampling:
  image_interpolation: linear
  mask_interpolation: nearest
```

These values are not an approved experimental protocol. In particular, registration and skull stripping are disabled in this YAML by default, and target spacing/orientation are unset. The standalone skull-stripping utility described below is separate from an end-to-end pipeline. Confirm protocol choices with the project team before reporting experiments.

## Commands you can run now

Run these commands from the repository root with the virtual environment activated.

### Environment check

```bash
fcd-check-environment
```

This reports the Python version/executable, operating system, installed versions of NumPy, PyYAML, nibabel, and PyTorch when installed, plus CUDA availability if PyTorch is available.

Equivalent direct invocation:

```bash
python scripts/check_environment.py
```

### Dataset audit / FLAIR inventory

First inspect supported arguments:

```bash
fcd-audit-dataset --help
```

Inventory candidate FLAIR images and write a CSV:

```bash
fcd-audit-dataset --data-root data/raw/dataset --output-csv outputs/flair_inventory.csv
```

Equivalent direct invocation:

```bash
python scripts/audit_dataset.py --data-root data/raw/dataset --output-csv outputs/flair_inventory.csv
```

The command prints the number of discovered FLAIR files and writes a CSV containing `subject_id` and `flair_path`. The current implementation uses the `--data-root` and `--output-csv` arguments. It is an inventory, **not** a full dataset validator; it does not prove masks are present/correct, modalities are paired, or images are spatially aligned.

If using a different dataset directory, pass its path explicitly:

```bash
fcd-audit-dataset --data-root /path/to/dataset --output-csv outputs/flair_inventory.csv
```

On Windows PowerShell, for example:

```powershell
fcd-audit-dataset --data-root "D:\MRI\data-set" --output-csv outputs\flair_inventory.csv
```

### Dataset preparation pipeline: not yet available

The following script is currently a placeholder:

```bash
python scripts/prepare_dataset.py --help
```

Calling it with arguments such as `--data-root` and `--config` currently ends with an explicit “Not implemented yet” error. Do not use it as if it preprocesses the dataset end to end.

## Preprocessing utilities

The following building blocks and tests are present in the source tree:

- NIfTI image I/O
- Image geometry validation
- Resampling utilities
- Foreground-based intensity normalization
- HD-BET skull stripping and quality-control generation

These are not yet exposed as one complete, configured preprocessing command. Use only functions whose behavior and arguments have been verified in the source. Do not assume that changing `configs/preprocessing.yaml` automatically runs these utilities.

### HD-BET skull stripping

The skull-stripping implementation calls an external `hd-bet` executable. It is not installed by the project's default `imaging` extra, so install it separately in the active environment only when you intend to run this component. The implementation's error message recommends `hd-bet==2.0.1`; verify the package's current installation instructions and compatibility before installing it.

Check whether the command is available:

Linux:
```bash
command -v hd-bet
```

Windows PowerShell:
```powershell
Get-Command hd-bet
```

If it is missing, the skull-stripping utility cannot run. HD-BET and its PyTorch dependencies can be platform-sensitive; confirm that the selected installation supports the operating system and hardware before promising cross-platform execution.

The Python utility provides `run_hd_bet`, `process_subject`, `discover_flair_files`, `run_dataset_hd_bet`, and QC helpers. Their arguments and behavior should be read from `src/fcd_segmentation/preprocessing/skull_stripping.py`; there is not currently a documented standalone CLI command for this module. Do not invent a command such as `python .../skull_stripping.py` and expect it to work.

When invoked through the implemented functions, the code is designed to save subject-level skull-stripped images and brain masks, create per-subject QC images, and generate a QC index/summary. The repository currently contains sample/output artifacts under `outputs/skull_stripping/`; inspect these locally rather than assuming that an existing output directory means a new run has completed successfully.

## Testing and linting

Run from the project root with `.venv` activated:

```bash
python -m pytest -q
ruff check .
```

Run a focused test file when debugging:

```bash
python -m pytest tests/preprocessing/test_geometry.py -v
python -m pytest tests/preprocessing/test_io.py -v
python -m pytest tests/preprocessing/test_resampling.py -v
python -m pytest tests/preprocessing/test_normalization.py -v
python -m pytest tests/preprocessing/test_skull_stripping.py -v
```

Some skull-stripping integration tests may require real MRI data and an installed HD-BET executable. They should be skipped when required external resources are unavailable, not reported as successful execution of the real-data pipeline. Read the test summary and traceback.

## Troubleshooting

### Wrong Python version or interpreter

Linux:
```bash
python --version
which python
python -m pip --version
```

Windows PowerShell:
```powershell
python --version
Get-Command python
python -m pip --version
```

Activate the repository's `.venv`. Use `python -m pip` so installation targets the active interpreter.

### Dataset audit finds zero files

```bash
fcd-audit-dataset --help
```

Check the supplied root path, actual filename extensions, and case-sensitive naming rules. The configured FLAIR pattern is `*_FLAIR.nii*`, but the audit relies on the actual discovery implementation. Do not rename files until the dataset's naming rules are confirmed.

### `hd-bet` is not found

Verify the active environment and check `command -v hd-bet` (Linux) or `Get-Command hd-bet` (PowerShell). Install the external dependency according to its compatible installation instructions. The default project extras do not install it.

### `torch.cuda.is_available()` is false

GPU use requires compatible NVIDIA hardware, drivers, and a compatible PyTorch build. Check:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

These checks also work in a Windows terminal when the NVIDIA driver provides `nvidia-smi`. For a CPU-only environment, CUDA being unavailable is not an error for ordinary utility tests.

### Image/mask shapes or geometry differ

Equal array dimensions do not guarantee spatial alignment. Inspect dimensions, affine matrices, orientation, and voxel spacing. Do not fix mismatches by reshaping arrays. Registration and resampling must be deliberate, tested, and logged. Use nearest-neighbor interpolation for discrete masks.

### Tests fail

Run the failing test alone first, inspect the traceback, and decide whether the cause is code, test expectations, missing dependencies, or unavailable data. Do not change a test merely to hide a real failure.

### Generated files appear in Git

```bash
git status --short
git check-ignore -v data/raw/example.nii.gz
```

Do not stage MRI scans, `.venv/`, secrets, or generated model artifacts. If a generated file was already tracked, `.gitignore` alone will not untrack it; inspect the Git history and status before changing tracking.

## Development workflow

1. Pull the team's agreed branch.
2. Activate the local `.venv`.
3. Make a focused code change and add/update tests.
4. Run the relevant tests, then `python -m pytest -q` and `ruff check .`.
5. Review `git diff --check`, `git diff`, and `git status --short`.
6. Ensure no private data, credentials, or large generated outputs are staged.
7. Commit and push the code and documentation.

Example:

```bash
git diff --check
git status --short
git add README.md
git commit -m "Consolidate cross-platform project documentation"
git push
```

On Windows, the same Git commands work in PowerShell when Git is installed.

## Related implementation notes

Supporting documents currently in `docs/` may contain additional architecture, experiment, onboarding, and development notes. This README is the main cross-platform getting-started guide; supporting notes should not contradict the actual code or claim unfinished workflows are operational.
