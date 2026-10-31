# FCD Segmentation

A cross-platform research codebase for developing and evaluating focal cortical dysplasia (FCD) segmentation methods from structural brain MRI.

> **Project status: early implementation scaffold — not a complete or clinically validated segmentation system.**
>
> The repository currently provides project structure, configuration utilities, dataset-discovery/validation utilities, selected preprocessing helpers, evaluation metrics, environment checks, tests, and CI. The FLAIR/T1/fusion model builders and training loop are placeholders. Registration, skull stripping, full preprocessing, end-to-end training, and inference still require implementation and validation. No MRI dataset, trained weights, or clinical-use claims are included.

## Contents

- [Project goals](#project-goals)
- [Current implementation status](#current-implementation-status)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Install on Linux](#install-on-linux)
- [Install on Windows](#install-on-windows)
- [Install PyTorch and enable the GPU](#install-pytorch-and-enable-the-gpu)
- [Verify the installation](#verify-the-installation)
- [Dataset setup and privacy](#dataset-setup-and-privacy)
- [Available commands](#available-commands)
- [Configuration](#configuration)
- [Development workflow for a team](#development-workflow-for-a-team)
- [Testing and continuous integration](#testing-and-continuous-integration)
- [Reproducibility and experiment reporting](#reproducibility-and-experiment-reporting)
- [Known limitations and planned work](#known-limitations-and-planned-work)
- [Troubleshooting](#troubleshooting)
- [Documentation index](#documentation-index)
- [License and data rights](#license-and-data-rights)

## Project goals

The team's research goal is to develop and evaluate reproducible FCD segmentation methods using structural MRI. The codebase is organized to support separate experiments for:

1. FLAIR-only input.
2. T1-weighted-only input.
3. Multimodal FLAIR + T1 input/fusion.

These are intended experimental tracks, not yet completed or validated model implementations. The exact preprocessing choices, label convention, model architecture, split protocol, losses, and metrics must be defined against the project report and dataset before experiments are considered comparable.

## Current implementation status

| Component | Status |
|---|---|
| Installable Python package and configuration loader | Basic implementation |
| Environment diagnostic | Implemented |
| Dataset file discovery | Basic implementation; current audit is limited |
| Dataset validation | Basic implementation; verify checks against dataset needs |
| MRI I/O, geometry, normalization helpers | Partial utilities; not a full pipeline |
| Resampling | Utility exists; end-to-end policy still to be defined |
| Registration | Placeholder/incomplete |
| Skull stripping | Placeholder/incomplete |
| FLAIR model | Placeholder (`NotImplementedError`) |
| T1 model | Placeholder (`NotImplementedError`) |
| Fusion model | Placeholder (`NotImplementedError`) |
| Training loop | Placeholder (`NotImplementedError`) |
| Inference | Incomplete; not an end-to-end predictor yet |
| Evaluation metrics | Basic metric utilities with unit tests |
| Automated tests | Current test suite passes in the development environment |
| Linux/Windows CI | Workflow scaffold present; verify the latest GitHub Actions run after pushing |

Passing the existing tests confirms only the tested behavior. It does **not** mean the end-to-end FCD segmentation system is ready.

## Repository structure

The following tree describes the intended/current top-level layout. Some modules are scaffolds or placeholders; the status table above is authoritative about implementation completeness.

```text
fcd-segmentation/
├── .github/
│   └── workflows/
│       └── tests.yml              # CI tests on supported operating systems
├── configs/
│   ├── data.yaml                  # Dataset and data discovery settings
│   ├── evaluation.yaml            # Evaluation configuration
│   ├── preprocessing.yaml         # Preprocessing settings
│   ├── training.yaml              # Training settings
│   ├── experiments/
│   │   └── README.md              # Experiment configuration guidance
│   └── models/
│       ├── flair.yaml             # FLAIR model settings
│       ├── t1.yaml                # T1 model settings
│       └── fusion.yaml            # Multimodal fusion settings
├── data/
│   ├── raw/
│   │   └── .gitkeep               # Local raw data directory; data is not committed
│   └── processed/
│       └── .gitkeep               # Local derived data directory; data is not committed
├── docs/
│   ├── architecture.md            # Module boundaries and data flow
│   ├── dataset-setup.md           # Local dataset organization and audit
│   ├── development.md             # Developer workflow and coding practices
│   ├── experiments.md             # Experiment design and reporting
│   ├── installation-linux.md      # Linux setup
│   ├── installation-windows.md   # Windows setup
│   └── troubleshooting.md         # Common environment and workflow issues
├── outputs/
│   └── .gitkeep                   # Local logs, predictions, checkpoints and reports
├── scripts/
│   ├── audit_dataset.py           # Dataset inventory/audit entry point
│   ├── check_environment.py       # Python/dependency/GPU diagnostic
│   ├── evaluate.py                # Evaluation entry point (workflow incomplete)
│   ├── predict.py                 # Prediction entry point (workflow incomplete)
│   ├── prepare_dataset.py          # Dataset preparation entry point (workflow incomplete)
│   └── train.py                   # Training entry point (workflow incomplete)
├── src/
│   └── fcd_segmentation/
│       ├── __init__.py
│       ├── cli.py                 # CLI helpers
│       ├── config.py              # Configuration loading/validation helpers
│       ├── runtime.py             # Optional PyTorch device selection
│       ├── data/
│       │   ├── __init__.py
│       │   ├── discovery.py       # Candidate file discovery
│       │   └── validation.py      # Dataset checks
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── io.py              # Image I/O helpers
│       │   ├── geometry.py        # Geometry/affine helpers
│       │   ├── normalization.py   # Intensity normalization helpers
│       │   ├── resampling.py      # Resampling helpers
│       │   ├── registration.py    # Registration placeholder/incomplete
│       │   └── skull_stripping.py # Skull-stripping placeholder/incomplete
│       ├── models/
│       │   ├── __init__.py
│       │   ├── flair_model.py     # FLAIR model builder placeholder
│       │   ├── t1_model.py        # T1 model builder placeholder
│       │   └── fusion_model.py    # Fusion model builder placeholder
│       ├── training/
│       │   ├── __init__.py
│       │   ├── reproducibility.py # Reproducibility helpers
│       │   └── trainer.py         # Training placeholder
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predict.py         # Inference utilities; workflow incomplete
│       └── evaluation/
│           ├── __init__.py
│           └── metrics.py         # Metric utilities
├── tests/
│   ├── conftest.py                 # Shared synthetic fixtures
│   ├── test_config.py              # Configuration tests
│   ├── test_discovery.py           # Dataset discovery tests
│   ├── test_metrics.py             # Evaluation metric tests
│   ├── test_runtime.py             # Runtime/environment tests
│   └── preprocessing/
│       ├── test_geometry.py        # Image geometry and affine tests
│       ├── test_io.py              # Medical image I/O tests
│       ├── test_normalization.py   # Intensity normalization tests
│       ├── test_resampling.py      # Resampling tests
│       └── test_skull_stripping.py # Skull-stripping and QC tests
├── .env.example                   # Example local environment variables; no secrets
├── .gitattributes                 # Consistent text-file handling
├── .gitignore                     # Excludes local data, outputs and generated files
├── CONTRIBUTING.md                # Contribution and branch/PR workflow
├── LICENSE                        # Currently states no license has been selected
├── pyproject.toml                 # Package metadata and dependencies
├── README.md
└── SECURITY.md                    # Security/privacy reporting guidance
```

Files such as `.venv/`, `__pycache__/`, `*.egg-info/`, raw/processed medical images, checkpoints, and local outputs should not be committed. The `.gitkeep` files only preserve empty folders in Git; they are not data.

## Requirements

### General requirements

- Git.
- Python 3.12 recommended for the current project setup.
- Internet access for installing Python dependencies.
- Storage for the dataset and processed volumes.
- A CPU is sufficient for utility tests and early development.
- An NVIDIA GPU with a compatible driver is recommended for deep-learning experiments, but CUDA is optional for CPU workflows.

### GPU expectations

The current developer environment has been tested with Python 3.12.15, PyTorch 2.11.0+cu128, CUDA runtime 12.8, and an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB VRAM. This is a record of one environment, **not a universal dependency pin or minimum GPU specification**. Other team members should install the PyTorch build appropriate for their operating system and hardware.

A 4 GB GPU will impose significant memory limits for 3D MRI. Plan to explore patch-based training, small batches, and mixed precision where appropriate. Do not assume a model fits in GPU memory until tested.

## Quick start

Replace `<YOUR_REPOSITORY_URL>` with the actual GitHub clone URL after the repository has been created.

### Linux

```bash
git clone <YOUR_REPOSITORY_URL>
cd fcd-segmentation

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"

# Install PyTorch separately; see the GPU section below.
python scripts/check_environment.py
python -m pytest
```

### Windows PowerShell

```powershell
git clone <YOUR_REPOSITORY_URL>
Set-Location fcd-segmentation

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"

# Install PyTorch separately; see the GPU section below.
python scripts/check_environment.py
python -m pytest
```

The first setup creates a local virtual environment. Each team member should create their own environment; do not commit or copy `.venv/` between machines.

## Install on Linux

For Ubuntu or another Linux distribution with Python 3.12 available:

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv python3-pip

git clone <YOUR_REPOSITORY_URL>
cd fcd-segmentation

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"
```

If `python3.12` is not available through your distribution's package manager, install Python 3.12 using a trusted method appropriate for that distribution, then repeat the virtual-environment commands.

### Deactivate the environment

```bash
deactivate
```

### Reopen the project later

```bash
cd fcd-segmentation
source .venv/bin/activate
```

## Install on Windows

Use PowerShell and install Git and Python 3.12 first. During Python setup, enable the Python launcher if offered.

```powershell
git clone <YOUR_REPOSITORY_URL>
Set-Location fcd-segmentation

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"
```

If PowerShell blocks activation due to the current execution policy, do not change the machine-wide policy blindly. One alternative is to use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Then run the installation commands from that activated environment.

### Reopen the project later

```powershell
Set-Location path\to\fcd-segmentation
.\.venv\Scripts\Activate.ps1
```

### CPU-only or GPU workstation

A CUDA-enabled PyTorch build requires a compatible NVIDIA GPU and driver. On a machine without compatible NVIDIA hardware, install the CPU build using the official PyTorch installation selector. Avoid copying a Linux virtual environment to Windows or vice versa.

## Install PyTorch and enable the GPU

PyTorch is installed separately from the base project dependencies so each developer can choose a suitable build.

1. Open the official PyTorch installation selector: https://pytorch.org/get-started/locally/
2. Select the appropriate operating system, package manager (Pip), language (Python), and compute platform.
3. Run the generated command inside the activated `.venv`.
4. Do not choose a CUDA wheel merely because `nvidia-smi` displays a CUDA version. The driver reports its supported CUDA capability; PyTorch wheels bundle their own runtime libraries. Follow the current official compatibility instructions.

For example, the following command installed successfully in the current Linux development environment at the time this documentation was written:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

This is an example for the recorded environment, not a command that every developer must use. The official selector may offer different versions over time.

Verify installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

Test a GPU operation:

```bash
python -c "import torch; x = torch.randn(1000, 1000, device='cuda'); print('Tensor device:', x.device); print('Result shape:', (x @ x).shape)"
```

If `torch.cuda.is_available()` is `False`, the base CPU tests can still run. Consult `docs/troubleshooting.md`.

## Verify the installation

Run all of these from the repository root with `.venv` activated:

```bash
python --version
python -m pip --version
python scripts/check_environment.py
python -m pytest
```

The current development environment's full test run completed with **142 tests passing**. The exact count can change as tests are added or removed. A successful run should finish with all collected tests passing.

Useful commands from the repository root, with `.venv` activated:

```bash
# Run the complete test suite
python -m pytest -q

# Run only preprocessing tests
python -m pytest -q tests/preprocessing/

# Run tests for an individual preprocessing component
python -m pytest -q tests/preprocessing/test_geometry.py
python -m pytest -q tests/preprocessing/test_io.py
python -m pytest -q tests/preprocessing/test_normalization.py
python -m pytest -q tests/preprocessing/test_resampling.py
python -m pytest -q tests/preprocessing/test_skull_stripping.py

# Check for whitespace errors in pending Git changes
git diff --check
```

### Test organization

- `tests/conftest.py` contains shared synthetic fixtures, including sample image data, affine transforms, images, masks, and a helper for writing temporary NIfTI files.
- `tests/test_config.py` covers configuration behavior.
- `tests/test_discovery.py` covers dataset discovery.
- `tests/test_metrics.py` covers evaluation metrics.
- `tests/test_runtime.py` covers runtime/environment behavior.
- `tests/preprocessing/test_geometry.py` covers image geometry and affine behavior.
- `tests/preprocessing/test_io.py` covers medical image input/output.
- `tests/preprocessing/test_normalization.py` covers intensity normalization, including foreground/background handling and invalid inputs.
- `tests/preprocessing/test_resampling.py` covers resampling behavior.
- `tests/preprocessing/test_skull_stripping.py` covers skull-stripping utilities and quality-control report generation.

The shared fixtures use synthetic data; private MRI study data is not required to run the test suite. The passing tests verify only the behavior they exercise and do not establish end-to-end segmentation performance or clinical validity.

Do not hide or skip a failing test simply to make CI green. Investigate the cause and add a regression test when appropriate.

## Dataset setup and privacy

**Do not commit MRI scans, segmentation masks, clinical metadata, participant identifiers, or other sensitive information.** Confirm that the team has permission to access and process the data and that the dataset's terms permit the intended research.

The repository reserves these local directories:

```text
data/
├── raw/        # Original dataset, local only
└── processed/  # Derived/preprocessed volumes, local only
outputs/        # Logs, checkpoints, predictions, metrics and figures
```

These folders are ignored by Git except for `.gitkeep` files. Do not weaken `.gitignore` to upload medical data.

The currently documented audit command is:

```bash
python scripts/audit_dataset.py --data-root data/raw/dataset
```

The audit is currently limited to candidate FLAIR file discovery/inventory. It is not a complete BIDS validator, label validator, spatial-alignment check, or data-quality certification. Read `docs/dataset-setup.md` before placing data or interpreting audit results.

Dataset-specific file naming, modality availability, ground-truth mask format, label values, orientation conventions, and split protocol must be agreed by the team and recorded before training is implemented.

## Available commands

| Command | Purpose | Current limitation |
|---|---|---|
| `python scripts/check_environment.py` | Report Python, dependencies, and optional CUDA status | Diagnostic only |
| `python -m pytest` | Run automated tests | Covers implemented utilities, not end-to-end model performance |
| `python scripts/audit_dataset.py --data-root PATH` | Inventory candidate FLAIR files | Limited audit; not full validation |
| `python scripts/prepare_dataset.py` | Dataset-preparation entry point | Workflow incomplete; inspect `--help` and implementation before use |
| `python scripts/train.py` | Training entry point | Training loop/model builders are placeholders |
| `python scripts/predict.py` | Prediction entry point | End-to-end inference incomplete |
| `python scripts/evaluate.py` | Evaluation entry point | End-to-end evaluation workflow incomplete |

Before using a script beyond the commands explicitly shown, inspect its supported options:

```bash
python scripts/audit_dataset.py --help
python scripts/prepare_dataset.py --help
python scripts/train.py --help
python scripts/predict.py --help
python scripts/evaluate.py --help
```

Do not interpret the presence of a script as proof that the full workflow is implemented.

## Configuration

YAML files in `configs/` separate settings from code:

- `data.yaml`: data paths and discovery-related settings.
- `preprocessing.yaml`: preprocessing options.
- `training.yaml`: training options.
- `evaluation.yaml`: evaluation options.
- `models/flair.yaml`: FLAIR model settings.
- `models/t1.yaml`: T1 model settings.
- `models/fusion.yaml`: multimodal fusion settings.
- `experiments/README.md`: experiment configuration guidance.

Review the actual configuration keys before editing them. A setting in a YAML file may not yet be consumed by a complete pipeline. Do not add absolute paths such as `/home/<person>/...` or `C:\Users\<person>\...` to committed configuration files. Prefer repository-relative paths and documented command-line overrides.

## Development workflow for a team

### 1. Clone and create a local environment

Follow the platform-specific installation instructions above. Every developer has their own `.venv`.

### 2. Start from an up-to-date main branch

```bash
git switch main
git pull
git switch -c feature/short-description
```

If the repository's default branch is named `master` or another name, use that name instead of `main`.

### 3. Make a focused change

Keep changes small and focused. Examples include adding a unit-tested normalization option, implementing a dataset check, or adding one model component. Avoid mixing preprocessing changes, architecture changes, and evaluation changes in a single commit unless they are inseparable.

### 4. Run checks before committing

```bash
python -m pytest
python scripts/check_environment.py
git diff --check
git status --short
```

For model code, include CPU-forward tests and tensor shape tests even when GPU training is the main target. CUDA-specific tests should be optional so contributors without an NVIDIA GPU can run the standard suite.

### 5. Commit and push your branch

```bash
git add <specific-files>
git commit -m "Add concise description"
git push -u origin feature/short-description
```

Prefer adding explicit paths over `git add .` when data or generated outputs might be present. Never commit `.env` files containing secrets, medical data, checkpoints, virtual environments, or machine-specific paths.

### 6. Open a pull request

The pull request should describe:
- The problem and intended change.
- Files or modules affected.
- Tests run and their results.
- Any configuration changes.
- Expected effects on data processing or metrics.
- Known limitations and how another teammate can reproduce the result.

Ask a teammate to review before merging. Keep `main` in a working, tested state.

More detail is in `docs/development.md` and `CONTRIBUTING.md`.

## Testing and continuous integration

The project uses pytest. CI is configured under `.github/workflows/tests.yml` to run tests on Linux and Windows with Python 3.12. Check the GitHub Actions run after pushing; local success alone does not prove the hosted workflow works.

Tests should cover:
- Input validation and meaningful error messages.
- Geometry/affine behavior using synthetic images.
- Normalization edge cases.
- Metric edge cases such as empty masks.
- Model input/output shapes and CPU forward passes once models are implemented.
- Dataset pairing and subject-level split logic once the dataset contract is defined.

Use synthetic fixtures for CI. Never upload private MRI data to GitHub Actions.

## Reproducibility and experiment reporting

Before comparing model variants, document:
- Dataset source/version and access conditions (without exposing sensitive identifiers).
- Inclusion/exclusion criteria.
- Subject-level train/validation/test split and split seed.
- Modalities and missing-modality policy.
- Image orientation, spacing, registration target, skull-stripping method, normalization, and patch strategy.
- Model architecture and parameter count.
- Loss, optimizer, learning rate, scheduler, batch size, epoch count, stopping rule, and random seeds.
- Hardware, Python/PyTorch versions, runtime, peak GPU memory where available.
- Validation-selection rule and final test protocol.
- Per-subject and aggregate metrics, uncertainty where feasible, and failure cases.

Do not tune on the held-out test set. Compare experiments using the same split and evaluation code. Report limitations and negative results as well as improvements. See `docs/experiments.md`.

## Known limitations and planned work

The team should implement and validate the pipeline in stages:

1. Agree on the dataset contract and mask semantics.
2. Strengthen dataset validation and pairing.
3. Implement preprocessing with geometry-preserving tests.
4. Choose and validate registration and skull-stripping tools.
5. Implement a minimal FLAIR baseline and CPU shape/forward tests.
6. Add a reproducible training loop, checkpointing, and validation.
7. Implement prediction and evaluation end to end.
8. Add T1-only and multimodal fusion experiments.
9. Benchmark ablations against a fixed baseline.
10. Review cross-platform CI and document exact reproduction steps.

This repository is a research project, not a medical device. Model outputs must not be used for clinical diagnosis or treatment decisions. Any future clinical application requires appropriate independent validation, governance, and regulatory review.

## Troubleshooting

See `docs/troubleshooting.md` for virtual-environment, dependency, CUDA, dataset, Git, and test issues.

## Documentation index

- `docs/architecture.md` — module boundaries and data flow.
- `docs/dataset-setup.md` — data placement, dataset contract, and audit.
- `docs/development.md` — environment, coding, tests, Git workflow.
- `docs/experiments.md` — fair comparisons and reproducibility.
- `docs/installation-linux.md` — Linux setup.
- `docs/installation-windows.md` — Windows setup.
- `docs/troubleshooting.md` — common problems.
- `CONTRIBUTING.md` — contribution rules.
- `SECURITY.md` — reporting security/privacy concerns.

## License and data rights

The repository's `LICENSE` file currently states that no software license has been selected. Do not assume the code is open-source or freely reusable until the owners select a license. Dataset access and redistribution rights are separate from the code license.