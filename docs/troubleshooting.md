# Troubleshooting

## The wrong Python interpreter is active

Check:

```bash
python --version
python -m pip --version
```

On Linux/macOS-like shells:

```bash
which python
```

On Windows PowerShell:

```powershell
Get-Command python
```

Activate the repository's virtual environment again. Use `python -m pip` rather than a bare `pip` command to ensure packages install into the active interpreter.

## `python3.12` or `py -3.12` is not found

Python 3.12 is the recommended version for the current repository. Install it using a trusted method for your operating system, then recreate the environment with the correct interpreter. Do not create the environment with one Python version and assume it changes later.

## Package installation fails

1. Activate `.venv`.
2. Upgrade pip: `python -m pip install --upgrade pip`.
3. Read the first actual error in the pip output.
4. Confirm Python version and internet access.
5. Retry the documented install command only after addressing the cause.

If an editable installation has become inconsistent after dependency or packaging changes, remove and recreate `.venv` rather than deleting arbitrary package files inside it. Preserve your source files and local data.

## `torch.cuda.is_available()` returns `False`

CUDA requires compatible NVIDIA hardware, a supported driver, and a compatible PyTorch package.

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

On Windows, `nvidia-smi` can also be run in a terminal if the NVIDIA driver provides it.

Install the correct PyTorch build using the official selector: https://pytorch.org/get-started/locally/. Do not install a CUDA Toolkit blindly or assume that the CUDA version printed by `nvidia-smi` must exactly match the wheel's CUDA runtime.

If no compatible NVIDIA GPU is available, use the CPU build. Standard utility tests should not require CUDA.

## CUDA out-of-memory

3D MRI tensors can be large. Reduce batch size and patch dimensions, ensure other processes are not using GPU memory, and consider mixed precision if it is numerically appropriate. Use `nvidia-smi` to inspect GPU memory. Do not change several training settings at once without recording the experiment.

The current developer's 4 GB RTX 3050 Ti is memory constrained for full-volume 3D training.

## Tests fail

Run a focused test first:

```bash
python -m pytest tests/test_runtime.py -v
```

Then run the complete suite:

```bash
python -m pytest
```

Read the first failure and traceback. Do not edit tests merely to suppress a failure; determine whether the code, test expectation, or environment is wrong. If the behavior changes intentionally, update the test and document why.

## Dataset audit finds no FLAIR files

Check the actual data path and script help:

```bash
python scripts/audit_dataset.py --help
```

The audit currently searches for candidate FLAIR files and is not a full validator. Confirm the dataset's file extensions, naming rules, and directory structure. Do not rename the dataset until the discovery rules have been reviewed.

## Image and mask shapes differ

Different shapes may indicate different spacing, orientation, field of view, or reference space. Do not simply reshape arrays. Inspect image metadata and affines, decide the reference space, and implement a tested registration/resampling procedure. Use nearest-neighbor interpolation for discrete labels.

## Git shows `__pycache__` or `.egg-info`

These are generated artifacts and should be ignored. Check:

```bash
git check-ignore -v src/fcd_segmentation/__pycache__/cli.cpython-312.pyc
git check-ignore -v src/fcd_segmentation.egg-info/PKG-INFO
git status --short
```

If a generated file was already tracked, `.gitignore` alone does not untrack it. Review carefully before using `git rm --cached`; never delete user data to clean Git status.

## `data/` or `outputs/` appears in Git status

Inspect before staging:

```bash
git status --short
git check-ignore -v data/raw/example.nii.gz
```

Do not commit medical data or model outputs. If the ignore rules do not cover a file, fix `.gitignore` before staging. Keep `.gitkeep` placeholders if you want the empty directories to appear in the repository.

## Windows path or activation errors

Use PowerShell commands from `docs/installation-windows.md`. Create a fresh Windows `.venv`; never reuse a Linux environment. If PowerShell activation is blocked, use Command Prompt activation rather than changing a system-wide policy without review.

## A script exists but raises `NotImplementedError`

This is expected for current placeholders such as model builders and the trainer. The repository is an early scaffold. Follow the implementation roadmap in the README and architecture documentation rather than treating placeholder entry points as finished features.
