# Windows Installation

These instructions use Windows PowerShell and Python 3.12.

## 1. Install prerequisites

Install Git for Windows and Python 3.12. Ensure the Python launcher (`py`) is available. Restart PowerShell after installing if the commands are not recognized.

## 2. Clone the repository

Replace the placeholder URL with the repository URL:

```powershell
git clone <YOUR_REPOSITORY_URL>
Set-Location fcd-segmentation
```

## 3. Create and activate the virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Verify the active interpreter:

```powershell
python --version
python -m pip --version
```

The paths should refer to the repository's `.venv`.

### If PowerShell blocks activation

Use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Then continue using the `python` commands below. Avoid changing machine-wide execution policy without understanding the implications.

## 4. Install project dependencies

```powershell
python -m pip install -e ".[dev,imaging]"
```

Install PyTorch separately from the official selector if required: https://pytorch.org/get-started/locally/

## 5. Verify

```powershell
python scripts/check_environment.py
python -m pytest
```

## 6. Later sessions

```powershell
Set-Location path\to\fcd-segmentation
.\.venv\Scripts\Activate.ps1
```

## 7. Update your branch

```powershell
git switch main
git pull
```

Use your team's actual default branch name if it differs.

## Notes

- Do not copy a `.venv` from Linux or another Windows machine.
- Never commit `.venv/`, MRI data, outputs, or secrets.
- CUDA is available only when compatible NVIDIA hardware, drivers, and a suitable PyTorch build are installed.
- Utility tests should run without a GPU. End-to-end model training and inference are not yet implemented.
