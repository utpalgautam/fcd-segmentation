# Linux Installation

These instructions target Linux with Python 3.12. Commands assume the repository's default directory name is `fcd-segmentation`.

## 1. Install system tools

On Ubuntu:

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv python3-pip
```

If your distribution does not provide Python 3.12 under these package names, install a compatible Python 3.12 distribution using a trusted source for your OS.

## 2. Clone the repository

After the GitHub repository has been created, replace the placeholder URL:

```bash
git clone <YOUR_REPOSITORY_URL>
cd fcd-segmentation
```

## 3. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Confirm that Python points into `.venv`:

```bash
which python
python --version
```

The executable path should contain the repository's `.venv` directory.

## 4. Install project dependencies

```bash
python -m pip install -e ".[dev,imaging]"
```

Install PyTorch separately using the official selector if needed: https://pytorch.org/get-started/locally/

## 5. Verify

```bash
python scripts/check_environment.py
python -m pytest
```

## 6. Later sessions

```bash
cd fcd-segmentation
source .venv/bin/activate
```

Deactivate when finished:

```bash
deactivate
```

## 7. Update your branch

```bash
git switch main
git pull
```

If your team uses a different default branch, use that branch name.

## Notes

- Never commit `.venv/`.
- Keep MRI data and outputs in the local ignored directories.
- A CUDA-enabled PyTorch installation is optional for utility tests.
- The model builders and end-to-end training/inference workflows are not complete yet.
