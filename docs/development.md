# Development Guide

## 1. Environment setup

Use Python 3.12 for consistency with the current CI configuration. Create a separate virtual environment for each clone and operating system.

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,imaging]"
```

Install PyTorch separately from the official selector if working on deep-learning functionality: https://pytorch.org/get-started/locally/

## 2. Verify before coding

```bash
python scripts/check_environment.py
python -m pytest
git status --short
```

On Windows, use the same `python` commands after activating `.venv`.

## 3. Repository conventions

- Keep importable application code in `src/fcd_segmentation/`.
- Put command-line entry points in `scripts/`.
- Put configuration in `configs/`, and document configuration options.
- Add unit tests under `tests/` for new behavior and edge cases.
- Use `pathlib.Path` rather than hard-coded slash paths.
- Do not hard-code a developer's home directory or machine-specific paths.
- Use explicit exceptions and informative error messages.
- Keep optional GPU/deep-learning dependencies optional for CPU utility tests.
- Use type hints and docstrings for public functions.
- Keep label arrays discrete; do not treat masks like intensity images.
- Make randomness configurable and record seeds.
- Avoid silent exception handling or silently dropping invalid subjects.
- Do not commit raw or processed medical images, logs containing sensitive data, checkpoints, local environment files, or generated outputs.

## 4. Configuration changes

Before adding a YAML key:
1. Decide which module owns it.
2. Add validation and a useful error for invalid values.
3. Pass the setting into the relevant implementation.
4. Add tests for valid and invalid configurations.
5. Update documentation and sample configuration.
6. Verify that Linux and Windows paths behave correctly.

A configuration key is not implemented merely because it appears in YAML.

## 5. Tests

Run the full suite:

```bash
python -m pytest
```

Run one file:

```bash
python -m pytest tests/test_normalization.py -v
```

Run with coverage if the development dependencies provide pytest-cov:

```bash
python -m pytest --cov=fcd_segmentation --cov-report=term-missing
```

Tests should use synthetic data. Avoid requiring the private dataset or an NVIDIA GPU for standard CI tests. Add optional GPU tests only when they skip cleanly if CUDA is unavailable.

For each model implementation, add at least:
- construction tests;
- input/output shape tests;
- a CPU forward-pass test;
- invalid-input tests where applicable;
- loss/label compatibility tests;
- tests for configurable input channels and output classes.

## 6. Git workflow

Check the repository's default branch on GitHub. If it is `main`, a typical workflow is:

```bash
git switch main
git pull
git switch -c feature/short-description
```

Make a focused change, then run:

```bash
python -m pytest
git diff --check
git status --short
git diff
```

Stage only intended files:

```bash
git add src/fcd_segmentation/example.py tests/test_example.py docs/example.md
git commit -m "Implement example behavior"
git push -u origin feature/short-description
```

Open a pull request and describe the design, tests, configuration changes, and any effect on preprocessing or results. Do not merge unreviewed changes directly into the shared default branch unless the team has explicitly agreed to that workflow.

## 7. Review checklist

- [ ] Change has a clear scope.
- [ ] No private dataset, participant identifiers, secrets, local paths, or generated files are staged.
- [ ] Tests added or updated.
- [ ] Full test suite passes.
- [ ] `git diff --check` passes.
- [ ] Configuration and documentation are updated.
- [ ] Behavior is compatible with Windows and Linux where intended.
- [ ] Errors and edge cases are handled explicitly.
- [ ] Research-impacting choices are recorded.

## 8. Model and data-processing review

Changes to preprocessing can alter model inputs and invalidate comparisons. For such changes, record the old/new behavior, affected subjects, spatial metadata checks, and the experiment that will measure the impact. Keep a stable baseline to compare against.

Never use test-set performance to choose preprocessing parameters or model architecture.
