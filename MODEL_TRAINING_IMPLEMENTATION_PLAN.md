# Model Training Implementation Plan — FCD Lesion Segmentation

**Project:** FCD lesion segmentation  
**Target:** nnU-Net v2 training workflow for FLAIR, T1, and multimodal FLAIR+T1 datasets  
**Environment:** Windows + PowerShell, developed in Antigravity; training may later run on a professor's GPU machine  
**Purpose:** Implementation specification for the coding agent. This document does not claim that training has already been run.

## 1. Objective

Implement a reliable, reproducible model-training workflow in the existing repository using **nnU-Net v2**. Integrate with the existing preprocessing outputs and project conventions rather than creating a separate, conflicting pipeline.

| Dataset ID | Dataset folder | Input channels | Purpose |
|---|---|---|---|
| 101 | `Dataset101_FLAIR` | FLAIR | FLAIR-only baseline |
| 102 | `Dataset102_T1` | T1 | T1-only baseline |
| 103 | `Dataset103_Fusion` | FLAIR + T1 | Multimodal model |

The workflow should support dataset validation, nnU-Net planning/preprocessing, five-fold cross-validation training, resumable training, clear logs, and a summary of what ran and what remains.

**Important current state:** The local processed data was intentionally generated with `--limit 1` as a smoke test. Do not treat it as the final training cohort. Detect the small cohort and refuse to start five-fold training unless there are enough valid labeled cases. Never bypass the guard just to make training start.

## 2. Inspect the existing repository first

Before editing, inspect the repository and read relevant files, including (if present):

- `scripts/train.py`
- `scripts/prepare_dataset.py`
- `scripts/predict.py`
- `scripts/evaluate.py`
- `scripts/audit_dataset.py`
- `configs/data.yaml`
- `configs/training.yaml`
- `configs/evaluation.yaml`
- `configs/preprocessing_config.json`
- model configs under `configs/models/`
- `src/` and `tests/`
- `README.md` and relevant documentation under `docs/`
- generated `dataset.json` files in all three processed nnU-Net dataset folders

Follow existing naming, logging, configuration, and error-handling conventions. Preserve useful functionality already present. Do not blindly overwrite existing files. If part of the workflow already exists, extend and test it.

Do not modify the preprocessing algorithm or dataset-generation logic unless a training integration defect requires it. If a defect is found, explain it and make the smallest justified change.

## 3. Expected nnU-Net dataset layout

The intended nnU-Net raw root is `data/processed/`, containing:

```text
data/processed/
├── Dataset101_FLAIR/
│   ├── dataset.json
│   ├── imagesTr/<case_id>_0000.nii.gz
│   ├── labelsTr/<case_id>.nii.gz
│   └── imagesTs/                 # optional until held-out cases are staged
├── Dataset102_T1/
│   ├── dataset.json
│   ├── imagesTr/<case_id>_0000.nii.gz
│   ├── labelsTr/<case_id>.nii.gz
│   └── imagesTs/                 # optional
└── Dataset103_Fusion/
    ├── dataset.json
    ├── imagesTr/<case_id>_0000.nii.gz  # FLAIR
    ├── imagesTr/<case_id>_0001.nii.gz  # T1
    ├── labelsTr/<case_id>.nii.gz
    └── imagesTs/                 # optional
```

Confirm the actual outputs before assuming every detail above. Use the case IDs and naming rules produced by preprocessing. Do not rename files or fabricate missing masks.

`dataset.json` should accurately describe channel names and labels. For fusion, channel 0 is FLAIR and channel 1 is T1. The foreground lesion label is `1`, background is `0`. Verify these against actual files and project configuration before changing them.

## 4. Environment and paths

The workflow must work in Windows PowerShell and should not require hard-coded absolute paths tied to a specific computer.

Use repository-relative defaults, resolved from the project root. Configure or validate:

- `nnUNet_raw` → absolute path to `data/processed`
- `nnUNet_preprocessed` → absolute path to `data/nnunet_preprocessed`, unless the repository already has an approved location
- `nnUNet_results` → absolute path to `outputs/nnunet_results`, unless the repository already has an approved location

Requirements:

1. Do not overwrite a user's explicitly configured environment paths without explaining it.
2. Pass resolved environment values to child processes explicitly.
3. Print effective paths clearly.
4. Create output directories when appropriate.
5. Explain that PowerShell `$env:` values are session-local unless the user chooses to persist them.
6. Keep paths portable; do not embed the user's Windows username or the professor's machine path.
7. Prefer an existing project configuration mechanism over duplicating settings.

## 5. Implementation requirements

### 5.1 Dataset discovery and validation

For every selected dataset, check:

- Dataset directory and `dataset.json` exist.
- `imagesTr/` and `labelsTr/` exist.
- `dataset.json` is valid JSON and required fields are present.
- Every training image has exactly one matching label using the existing case-ID convention.
- Every label has a corresponding image case.
- Every expected modality exists exactly once per case.
- FLAIR/T1/fusion channel suffixes match the intended modality order.
- No duplicate case IDs exist.
- Files are non-empty and use the expected `.nii.gz` extension.
- NIfTI files can be opened/read with an appropriate installed library (prefer existing dependencies such as nibabel).
- Labels contain only configured label values (normally 0 and 1); report unexpected values as errors.
- Image/label geometry and shape are compatible where required by the existing preprocessing contract.
- `dataset.json`'s `numTraining` matches the actual training-case count, or report/fix it safely only if existing conventions permit.
- Compare case IDs across FLAIR, T1, and fusion datasets and report mismatches. Do not assume cohorts match.

Errors must identify dataset, case ID, path, and suggested corrective action. Do not silently skip invalid cases. If a deliberate skip mode exists, report skipped cases and reasons in a machine-readable summary. Default behavior should fail safely on inconsistencies.

### 5.2 Cohort size and five-fold safeguards

Before starting five-fold training:

- Require at least **5 valid labeled cases** in the selected dataset as a hard minimum for five-fold splitting.
- Warn that five cases is only a technical minimum and is not a scientifically adequate cohort for reliable performance claims.
- If fewer than five cases exist, stop before invoking nnU-Net training. Explain that the current `--limit 1` smoke-test dataset must be replaced with the full successfully preprocessed cohort.
- Never duplicate cases, fabricate labels, or pretend training succeeded.
- Honor stricter requirements imposed by the installed nnU-Net operation.

### 5.3 nnU-Net planning and preprocessing

Implement an explicit preparation operation for selected datasets, using the installed nnU-Net v2 planner/preprocessor and dataset integrity verification.

- Support one dataset or all three.
- Invoke the CLI with `subprocess.run` and an argument list, not a shell-composed command string.
- Pass the same environment variables/paths as training.
- Capture command, exit code, stdout/stderr or log files.
- On failure, stop and show the failed command and useful diagnostics.
- Do not swallow exceptions.
- Avoid repeating expensive planning if a safe, explicit skip/reuse mechanism exists; never silently trust stale outputs.
- Do not run planning during import or automatically on module import.
- Verify actual installed CLI syntax before coding.
- Do not assume planning succeeds with a one-case smoke-test cohort. If it fails due to cohort size, report it rather than weakening validation.

### 5.4 Five-fold training

Implement training for dataset IDs 101, 102, and 103 using the nnU-Net v2 CLI and a configuration supported by the actual generated plans.

Capabilities:

- Train one selected dataset or all three.
- Train one fold or folds `0, 1, 2, 3, 4`.
- Use `3d_fullres` only if supported by the actual plans; otherwise use a valid configuration from the generated plans and document the choice.
- Run folds sequentially by default to avoid competing for GPU memory.
- Log dataset, configuration, fold, start/end time, elapsed time, command, exit code, and outcome.
- Stop on first failure by default. An explicit `--keep-going` mode may record failures and continue.
- Support resuming an interrupted fold using the installed nnU-Net resume option. Verify `nnUNetv2_train --help`; the flag is commonly `--c`, but do not assume.
- Do not delete checkpoints, plans, preprocessed data, or prior results.
- Never train if validation or case-count checks fail.
- If all datasets are selected, explain this means three experiments and up to fifteen fold runs.
- Do not claim a model was trained until the command succeeds and expected artifacts are verified.

Prefer these wrapper interfaces, adapting if existing code already has a compatible CLI:

```powershell
python .\scripts\train.py --check
python .\scripts\train.py --prepare
python .\scripts\train.py --dataset 101 --fold 0
python .\scripts\train.py --dataset 101 --all-folds
python .\scripts\train.py --dataset all --all-folds
python .\scripts\train.py --dataset 101 --fold 0 --continue
```

If the existing CLI differs, preserve compatibility and document the final commands. Map wrapper `--continue` to the verified nnU-Net resume flag only after checking the installed CLI help.

### 5.5 Logs and run summaries

Write reproducible records, for example under `outputs/training/`:

- timestamped logs
- machine-readable JSON run summary
- concise human-readable summary if useful

Include dataset path/ID, configuration and folds, case count and validation result, effective environment paths, exact commands, timestamps, elapsed time, exit status, log path, results directory, detected artifacts, failure message, and next action. Do not log secrets or include large NIfTI contents.

### 5.6 GPU/device support

- Use only device arguments supported by the installed nnU-Net v2 CLI.
- Default to installed nnU-Net behavior unless the project already specifies a device.
- Optional device selection may be supported if valid.
- Report PyTorch/CUDA status, but do not require CUDA for `--check` or validation.
- Do not assume the local RTX 3050 is the final training GPU.
- Do not hard-code GPU names, CUDA versions, or absolute paths.

### 5.7 Configuration and documentation

Use `configs/training.yaml` as the primary configuration if that fits existing conventions. Extend it carefully rather than creating competing config files.

Where appropriate, configure dataset IDs/folders, nnU-Net configuration, folds, results/log paths, path defaults, device, and error-handling behavior. Do not place secrets or false cohort assumptions in configuration.

Update README or relevant docs with:

- required Python and nnU-Net version assumptions
- activation of the existing virtual environment
- checks and dataset validation
- dataset preparation
- one-fold, all-fold, and all-dataset training
- resuming interrupted training
- locations of logs, checkpoints, plans, and results
- setting paths on the professor's GPU machine
- meaning of the one-case smoke test and what must happen after full preprocessing
- common errors and fixes
- distinction between smoke testing, training, cross-validation, and final held-out testing

Use Windows PowerShell commands in documentation.

### 5.8 Tests

Add/update tests using the existing test framework. Tests must not start GPU training or require the full MRI dataset.

At minimum test:

- dataset ID/name mapping
- environment path resolution
- missing dataset directory or `dataset.json`
- invalid JSON
- missing image or label
- missing fusion channel
- duplicate/mismatched case IDs
- invalid label values where practical
- fewer than five cases blocks training
- a valid five-case fixture passes the technical count guard
- planning/training command construction
- resume flag mapping based on verified CLI behavior
- nonzero subprocess exit is reported as failure
- importing training module does not execute training

Use temporary directories and mocked subprocess calls. If nibabel is already a dependency, use small synthetic NIfTI fixtures. Avoid adding heavyweight dependencies without justification.

## 6. Non-goals

Do not implement/change these unless a minimal integration fix is required:

- a new neural network architecture from scratch
- a custom U-Net loop outside nnU-Net
- MRI preprocessing algorithms
- labels or lesion masks
- synthetic/duplicated cases
- final evaluation metrics (Dice, HD95, sensitivity, MAE/MdAE, etc.), except preserving an existing interface if necessary
- final held-out test evaluation
- inference/prediction changes
- hyperparameter optimization
- claims of improvement without experiments

This task is the **model training workflow**, including validation, logging, tests, and documentation.

## 7. Implementation sequence

1. Inspect repository/current training code and report a brief inventory before editing.
2. Verify installed nnU-Net v2 version and actual `nnUNetv2_plan_and_preprocess --help` and `nnUNetv2_train --help`.
3. Define dataset mapping and effective paths.
4. Implement dataset validation and the five-case guard.
5. Implement `--check` without planning/training.
6. Implement `--prepare` with integrity verification and captured logs.
7. Implement one-dataset, one-fold training.
8. Add all-fold and all-dataset orchestration, sequential by default.
9. Add resume support based on verified CLI behavior.
10. Add logs, run summaries, documentation, and tests.
11. Run tests and syntax/static checks.
12. Safely validate the current one-case outputs and confirm training is blocked. Do not start actual GPU training.
13. Review final diff for unrelated changes.
14. Report changed files, checks/tests, later commands, and remaining limitations.

## 8. Acceptance criteria

- [ ] Existing code and repository conventions were inspected first.
- [ ] Existing useful behavior is preserved.
- [ ] `--check` reports nnU-Net/PyTorch/CUDA status and effective paths without training.
- [ ] Dataset validation checks all three layouts, modalities, labels, and metadata.
- [ ] One-case data is identified as a smoke test.
- [ ] Training is blocked before any training command if fewer than five valid cases exist.
- [ ] Planning commands use verified nnU-Net syntax.
- [ ] Single-fold training works when prerequisites are satisfied.
- [ ] Five-fold and all-dataset orchestration work sequentially by default.
- [ ] Resume uses the verified CLI option.
- [ ] Logs and JSON summaries record successes/failures.
- [ ] Errors provide context and next steps.
- [ ] Tests do not require GPU or the full MRI dataset.
- [ ] Tests/syntax checks pass or failures are explicitly reported.
- [ ] Documentation contains exact Windows PowerShell commands.
- [ ] No fabricated training results or metrics.
- [ ] No destructive deletion or overwriting of existing results.
- [ ] Final report lists changed files, tests, commands, and next actions.

## 9. Copy-paste prompt for Antigravity

You are implementing the model-training part of my existing FCD lesion segmentation project. Read `MODEL_TRAINING_IMPLEMENTATION_PLAN.md` completely and treat it as the specification.

Work in the existing repository. Inspect current files before editing, especially `scripts/train.py`, preprocessing/dataset scripts, `configs/training.yaml`, the three generated `dataset.json` files, relevant `src/`, tests, README, and docs. Preserve useful behavior and make the smallest justified changes. Do not blindly replace files.

Implement the complete nnU-Net v2 workflow for Dataset101_FLAIR, Dataset102_T1, and Dataset103_Fusion: dataset validation, path/environment setup, planning and integrity verification, single-fold training, five-fold training, all-dataset orchestration, resumable training, structured logs/run summaries, tests, and Windows PowerShell documentation.

Before implementing CLI calls, inspect the installed nnU-Net version and actual output of `nnUNetv2_plan_and_preprocess --help` and `nnUNetv2_train --help`. Use only supported arguments. Use `subprocess.run` with argument lists, capture logs, propagate failures, and never invoke training at import time. Do not delete checkpoints or prior results.

**Current state:** processed datasets contain one subject because I intentionally ran preprocessing with `--limit 1` as a smoke test. This is not the final cohort. Add a hard guard that blocks five-fold training with fewer than five valid labeled cases. Do not duplicate/fabricate cases or labels, and do not start actual training during this implementation task. Safe checks, tests, syntax checks, and validation are allowed.

Use `data/processed` as the default nnU-Net raw root only after confirming it contains the three expected Dataset folders. Resolve paths relative to the repository. Do not hard-code my Windows username or the professor's GPU path. Preserve existing configuration conventions.

Add tests with temporary fixtures and mocked subprocesses; tests must not need a GPU or the full MRI dataset. Update README or relevant docs with exact PowerShell commands for checking, preparing, training one fold, training all folds, training all three datasets, and resuming a fold.

Suggested CLI, adapting to existing conventions if necessary:

```powershell
python .\scripts\train.py --check
python .\scripts\train.py --prepare
python .\scripts\train.py --dataset 101 --fold 0
python .\scripts\train.py --dataset 101 --all-folds
python .\scripts\train.py --dataset all --all-folds
python .\scripts\train.py --dataset 101 --fold 0 --continue
```

Do not claim training succeeded unless it actually ran and expected artifacts exist. Do not fabricate metrics. Do not alter preprocessing behavior unless a minimal, demonstrated integration fix is required.

Work in this sequence: inspect and summarize first; implement in logical steps; run tests and syntax checks; validate the current one-case data and confirm training is safely blocked; inspect the final diff for unrelated changes. Then report (1) files changed, (2) design decisions, (3) tests/checks and outcomes, (4) exact commands for later full-cohort training, (5) failures/unresolved issues, and (6) confirmation that no training was started.

If a requirement conflicts with the current code or installed CLI, do not guess. Explain the discrepancy, inspect the relevant source/help, and implement the supported equivalent.

## 10. After Antigravity finishes

Run these from the repository root in PowerShell after reviewing the changes:

```powershell
.\.venv\Scripts\Activate.ps1
git status --short
git diff --check
git diff
python -m pytest -q
python .\scripts\train.py --check
```

Use the test command documented by the project if it does not use pytest.

**Do not run `--prepare` or training until full preprocessing has completed and validation confirms the full cohort is ready.** Once full preprocessing finishes, verify all three datasets contain the expected cases and labels, then prepare nnU-Net and begin with one dataset and one fold before launching all experiments.
