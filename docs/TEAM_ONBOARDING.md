# Team Onboarding Checklist

Use this checklist when a new teammate joins the repository.

## First-time setup

- [ ] Obtain authorized access to the GitHub repository.
- [ ] Read the README and architecture documentation.
- [ ] Install Git and Python 3.12.
- [ ] Clone the repository.
- [ ] Create and activate a local `.venv`.
- [ ] Install `.[dev,imaging]`.
- [ ] Install a suitable PyTorch build if working on deep-learning code.
- [ ] Run `python scripts/check_environment.py`.
- [ ] Run `python -m pytest` and confirm the current suite passes.
- [ ] Confirm local MRI data and outputs are not tracked by Git.

## Before working on the dataset

- [ ] Confirm data-access authorization and storage location.
- [ ] Read the dataset's official documentation.
- [ ] Confirm modalities, file naming, mask semantics, and spatial conventions.
- [ ] Review the current audit script's limitations.
- [ ] Do not upload dataset data to GitHub.

## Before contributing code

- [ ] Pull the latest default branch.
- [ ] Create a feature branch.
- [ ] Understand whether the module is implemented or a placeholder.
- [ ] Add tests for new behavior.
- [ ] Run tests and `git diff --check`.
- [ ] Inspect staged files before committing.
- [ ] Open a pull request with a reproducible description.

## Questions the team must resolve

- Which dataset version and mask-label convention are authoritative?
- What are the expected modalities and missing-modality policy?
- What reference space and voxel spacing should be used?
- Which registration and skull-stripping tools are approved?
- What is the subject-level split?
- What is the primary metric and how are empty masks handled?
- Which experiments are baseline, ablation, and final evaluation?
- Who reviews data-processing and model changes?
