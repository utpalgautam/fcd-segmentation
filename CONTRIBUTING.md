# Contributing

Thank you for contributing to the FCD Segmentation research project.

## Before starting

1. Read `README.md`, `docs/architecture.md`, and `docs/development.md`.
2. Confirm the dataset and research constraints relevant to your change.
3. Check the current GitHub issues/board so work does not conflict.
4. Create your own virtual environment using the platform-specific installation guide.

## Branches and commits

Use a descriptive branch name, for example:

- `docs/dataset-contract`
- `feature/flair-baseline`
- `feature/mask-validation`
- `fix/affine-check`

Keep each pull request focused. Use clear commit messages such as `Add mask label validation`.

## Tests and documentation

Run:

```bash
python -m pytest
git diff --check
git status --short
```

Add or update tests for behavior changes. Update configuration examples and documentation when public behavior, CLI arguments, data assumptions, or outputs change. Use synthetic data for automated tests.

## Medical data and security

Never commit raw or processed MRI, masks, identifiable metadata, private dataset paths, credentials, local environment files, or checkpoints. Check `git status` and review the staged diff before every commit. Use only data that the team is authorized to access and process.

## Research quality

For changes to preprocessing, model architecture, training, or metrics:
- Explain the scientific or engineering motivation.
- Record important assumptions and parameter choices.
- Add tests for geometry, label integrity, shapes, and edge cases as appropriate.
- Keep comparisons fair by preserving data splits and evaluation conventions.
- Do not use test-set results for model selection.
- Report limitations and negative results honestly.

## Pull request checklist

- [ ] Scope and rationale are explained.
- [ ] Relevant tests added/updated and passing.
- [ ] `git diff --check` passes.
- [ ] Documentation/configuration updated.
- [ ] No data, secrets, generated files, or machine-specific paths staged.
- [ ] Cross-platform implications considered.
- [ ] Limitations and validation status are stated.
