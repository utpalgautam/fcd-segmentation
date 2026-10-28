# Experiment Protocol and Reporting

## Purpose

This document defines how the team should plan and report experiments once end-to-end model training and evaluation are implemented. It does not prescribe a final architecture or claim any result.

## Experiment tracks

The configuration tree reserves separate tracks for:

- `configs/models/flair.yaml`: FLAIR-only.
- `configs/models/t1.yaml`: T1-weighted-only.
- `configs/models/fusion.yaml`: multimodal FLAIR + T1.

The actual model builders are currently placeholders. Before comparing these tracks, make their input/output conventions, training protocol, data splits, and evaluation procedure consistent.

## Required experiment record

Each experiment should have a unique, stable ID and record:

| Category | Record |
|---|---|
| Code | Git commit hash and branch |
| Data | Dataset version/source, inclusion criteria, subject counts by split |
| Split | Subject IDs or a protected mapping, split method, random seed |
| Modalities | Inputs used and missing-modality policy |
| Preprocessing | Orientation, reference space, registration, spacing, skull stripping, normalization, patch size |
| Model | Architecture, input/output channels, parameter count |
| Optimization | Loss, optimizer, learning rate, scheduler, batch size, epochs, stopping rule |
| Reproducibility | Random seeds and deterministic settings |
| Hardware | CPU/GPU, VRAM, OS, Python/PyTorch versions |
| Selection | Validation metric and checkpoint-selection rule |
| Evaluation | Metrics, aggregation method, confidence intervals if available |
| Artifacts | Checkpoint path, configuration snapshot, logs, predictions and plots |
| Interpretation | Main findings, failures, limitations, next action |

Do not include sensitive participant details in a public report.

## Baseline-first principle

1. Define and freeze a subject-level split.
2. Implement one minimal baseline end to end.
3. Verify labels, geometry, and predictions on synthetic examples and selected approved data.
4. Run the baseline with a recorded configuration and seed.
5. Change one major factor at a time when practical.
6. Compare against the same baseline, split, and metrics.
7. Preserve both positive and negative results.

Do not claim that a preprocessing change improves performance until it has been tested under a controlled comparison.

## Avoid leakage

- Split by subject, not slice, patch, or volume fragment.
- Keep multiple sessions from one subject together unless the protocol explicitly justifies otherwise.
- Fit data-dependent normalization/statistics on training data only when applicable.
- Do not select epochs, thresholds, or hyperparameters using the test set.
- Use the validation set for model selection and the held-out test set for final evaluation.
- Document exclusions and missing data.

## Metrics

The primary metric must be selected based on the project's scientific objective and dataset label semantics. Candidate segmentation metrics may include Dice, sensitivity/recall, precision, and specificity, but no one metric should be assumed sufficient. For small lesions, report per-subject results and inspect false positives/false negatives; pooled voxel metrics can obscure subject-level failures.

Document:
- whether background is included;
- handling of empty reference masks and empty predictions;
- whether metrics are averaged per subject or computed over pooled voxels;
- thresholding/argmax rules;
- whether lesion-level detection metrics are needed;
- uncertainty estimates where feasible.

Use the same implementation and conventions for all compared models.

## Report template

For each run, report:

1. **Question/hypothesis**
2. **Change from baseline**
3. **Fixed variables**
4. **Dataset and split**
5. **Configuration and seed**
6. **Training details**
7. **Validation and test results** (clearly separated)
8. **Qualitative examples and QC**
9. **Failure cases**
10. **Limitations**
11. **Conclusion**
12. **Next experiment**

A result is not reproducible if the code revision, data version, split, configuration, and evaluation conventions cannot be recovered.

## Current status

At the time this document was written, full model training, inference, and evaluation workflows are incomplete. Do not publish quantitative model-performance claims until those workflows and their validation are implemented.
