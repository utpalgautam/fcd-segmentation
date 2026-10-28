# Architecture and Module Responsibilities

## Status

This document describes the intended architecture of the repository as it exists today. It is not a claim that the full end-to-end pipeline has been implemented. Model builders, the training loop, registration, skull stripping, and end-to-end inference remain incomplete.

## High-level data flow

```text
Local dataset
    |
    v
Discovery and validation
    |
    v
Preprocessing
  - image I/O
  - orientation/geometry checks
  - registration (to be implemented/validated)
  - resampling
  - skull stripping (to be implemented/validated)
  - intensity normalization
    |
    v
Subject-level splits and dataset loader (to be implemented)
    |
    +------------------+-------------------+
    |                  |                   |
    v                  v                   v
 FLAIR model        T1 model          FLAIR + T1 fusion
    |                  |                   |
    +------------------+-------------------+
                       |
                       v
                Training and validation
                       |
                       v
                Checkpoint/model artifact
                       |
                       v
                Inference on held-out data
                       |
                       v
              Metrics, reports, QC and analysis
```

## Package boundaries

### `fcd_segmentation.config`

Loads and validates configuration. Keep paths and experiment settings out of hard-coded machine-specific locations. Verify that any new configuration option is actually consumed by the relevant code.

### `fcd_segmentation.runtime`

Handles optional PyTorch device selection. Utility and data tests should not require CUDA. Code requesting an unsupported device should fail with a clear error.

### `fcd_segmentation.data`

- `discovery.py`: finds candidate input files.
- `validation.py`: performs basic data checks.

This layer will need a documented dataset contract, robust modality/mask pairing, subject-level grouping, and explicit validation errors before it can feed a production training pipeline.

### `fcd_segmentation.preprocessing`

- `io.py`: image read/write helpers.
- `geometry.py`: spatial geometry and affine helpers.
- `normalization.py`: intensity normalization helpers.
- `resampling.py`: resampling helpers.
- `registration.py`: currently incomplete/placeholder.
- `skull_stripping.py`: currently incomplete/placeholder.

Preprocessing must preserve spatial correspondence between images and labels. Intensity interpolation and label interpolation are different: continuous MRI intensities generally need continuous interpolation, while discrete masks require label-safe interpolation such as nearest-neighbor. The team's exact reference space and target spacing must be decided and tested.

### `fcd_segmentation.models`

- `flair_model.py`: FLAIR-only model builder placeholder.
- `t1_model.py`: T1-only model builder placeholder.
- `fusion_model.py`: multimodal fusion model builder placeholder.

The three model tracks should expose documented input/output tensor shapes and consistent segmentation-logit conventions. Fusion design depends on whether modalities are aligned and what the project report requires. Define a missing-modality policy rather than silently substituting data.

### `fcd_segmentation.training`

- `reproducibility.py`: reproducibility helpers.
- `trainer.py`: training placeholder.

The training implementation needs a dataset loader, subject-level splits, loss, optimizer, validation loop, checkpoint-selection rule, logging, and reproducibility settings. Keep held-out test data out of training and model selection.

### `fcd_segmentation.inference`

`predict.py` contains inference utilities but the full prediction workflow is not complete. A future predictor must record preprocessing metadata, restore predictions to the intended image space if needed, and save segmentation outputs with valid spatial metadata.

### `fcd_segmentation.evaluation`

`metrics.py` provides basic metric utilities. Full evaluation must define empty-mask behavior, aggregation (per subject vs pooled voxels), lesion-level analysis if required, and how predictions and labels are aligned.

## Configuration and entry points

- `configs/` holds YAML configuration files.
- `scripts/` contains executable entry points for environment checks, dataset audit, preparation, training, prediction, and evaluation.
- `tests/` contains automated tests.
- `docs/` describes usage and decisions.

The existence of an entry-point script or configuration file does not imply the corresponding end-to-end workflow is complete.

## Design principles

1. **Geometry is part of the data.** Never compare voxel arrays without verifying shape, affine, orientation, and spatial alignment.
2. **Split by subject.** Multiple scans or patches from one participant must not leak across train/validation/test.
3. **Keep labels discrete.** Do not apply intensity normalization to masks; preserve and validate label values.
4. **Make preprocessing deterministic and traceable.** Record configuration and tool versions.
5. **Fail loudly on ambiguous data.** Missing or mismatched modalities/masks should be reported, not silently skipped.
6. **Separate experiments.** Change one major factor at a time where possible.
7. **Keep private data local.** CI and source control should use synthetic fixtures.
8. **Test CPU paths.** A contributor without a GPU should still be able to run utility and shape tests.

## Decisions to record before implementing the full pipeline

- Dataset format, naming conventions, modality availability, mask encoding and label semantics.
- Reference modality and target image space.
- Registration tool, transform direction, and resampling interpolation.
- Skull-stripping method and failure/QC policy.
- Intensity normalization method and foreground definition.
- 2D vs 3D model and patch/volume strategy.
- Model input channels, output channels, and activation/loss convention.
- Missing-modality behavior for fusion.
- Subject-level split strategy and seed.
- Primary metrics and checkpoint-selection rule.
