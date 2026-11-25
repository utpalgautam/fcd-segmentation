# Experiment Protocol & Benchmarking Methodology

This document outlines the experimental tracks, cross-validation design, evaluation metrics, and standardization protocol used in the FCD lesion segmentation project.

---

## 1. Experimental Tracks

The study systematically evaluates the diagnostic contribution of structural MRI modalities for automated Focal Cortical Dysplasia (FCD Type II) detection:

| Track | Dataset ID | Input Channels | Clinical Focus | Primary Objective |
| :--- | :---: | :---: | :--- | :--- |
| **FLAIR Baseline** | `Dataset101_FLAIR` | 1 (`_0000`: FLAIR) | Cortical hyperintensity, subcortical signal change | Baseline single-sequence detection |
| **T1 Baseline** | `Dataset102_T1` | 1 (`_0000`: T1w) | Cortical thickening, gray-white junction blurring | Morphological sequence detection |
| **Multimodal Fusion** | `Dataset103_Fusion` | 2 (`_0000`: FLAIR, `_0001`: T1w) | Simultaneous intensity + morphological dyslamination | Multimodal complementary synergy |

---

## 2. Cross-Validation & Cohort Control

To ensure reliable, unbiased reporting:
1. **Group-Stratified 5-Fold Splitting:**
   - The 142 training cases (57 FCD patients, 85 Healthy Controls) are partitioned into 5 balanced folds.
   - Each validation fold strictly contains 11–12 FCD subjects and 17 Healthy Controls.
   - The exact same subject splits (`splits_final.json`) are used across all three datasets to ensure matched paired comparisons.
2. **Strict Test Cohort Isolation:**
   - 28 FCD subjects are reserved exclusively for external held-out testing (`labelsTs`).
   - Test scans are never touched during fingerprint extraction, plan generation, training, or cross-validation.

---

## 3. Training Protocol

All models are trained under identical, controlled hyperparameters:
- **Architecture:** 3D PlainConvUNet (`dynamic_network_architectures`)
- **Plans:** `nnUNetPlans` (Full resolution 3D fullres, patch size $128 \times 160 \times 112$, batch size 2, 6 stages with up to 320 channels)
- **Trainer:** `FCDnnUNetTrainer_100epochs`
  - Training length: 100 epochs (25,000 total optimization steps)
  - Optimizer: SGD with momentum 0.99, weight decay $3 \times 10^{-5}$, Nesterov momentum
  - Initial Learning Rate: `0.01` ($1 \times 10^{-2}$)
  - Scheduler: Polynomial learning rate decay to 0 at epoch 100
  - Foreground Oversampling: `0.50` (50% patches guaranteed to contain lesion)
  - Checkpoint Frequency: Every 25 epochs + best validation score (`checkpoint_best.pth`)

---

## 4. Evaluation Metrics & Statistical Standards

Evaluations strictly report:
* **Dice Similarity Coefficient (DSC):** Voxel-wise spatial overlap between prediction and ground truth.
* **Sensitivity (True Positive Rate / Recall):** Proportion of true lesion voxels correctly detected.
* **Specificity (True Negative Rate):** Proportion of normal brain tissue correctly classified as non-lesion.
* **Precision (Positive Predictive Value):** Proportion of predicted lesion voxels that are truly lesion.
* **95th Percentile Hausdorff Distance (HD95):** Boundary distance in millimeters (computed with isotropic physical voxel spacing).
* **Group-Stratified Metrics:** Metrics are computed separately for FCD patients and Healthy Controls to distinguish lesion overlap from healthy false-positive rates.
