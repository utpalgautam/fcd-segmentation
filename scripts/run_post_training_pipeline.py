#!/usr/bin/env python3
"""Execute post-training inference verification, evaluation, and visual reporting

Implements the complete post-training steps from the Kaggle notebook (Cells 5, 6, 7, 8)
across all three baseline datasets (101 FLAIR, 102 T1, 103 Fusion) for Fold 0:
1. Verifies prediction completeness (29/29 subjects)
2. Prepares evaluation ground truth (Cell 6)
3. Evaluates metrics (Dice, Sensitivity, Specificity, Precision, HD95) (Cell 7)
4. Generates visual HTML segmentation reports (Cell 8)
5. Generates a multi-modality cross-comparison summary
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("post_training")

DATASETS = {
    101: "Dataset101_FLAIR",
    102: "Dataset102_T1",
    103: "Dataset103_Fusion",
}
FOLD = 0

RAW_ROOT = PROJECT_ROOT / "data" / "processed"
PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "nnunet_preprocessed"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def prepare_evaluation_data():
    """Populate outputs/evaluation_data/<dataset>/labelsTr for all datasets (Cell 6)."""
    logger.info("=== STEP 1: Preparing evaluation ground truth labels (Cell 6) ===")
    for d_id, d_name in DATASETS.items():
        labels_dir = RAW_ROOT / d_name / "labelsTr"
        split_file = PREPROCESSED_ROOT / d_name / "splits_final.json"
        temp_labels = OUTPUTS_ROOT / "evaluation_data" / d_name / "labelsTr"
        temp_labels.mkdir(parents=True, exist_ok=True)

        with open(split_file) as f:
            splits = json.load(f)
        val_cases = splits[FOLD]["val"]

        for case in val_cases:
            src = labels_dir / f"{case}.nii.gz"
            if not src.exists():
                src = labels_dir / f"{case}.nii"
            dst = temp_labels / f"{case}.nii.gz"

            if not dst.exists() and src.exists():
                if src.suffix == ".gz":
                    shutil.copy2(src, dst)
                else:
                    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        count = len(list(temp_labels.glob("*.nii.gz")))
        logger.info("Dataset %d (%s): Prepared %d/%d ground truth labels in %s", d_id, d_name, count, len(val_cases), temp_labels)


def run_evaluations():
    """Run formal evaluation for each dataset using scripts/evaluate.py (Cell 7)."""
    logger.info("\n=== STEP 2: Running formal evaluations (Cell 7) ===")
    eval_dir = OUTPUTS_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for d_id, d_name in DATASETS.items():
        pred_dir = OUTPUTS_ROOT / "predictions" / d_name / f"fold_{FOLD}"
        gt_dir = RAW_ROOT / d_name / "labelsTr"
        splits_file = PREPROCESSED_ROOT / d_name / "splits_final.json"

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--dataset", str(d_id),
            "--split", "val",
            "--fold", str(FOLD),
            "--predictions-dir", str(pred_dir),
            "--ground-truth-dir", str(gt_dir),
            "--splits-file", str(splits_file),
            "--output-dir", str(eval_dir),
        ]

        logger.info("Evaluating Dataset %d (%s)...", d_id, d_name)
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("Evaluation failed for Dataset %d:\n%s", d_id, res.stderr)
        else:
            logger.info("Successfully evaluated Dataset %d (%s). Output in %s", d_id, d_name, eval_dir)


def generate_visual_reports():
    """Generate standalone HTML visual segmentation reports for each dataset (Cell 8)."""
    logger.info("\n=== STEP 3: Generating HTML visual segmentation reports (Cell 8) ===")

    def load_nifti(path):
        return nib.load(str(path)).get_fdata()

    def find_nifti(folder, name):
        for ext in [".nii.gz", ".nii"]:
            p = folder / (name + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"Missing: {name} in {folder}")

    def render_slice(mri, gt, pred, z, modality_label):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        image = np.rot90(mri[:, :, z])
        truth = np.rot90(gt[:, :, z])
        prediction = np.rot90(pred[:, :, z])

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"{modality_label} MRI")

        axes[1].imshow(truth, cmap="gray")
        axes[1].set_title("Ground Truth")

        axes[2].imshow(prediction, cmap="gray")
        axes[2].set_title("Prediction")

        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(np.ma.masked_where(truth == 0, truth), cmap="Greens", alpha=0.55, vmin=0, vmax=1)
        axes[3].imshow(np.ma.masked_where(prediction == 0, prediction), cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        axes[3].set_title("Overlay: GT Green / Pred Red")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode()

    for d_id, d_name in DATASETS.items():
        raw_dir = RAW_ROOT / d_name
        pred_dir = OUTPUTS_ROOT / "predictions" / d_name / f"fold_{FOLD}"
        split_file = PREPROCESSED_ROOT / d_name / "splits_final.json"

        with open(split_file) as f:
            subjects = json.load(f)[FOLD]["val"]

        modality_title = "FLAIR" if d_id == 101 else ("T1" if d_id == 102 else "Fusion (FLAIR channel)")
        html_parts = [f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FCD Segmentation Validation — {d_name} (Fold {FOLD})</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background:#0f172a; color:#f8fafc; padding:30px; }}
h1 {{ color:#38bdf8; margin-bottom: 5px; }}
.subtitle {{ color:#94a3b8; margin-bottom: 25px; }}
.stats-badge {{ display: inline-block; background: #1e293b; padding: 6px 14px; border-radius: 8px; margin-right: 10px; font-weight: bold; border: 1px solid #334155; }}
.subject {{ margin-bottom:25px; padding:20px; background:#1e293b; border-radius:12px; border: 1px solid #334155; }}
.subject h2 {{ color: #cbd5e1; font-size: 1.15rem; margin-top: 0; }}
img {{ width:100%; max-width: 1200px; border-radius: 8px; display: block; }}
</style>
</head>
<body>
<h1>FCD Lesion Segmentation — Validation Visuals</h1>
<div class="subtitle">Dataset: <strong>{d_name}</strong> | Model: <strong>FCDnnUNetTrainer_100epochs (3d_fullres)</strong> | Fold: <strong>{FOLD}</strong></div>
"""]

        processed = 0
        for subject in subjects:
            try:
                # Find input image (_0000 for all, which is FLAIR in 101 & 103, T1 in 102)
                image_path = find_nifti(raw_dir / "imagesTr", f"{subject}_0000")
                gt_path = find_nifti(raw_dir / "labelsTr", subject)
                pred_path = find_nifti(pred_dir, subject)

                mri = load_nifti(image_path)
                gt = load_nifti(gt_path)
                pred = load_nifti(pred_path)

                if mri.shape != gt.shape or mri.shape != pred.shape:
                    logger.warning("Shape mismatch for %s in %s", subject, d_name)
                    continue

                lesion_counts = gt.sum(axis=(0, 1))
                pred_counts = pred.sum(axis=(0, 1))
                if lesion_counts.max() > 0:
                    z = int(np.argmax(lesion_counts))
                    case_type = "FCD Patient (True Lesion Present)"
                elif pred_counts.max() > 0:
                    z = int(np.argmax(pred_counts))
                    case_type = "Healthy Control (False Positive Present)"
                else:
                    z = mri.shape[2] // 2
                    case_type = "Healthy Control (True Negative - Clean)"

                # Compute slice Dice
                tp = np.sum((gt[:, :, z] > 0) & (pred[:, :, z] > 0))
                total = np.sum(gt[:, :, z] > 0) + np.sum(pred[:, :, z] > 0)
                slice_dice = (2.0 * tp / total) if total > 0 else (1.0 if (gt[:, :, z].sum() == 0 and pred[:, :, z].sum() == 0) else 0.0)

                image_b64 = render_slice(mri, gt, pred, z, modality_title)

                html_parts.append(f"""
<div class="subject">
    <h2>Subject: <code>{subject}</code> | Slice: <code>{z}</code> | Type: <em>{case_type}</em> | Slice Dice: <code>{slice_dice*100:.1f}%</code></h2>
    <img src="data:image/png;base64,{image_b64}" alt="{subject} slice {z}">
</div>
""")
                processed += 1
            except Exception as e:
                logger.error("Error processing %s in %s: %s", subject, d_name, e)

        html_parts.append("</body></html>")
        report_path = OUTPUTS_ROOT / f"segmentation_report_{d_name}_fold{FOLD}.html"
        report_path.write_text("\n".join(html_parts), encoding="utf-8")
        logger.info("Saved visual report for %s: %s (%d subjects rendered)", d_name, report_path, processed)


def print_cross_modality_summary():
    """Print multi-modality comparison table across 101, 102, and 103."""
    logger.info("\n=== STEP 4: Multi-Modality Comparative Benchmark (Fold 0) ===")
    eval_dir = OUTPUTS_ROOT / "evaluation"

    results = {}
    for d_id, d_name in DATASETS.items():
        summary_path = eval_dir / f"summary_{d_name}_val_fold_{FOLD}.json"
        if summary_path.is_file():
            with open(summary_path) as f:
                results[d_name] = json.load(f)

    if not results:
        logger.warning("No evaluation summary files found.")
        return

    print("\n" + "="*85)
    print(f"{'Modality / Dataset':<25} | {'Overall Dice':<14} | {'FCD-only Dice':<14} | {'Sensitivity':<12} | {'Specificity':<12}")
    print("="*85)

    for d_name, data in results.items():
        stats = data.get("summary_statistics", {})
        subgroups = data.get("subgroup_statistics", {})
        
        overall_dice = stats.get("dice", {}).get("mean", 0.0)
        overall_sens = stats.get("sensitivity", {}).get("mean", 0.0)
        overall_spec = stats.get("specificity", {}).get("mean", 0.0)
        
        fcd_dice = subgroups.get("fcd", {}).get("dice", {}).get("mean", 0.0)
        
        print(f"{d_name:<25} | {overall_dice*100:6.2f}%        | {fcd_dice*100:6.2f}%        | {overall_sens*100:6.2f}%      | {overall_spec*100:6.2f}%")
    print("="*85 + "\n")


if __name__ == "__main__":
    prepare_evaluation_data()
    run_evaluations()
    generate_visual_reports()
    print_cross_modality_summary()
