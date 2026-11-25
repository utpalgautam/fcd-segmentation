#!/usr/bin/env python3
"""Generate high-resolution visual comparison figures between ground truth and model predictions."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path("/home/utpal/.gemini/antigravity/brain/76809655-e0b9-4459-9c18-b3cb00ac43c8")
OUT_DIR = PROJECT_ROOT / "outputs" / "prediction_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "nnunet_results" / "Dataset101_FLAIR" / "FCDnnUNetTrainer_50epochs__nnUNetPlans_smoke__3d_fullres" / "fold_0" / "validation"
GT_DIR = PROJECT_ROOT / "data" / "nnunet_preprocessed" / "Dataset101_FLAIR" / "gt_segmentations"
IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "Dataset101_FLAIR" / "imagesTr"


def plot_subject_comparison(subject_id: str) -> Path:
    img_path = IMAGE_DIR / f"{subject_id}_0000.nii.gz"
    gt_path = GT_DIR / f"{subject_id}.nii.gz"
    pred_path = RESULTS_DIR / f"{subject_id}.nii.gz"

    if not img_path.exists() or not gt_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"Missing files for {subject_id}")

    img_nii = nib.load(str(img_path))
    gt_nii = nib.load(str(gt_path))
    pred_nii = nib.load(str(pred_path))

    img_data = img_nii.get_fdata()
    gt_data = (gt_nii.get_fdata() > 0).astype(np.uint8)
    pred_data = (pred_nii.get_fdata() > 0).astype(np.uint8)

    # Compute metrics
    tp = int(np.sum((pred_data == 1) & (gt_data == 1)))
    fp = int(np.sum((pred_data == 1) & (gt_data == 0)))
    fn = int(np.sum((pred_data == 0) & (gt_data == 1)))
    n_gt = int(np.sum(gt_data == 1))
    n_pred = int(np.sum(pred_data == 1))
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0

    # Find centroid of ground truth lesion (or prediction if GT empty)
    if n_gt > 0:
        com = ndimage.center_of_mass(gt_data)
    elif n_pred > 0:
        com = ndimage.center_of_mass(pred_data)
    else:
        com = [s // 2 for s in img_data.shape]

    cz, cy, cx = [int(round(c)) for c in com]
    # Bound within volume
    cz = max(0, min(img_data.shape[0] - 1, cz))
    cy = max(0, min(img_data.shape[1] - 1, cy))
    cx = max(0, min(img_data.shape[2] - 1, cx))

    # Normalize image intensities for display (0.5 to 99.5 percentile)
    fg_vals = img_data[img_data > 0]
    p_low, p_high = np.percentile(fg_vals, [1, 99]) if len(fg_vals) else (0, 1)
    img_norm = np.clip((img_data - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)

    # Views: Axial (cz), Coronal (cy), Sagittal (cx)
    views = [
        ("Axial View (Z={})".format(cz), np.rot90(img_norm[cz, :, :]), np.rot90(gt_data[cz, :, :]), np.rot90(pred_data[cz, :, :])),
        ("Coronal View (Y={})".format(cy), np.rot90(img_norm[:, cy, :]), np.rot90(gt_data[:, cy, :]), np.rot90(pred_data[:, cy, :])),
        ("Sagittal View (X={})".format(cx), np.rot90(img_norm[:, :, cx]), np.rot90(gt_data[:, :, cx]), np.rot90(pred_data[:, :, cx])),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5), facecolor="#0f172a")
    plt.subplots_adjust(wspace=0.03, hspace=0.15, top=0.91, bottom=0.04, left=0.04, right=0.96)

    title_text = (
        f"{subject_id} — Ground Truth vs. Model Prediction\n"
        f"Dice Score: {dice:.3f} ({dice*100:.1f}%) | IoU: {iou:.3f} | "
        f"True Positives: {tp:,} voxels | False Positives: {fp:,} | False Negatives: {fn:,}"
    )
    fig.suptitle(title_text, color="#f8fafc", fontsize=16, fontweight="bold", y=0.96)

    col_titles = [
        "1. FLAIR MRI (Input)",
        "2. Ground Truth Lesion (Cyan)",
        "3. Model Prediction (Red)",
        "4. Overlap (Yellow=TP, Cyan=FN, Red=FP)",
    ]

    for col, col_t in enumerate(col_titles):
        axes[0, col].set_title(col_t, color="#94a3b8", fontsize=12, pad=10, fontweight="semibold")

    for row, (view_name, img_slice, gt_slice, pred_slice) in enumerate(views):
        # 1. MRI alone
        ax = axes[row, 0]
        ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
        ax.set_ylabel(view_name, color="#e2e8f0", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # 2. MRI + GT
        ax = axes[row, 1]
        ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
        if np.any(gt_slice):
            masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
            ax.imshow(masked_gt, cmap=ListedColormap(["#06b6d4"]), alpha=0.65)
            # Outline
            contours = ndimage.binary_dilation(gt_slice) ^ gt_slice
            if np.any(contours):
                ax.imshow(np.ma.masked_where(~contours, contours), cmap=ListedColormap(["#22d3ee"]), alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])

        # 3. MRI + Pred
        ax = axes[row, 2]
        ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
        if np.any(pred_slice):
            masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
            ax.imshow(masked_pred, cmap=ListedColormap(["#ef4444"]), alpha=0.65)
            contours = ndimage.binary_dilation(pred_slice) ^ pred_slice
            if np.any(contours):
                ax.imshow(np.ma.masked_where(~contours, contours), cmap=ListedColormap(["#f87171"]), alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])

        # 4. Composite Overlap
        # 0 = BG, 1 = FN (GT only, cyan), 2 = FP (Pred only, red), 3 = TP (both, yellow)
        ax = axes[row, 3]
        ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
        comp = np.zeros_like(gt_slice, dtype=np.uint8)
        comp[(gt_slice == 1) & (pred_slice == 0)] = 1  # FN
        comp[(gt_slice == 0) & (pred_slice == 1)] = 2  # FP
        comp[(gt_slice == 1) & (pred_slice == 1)] = 3  # TP

        if np.any(comp > 0):
            # Custom cmap: 1 -> Cyan, 2 -> Red, 3 -> Yellow (#facc15)
            rgba = np.zeros((*comp.shape, 4), dtype=np.float32)
            rgba[comp == 1] = [0.02, 0.71, 0.83, 0.65]  # FN (Cyan)
            rgba[comp == 2] = [0.93, 0.27, 0.27, 0.55]  # FP (Red)
            rgba[comp == 3] = [0.98, 0.80, 0.08, 0.85]  # TP (Yellow)
            ax.imshow(rgba)

        ax.set_xticks([])
        ax.set_yticks([])

    # Legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor="#facc15", edgecolor="#ca8a04", label="True Positive (Overlap: Both Ground Truth & Model Agreed)"),
        mpatches.Patch(facecolor="#06b6d4", edgecolor="#0891b2", label="False Negative (Missed: Ground Truth only)"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="#dc2626", label="False Positive (Over-segmented: Model only)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=11,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#f1f5f9",
        bbox_to_anchor=(0.5, 0.01),
    )

    out_file = OUT_DIR / f"{subject_id}_prediction_vs_gt.png"
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # Also copy to artifact dir
    artifact_file = ARTIFACT_DIR / f"{subject_id}_prediction_vs_gt.png"
    artifact_file.write_bytes(out_file.read_bytes())
    print(f"Generated: {out_file} and {artifact_file}")
    return out_file


def generate_html_report(subjects: list[str]) -> Path:
    html_cards = []
    for sid in subjects:
        png_path = OUT_DIR / f"{sid}_prediction_vs_gt.png"
        if not png_path.exists():
            continue
        b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
        html_cards.append(f"""
        <div class="card">
            <h2>{sid} — Ground Truth vs. Prediction</h2>
            <img src="data:image/png;base64,{b64}" alt="{sid} prediction visualization" />
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FCD Segmentation — Model Prediction vs Ground Truth</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #090d16;
        color: #f1f5f9;
        margin: 0;
        padding: 24px;
    }}
    h1 {{
        text-align: center;
        color: #38bdf8;
        font-size: 28px;
        margin-bottom: 8px;
    }}
    .subtitle {{
        text-align: center;
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 32px;
    }}
    .card {{
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 28px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    .card h2 {{
        margin-top: 0;
        color: #facc15;
        font-size: 20px;
    }}
    .card img {{
        width: 100%;
        height: auto;
        border-radius: 8px;
        border: 1px solid #374151;
    }}
</style>
</head>
<body>
    <h1>FCD Lesion Segmentation — Prediction vs Ground Truth Gallery</h1>
    <p class="subtitle">Dataset 101 (FLAIR) | Fold 0 Validation Cohort | nnU-Net 3D Full-Resolution</p>
    {''.join(html_cards)}
</body>
</html>
"""
    html_file = OUT_DIR / "prediction_comparison_report.html"
    html_file.write_text(html_content, encoding="utf-8")
    artifact_html = ARTIFACT_DIR / "prediction_comparison_report.html"
    artifact_html.write_text(html_content, encoding="utf-8")
    print(f"Generated HTML report: {html_file}")
    return html_file


def main():
    test_cases = ["sub-00055", "sub-00003", "sub-00123", "sub-00040", "sub-00001"]
    for sid in test_cases:
        try:
            plot_subject_comparison(sid)
        except Exception as e:
            print(f"Failed {sid}: {e}")
    generate_html_report(test_cases)


if __name__ == "__main__":
    main()
