import json
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import label

pred_dir = "outputs/nnunet_results/Dataset101_FLAIR/FCDnnUNetTrainer_50epochs__nnUNetPlans_smoke__3d_fullres/fold_0/validation"
gt_dir = "data/processed/Dataset101_FLAIR/labelsTr"
with open(os.path.join(pred_dir, "summary.json")) as f:
    summary = json.load(f)

fcd_cases = []
hc_cases = []

for item in summary["metric_per_case"]:
    p = item["prediction_file"]
    cid = os.path.basename(p).replace(".nii.gz", "")
    dice = item["metrics"]["1"]["Dice"]
    fp = item["metrics"]["1"]["FP"]
    tp = item["metrics"]["1"]["TP"]
    fn = item["metrics"]["1"]["FN"]
    is_hc = (tp + fn) == 0
    record = {"id": cid, "dice": dice, "tp": tp, "fp": fp, "fn": fn, "is_hc": is_hc}
    if is_hc:
        hc_cases.append(record)
    else:
        fcd_cases.append(record)

print(f"Total validation cases: {len(summary['metric_per_case'])} (FCD: {len(fcd_cases)}, HC: {len(hc_cases)})")

post_results = []
for c in sorted(fcd_cases, key=lambda x: x["dice"], reverse=True):
    cid = c["id"]
    pred_path = os.path.join(pred_dir, cid + ".nii.gz")
    gt_path = os.path.join(gt_dir, cid + ".nii.gz")
    pred = nib.load(pred_path).get_fdata() > 0
    gt = nib.load(gt_path).get_fdata() > 0
    
    labeled, num_comp = label(pred)
    if num_comp > 0:
        sizes = [int(np.sum(labeled == i)) for i in range(1, num_comp + 1)]
        largest_idx = int(np.argmax(sizes) + 1)
        largest_mask = (labeled == largest_idx)
        
        tp_post = int(np.sum(largest_mask & gt))
        fp_post = int(np.sum(largest_mask & ~gt))
        fn_post = int(np.sum(~largest_mask & gt))
        dice_post = (2.0 * tp_post) / (2.0 * tp_post + fp_post + fn_post) if (2.0 * tp_post + fp_post + fn_post) > 0 else 0.0
        
        overlaps_gt = tp_post > 0
        
        post_results.append({
            "id": cid, "raw_dice": c["dice"], "post_dice": dice_post,
            "raw_fp": c["fp"], "post_fp": fp_post,
            "num_clusters": num_comp, "largest_cluster": sizes[largest_idx-1],
            "gt_size": int(np.sum(gt)), "overlaps_gt": overlaps_gt
        })

print("\n--- Validation FCD Cases: Raw vs Largest-Component Post-Processing ---")
for r in post_results:
    cid = r["id"]
    print(f"{cid}: Raw Dice={r['raw_dice']*100:5.2f}% -> Post Dice={r['post_dice']*100:5.2f}% | FP: {r['raw_fp']:5d} -> {r['post_fp']:5d} | Clusters: {r['num_clusters']:2d} | Overlaps GT: {r['overlaps_gt']}")

raw_dices = [r["raw_dice"] for r in post_results]
post_dices = [r["post_dice"] for r in post_results]
raw_fps = [r["raw_fp"] for r in post_results]
post_fps = [r["post_fp"] for r in post_results]

print(f"\nSummary across all {len(post_results)} FCD cases:")
print(f"Mean FP: {np.mean(raw_fps):.1f} voxels -> {np.mean(post_fps):.1f} voxels (Reduction: {100*(1 - np.mean(post_fps)/np.mean(raw_fps)):.1f}%)")
print(f"Mean Dice: {np.mean(raw_dices)*100:.2f}% -> {np.mean(post_dices)*100:.2f}%")
