import os
import json
import shutil
import subprocess
from pathlib import Path

DATASET_NAME = "Dataset103_Fusion"
DATASET_ID = 103
FOLD = 0

split_file = Path("data/nnunet_preprocessed") / DATASET_NAME / "splits_final.json"
with open(split_file) as f:
    splits = json.load(f)
val_cases = splits[FOLD]["val"]

pred_dir = Path("outputs/predictions") / DATASET_NAME / f"fold_{FOLD}"
existing_preds = {f.name.replace(".nii.gz", "") for f in pred_dir.glob("*.nii.gz")}
missing_cases = [c for c in val_cases if c not in existing_preds]

print(f"Total validation cases: {len(val_cases)}")
print(f"Existing predictions: {len(existing_preds)}")
print(f"Missing cases to predict: {len(missing_cases)} -> {missing_cases}")

if missing_cases:
    temp_input = Path("outputs/validation_input") / DATASET_NAME / f"fold_{FOLD}"
    temp_input.mkdir(parents=True, exist_ok=True)
    images_dir = Path("data/processed") / DATASET_NAME / "imagesTr"
    
    for case in missing_cases:
        for ch in ["0000", "0001"]:
            src = images_dir / f"{case}_{ch}.nii.gz"
            dst = temp_input / f"{case}_{ch}.nii.gz"
            if not dst.exists():
                shutil.copy2(src, dst)
    
    input_count = len(list(temp_input.glob("*.nii.gz")))
    print(f"Prepared {input_count} channel files in {temp_input}")
    
    cmd = [
        ".venv/bin/nnUNetv2_predict",
        "-i", str(temp_input),
        "-o", str(pred_dir),
        "-d", str(DATASET_ID),
        "-c", "3d_fullres",
        "-f", str(FOLD),
        "-tr", "FCDnnUNetTrainer_100epochs",
        "-p", "nnUNetPlans",
        "-chk", "checkpoint_best.pth",
        "-device", "cuda"
    ]
    
    env = os.environ.copy()
    env["nnUNet_raw"] = str(Path("data/processed").resolve())
    env["nnUNet_preprocessed"] = str(Path("data/nnunet_preprocessed").resolve())
    env["nnUNet_results"] = str(Path("outputs/nnunet_results").resolve())
    env["nnUNet_extTrainer"] = str(Path("src/fcd_segmentation/training").resolve())
    
    print("Launching nnUNetv2_predict for remaining cases...")
    res = subprocess.run(cmd, env=env)
    print("Inference finished with code:", res.returncode)

final_preds = list(pred_dir.glob("*.nii.gz"))
print(f"Final predictions in {pred_dir}: {len(final_preds)} / {len(val_cases)}")
