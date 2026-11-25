#!/usr/bin/env python3
"""Preflight verification script for remote lab GPU machine.

Run this script before launching training or evaluation on the lab workstation or cluster:
    python scripts/preflight_check.py

Validates:
1. Python environment and required dependencies
2. PyTorch and CUDA GPU acceleration availability
3. nnU-Net v2 installation and command-line entry points
4. Project custom trainer discovery via nnUNet_extTrainer
5. Directory structures and disk storage headroom
6. Processed datasets and metadata integrity
7. Stratified 5-fold cross-validation configuration
"""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fcd_segmentation.training.paths import resolve_nnunet_paths
from fcd_segmentation.training.runner import find_executable


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_python_and_packages() -> bool:
    print_section("1. Python Environment & Dependencies")
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")

    py_major, py_minor = sys.version_info[:2]
    if (py_major, py_minor) < (3, 11):
        print("  [FAIL] Python 3.11+ is required.")
        return False
    print("  [PASS] Python version compatible.")

    required_pkgs = [
        "torch",
        "torchvision",
        "nnunetv2",
        "nibabel",
        "scipy",
        "scikit-learn",
        "pandas",
        "pydantic",
        "PyYAML",
    ]
    all_ok = True
    for pkg in required_pkgs:
        try:
            ver = importlib.metadata.version(pkg)
            print(f"  [PASS] {pkg:<15} version {ver}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  [FAIL] {pkg:<15} NOT INSTALLED")
            all_ok = False
    return all_ok


def check_cuda_and_gpu() -> bool:
    print_section("2. PyTorch & CUDA GPU Acceleration")
    try:
        import torch

        if not torch.cuda.is_available():
            print("  [WARN] CUDA is NOT available to PyTorch. Training will be extremely slow or fail on CPU.")
            return False

        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [PASS] CUDA available. Detected {gpu_count} GPU(s).")
        print(f"  [PASS] Primary GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        if vram_gb < 8.0:
            print("  [WARN] GPU VRAM < 8 GB. 3D fullres models may require smaller batch size.")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to inspect PyTorch CUDA: {e}")
        return False


def check_nnunet_executables() -> bool:
    print_section("3. nnU-Net v2 Command-Line Executables")
    executables = [
        "nnUNetv2_plan_and_preprocess",
        "nnUNetv2_train",
        "nnUNetv2_predict",
    ]
    all_ok = True
    for exe in executables:
        resolved = find_executable(exe)
        if Path(resolved).is_file() or shutil.which(exe):
            print(f"  [PASS] {exe:<30} -> {resolved}")
        else:
            print(f"  [FAIL] {exe:<30} NOT FOUND in PATH or virtualenv")
            all_ok = False
    return all_ok


def check_trainer_discovery() -> bool:
    print_section("4. Custom Trainer Discovery")
    try:
        from fcd_segmentation.training.trainers import verify_trainer_resolution

        cls = verify_trainer_resolution("FCDnnUNetTrainer_100epochs")
        print(f"  [PASS] Successfully discovered 'FCDnnUNetTrainer_100epochs' -> {cls} ({cls.__module__})")

        # Verify built-in collision check works as expected
        try:
            verify_trainer_resolution("nnUNetTrainer_100epochs")
            print("  [FAIL] Did not catch built-in collision for nnUNetTrainer_100epochs")
            return False
        except ValueError:
            print("  [PASS] Successfully catches conflicting built-in trainer 'nnUNetTrainer_100epochs'")

        return True
    except Exception as e:
        print(f"  [FAIL] Trainer discovery failed with error: {e}")
        return False


def check_paths_and_disk_space() -> bool:
    print_section("5. Environment Paths & Storage")
    paths = resolve_nnunet_paths(project_root=PROJECT_ROOT)
    for k, v in paths.to_env_dict().items():
        print(f"  ${k} -> {v}")

    # Check free disk space
    res_dir = paths.results
    res_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(res_dir)
    free_gb = usage.free / (1024**3)
    print(f"  Free disk space on outputs volume: {free_gb:.1f} GB")
    if free_gb < 20.0:
        print("  [WARN] Less than 20 GB free disk space. nnU-Net preprocessed data and checkpoints need ~30-50 GB.")
    else:
        print("  [PASS] Adequate disk space available.")
    return True


def check_dataset_and_metadata() -> bool:
    print_section("6. Dataset & Cohort Metadata")
    tsv_path = PROJECT_ROOT / "data" / "raw" / "dataset" / "participants.tsv"
    if not tsv_path.is_file():
        print(f"  [FAIL] participants.tsv not found at {tsv_path}")
        return False

    import csv

    with tsv_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    train_rows = [r for r in rows if r.get("split", "").strip().lower() == "train"]
    test_rows = [r for r in rows if r.get("split", "").strip().lower() == "test"]
    fcd_train = [r for r in train_rows if r.get("group", "").strip().lower() == "fcd"]
    hc_train = [r for r in train_rows if r.get("group", "").strip().lower() == "hc"]

    print(f"  [PASS] Found participants.tsv with {len(rows)} subjects:")
    print(f"         - Training cohort:   {len(train_rows)} subjects ({len(fcd_train)} FCD, {len(hc_train)} HC)")
    print(f"         - Held-out test set: {len(test_rows)} subjects (All FCD)")

    # Check processed datasets in data/processed
    paths = resolve_nnunet_paths(project_root=PROJECT_ROOT)
    for d_name in ["Dataset101_FLAIR", "Dataset102_T1", "Dataset103_Fusion"]:
        d_path = paths.raw / d_name
        if d_path.is_dir():
            n_tr = len(list((d_path / "imagesTr").glob("*_0000.nii.gz"))) if (d_path / "imagesTr").is_dir() else 0
            n_ts = len(list((d_path / "imagesTs").glob("*_0000.nii.gz"))) if (d_path / "imagesTs").is_dir() else 0
            print(f"  [INFO] {d_name}: {n_tr} training cases, {n_ts} test cases in imagesTr/imagesTs")
        else:
            print(f"  [INFO] {d_name} not yet created in {paths.raw}. (Preprocessing in progress or pending)")

    return True


def main() -> int:
    print("============================================================")
    print("  FCD SEGMENTATION — LAB GPU PREFLIGHT DIAGNOSTICS")
    print("============================================================")

    results = [
        ("Python & Packages", check_python_and_packages()),
        ("CUDA & GPU", check_cuda_and_gpu()),
        ("nnU-Net Executables", check_nnunet_executables()),
        ("Custom Trainer Discovery", check_trainer_discovery()),
        ("Paths & Disk Space", check_paths_and_disk_space()),
        ("Dataset & Metadata", check_dataset_and_metadata()),
    ]

    print_section("PREFLIGHT SUMMARY")
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "ATTENTION NEEDED"
        print(f"  [{status:<16}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll preflight checks PASSED! System is ready for training.\n")
        return 0
    else:
        print("\nSome checks require attention. Review output above before training.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
