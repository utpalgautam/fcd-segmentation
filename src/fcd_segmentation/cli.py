"""Command-line entry points."""
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

def check_environment_main():
    argparse.ArgumentParser(description="Check FCD development environment").parse_args()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    for package in ("numpy", "PyYAML", "nibabel", "torch"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: not installed (optional or missing)")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("CUDA status: unknown (PyTorch not installed)")
    return 0

def audit_dataset_main():
    parser = argparse.ArgumentParser(description="Inventory FLAIR NIfTI files")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/flair_inventory.csv"))
    args = parser.parse_args()
    from fcd_segmentation.data.discovery import discover_flair_files
    import csv
    files = discover_flair_files(args.data_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "flair_path"])
        for path in files:
            writer.writerow([path.parents[1].name, str(path.resolve())])
    print(f"Found {len(files)} FLAIR image(s); wrote {args.output_csv}")
    return 0
