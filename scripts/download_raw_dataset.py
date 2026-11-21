#!/usr/bin/env python3
"""
Download the public raw OpenNeuro BIDS dataset used by this project.

Default:
    Dataset: ds004199
    Output:  <project-root>/data/raw/dataset

This downloads RAW OpenNeuro data only. It does not run preprocessing,
create nnU-Net datasets, or start training.

The downloader is openneuro-py, a Python client for public OpenNeuro datasets.
No OpenNeuro account/API-key login is required for public datasets.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_DEFAULT = "ds004199"
EXPECTED_PARTICIPANTS = 170


def project_root() -> Path:
    # This file is expected at <project-root>/scripts/download_raw_dataset.py
    return Path(__file__).resolve().parent.parent


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the raw public OpenNeuro BIDS dataset (no login)."
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_DEFAULT,
        help=f"OpenNeuro accession (default: {DATASET_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "data" / "raw" / "dataset",
        help="Output directory (default: <project-root>/data/raw/dataset)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Do not check participants.tsv after download.",
    )
    args = parser.parse_args()

    if not args.dataset.startswith("ds") or not args.dataset[2:].isdigit():
        parser.error(f"Invalid OpenNeuro dataset accession: {args.dataset}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Avoid mixing the BIDS download with unrelated files.
    if output.exists() and any(output.iterdir()):
        has_bids = (output / "dataset_description.json").is_file()
        if not has_bids:
            print(
                f"ERROR: Output directory exists and is non-empty, but does not "
                f"look like a BIDS dataset:\n  {output}\n"
                "Choose another --output directory or inspect/move the existing "
                "contents. Nothing was deleted.",
                file=sys.stderr,
            )
            return 2
        print(f"Existing BIDS directory found; downloader may resume: {output}")

    # openneuro-py is a Python client with a CLI entry point. Public datasets
    # can be downloaded without an OpenNeuro API key.
    downloader = shutil.which("openneuro-py")
    if downloader is None:
        print(
            "ERROR: openneuro-py is not installed.\n"
            "Install it in this Python environment with:\n\n"
            f"  {sys.executable} -m pip install openneuro-py\n",
            file=sys.stderr,
        )
        return 3

    output.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {args.dataset}")
    print(f"Output  : {output}")
    print("Mode    : raw BIDS download only (no preprocessing/training)")
    print("\nDownloading dataset. This may take time and disk space.", flush=True)

    try:
        run(
            [
                downloader,
                "download",
                f"--dataset={args.dataset}",
                f"--target-dir={output}",
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"\nDownload command failed with exit code {exc.returncode}. "
            "You can rerun this script to continue the download.",
            file=sys.stderr,
        )
        return exc.returncode or 1

    if not args.skip_verification:
        participants = output / "participants.tsv"
        description = output / "dataset_description.json"

        missing = [
            str(path)
            for path in (participants, description)
            if not path.is_file()
        ]
        if missing:
            print(
                "\nERROR: Download finished but required BIDS metadata is missing:\n"
                + "\n".join(f"  {item}" for item in missing),
                file=sys.stderr,
            )
            return 4

        with participants.open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))

        groups: dict[str, int] = {}
        for row in rows:
            group = (row.get("group") or "UNKNOWN").strip().lower()
            groups[group] = groups.get(group, 0) + 1

        print("\nBIDS metadata verification:")
        print(f"  participants.tsv rows: {len(rows)}")
        print(f"  group counts: {groups}")

        if len(rows) != EXPECTED_PARTICIPANTS:
            print(
                f"WARNING: This project previously expected "
                f"{EXPECTED_PARTICIPANTS} participants. Check the downloaded "
                "dataset snapshot/version before using it.",
                file=sys.stderr,
            )

    print("\nRAW DATA DOWNLOAD FINISHED")
    print(f"Location: {output}")
    print("\nThis script has NOT downloaded or generated:")
    print("  - data/nnunet_preprocessed/Dataset101_FLAIR")
    print("  - data/nnunet_preprocessed/Dataset102_T1")
    print("  - data/nnunet_preprocessed/Dataset103_Fusion")
    print("It has NOT run preprocessing or model training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
