#!/usr/bin/env python3
"""
CLI entry point for the FCD MRI Preprocessing Pipeline.

Creates three synchronized nnU-Net v2 datasets (FLAIR, T1, Fusion)
with embedded HTML QC reports and manifests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path so package imports resolve seamlessly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fcd_segmentation.preprocessing.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FCD MRI Preprocessing Pipeline: creates FLAIR, T1, and Fusion nnU-Net v2 datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "--data-root",
        dest="input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dataset",
        help="Path to raw cohort directory containing sub-*/anat/",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Path to output processed directory for nnU-Net datasets",
    )
    parser.add_argument(
        "--reports",
        dest="reports",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprocessing",
        help="Path to directory for HTML QC reports, manifests, and logs",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing_config.json",
        help="Path to preprocessing JSON configuration file",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Run input discovery and inventory report without processing images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of subjects to process (for development and smoke testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip subjects that already have complete, validated outputs matching current inputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed datasets and QC reports",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop pipeline on first subject failure instead of logging and continuing",
    )
    parser.add_argument(
        "--keep-failed-intermediates",
        action="store_true",
        help="Retain intermediate working directory for debugging failed subjects",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Override computing device for HD-BET skull stripping",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Validate inputs
    if not args.config.is_file():
        parser.error(f"Configuration file not found: {args.config}")
    if not args.input.is_dir():
        parser.error(f"Input cohort directory not found: {args.input}")

    try:
        results = run_pipeline(
            config_path=args.config,
            raw_data_dir=args.input,
            processed_dir=args.output,
            reports_dir=args.reports,
            inventory_only=args.inventory_only,
            limit=args.limit,
            resume=args.resume,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
            keep_failed_intermediates=args.keep_failed_intermediates,
            device=args.device,
        )

        if args.inventory_only:
            print("\nInventory-only run completed successfully.")
            print(f"Inventory HTML: {results.get('inventory_html')}")
            return 0

        summary = results.get("summary", {})
        print("\n" + "=" * 60)
        print("PREPROCESSING EXECUTION FINISHED")
        print("=" * 60)
        print(f"Attempted: {summary.get('attempted', 0)}")
        print(f"Passed:    {summary.get('passed', 0)}")
        print(f"Warned:    {summary.get('warned', 0)}")
        print(f"Skipped:   {summary.get('skipped', 0)}")
        print(f"Failed:    {summary.get('failed', 0)}")
        print(f"Reports:   {args.reports / 'qc'}")
        print(f"Manifests: {args.reports / 'manifests'}")
        print("=" * 60)

        return 1 if summary.get("failed", 0) > 0 and args.fail_fast else 0

    except Exception as exc:
        print(f"\nFATAL PIPELINE ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
