"""nnU-Net v2 environment path resolution and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NNUNetPaths:
    project_root: Path
    raw: Path
    preprocessed: Path
    results: Path

    def to_env_dict(self) -> dict[str, str]:
        """Return environment variable dictionary for subprocesses."""
        return {
            "nnUNet_raw": str(self.raw.resolve()),
            "nnUNet_preprocessed": str(self.preprocessed.resolve()),
            "nnUNet_results": str(self.results.resolve()),
        }


def get_default_project_root() -> Path:
    """Return project root directory."""
    # src/fcd_segmentation/training/paths.py -> 4 levels up
    return Path(__file__).resolve().parents[3]


def resolve_nnunet_paths(
    project_root: Path | None = None,
    raw_dir: Path | str | None = None,
    preprocessed_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> NNUNetPaths:
    """Resolve nnU-Net environment paths with precedence:

    1. Explicit arguments (if provided)
    2. Existing environment variables (nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
    3. Repository-relative defaults:
       - nnUNet_raw: data/processed
       - nnUNet_preprocessed: data/nnunet_preprocessed
       - nnUNet_results: outputs/nnunet_results
    """
    root = (project_root or get_default_project_root()).resolve()

    # Raw
    if raw_dir is not None:
        raw = Path(raw_dir)
    elif "nnUNet_raw" in os.environ and os.environ["nnUNet_raw"].strip():
        raw = Path(os.environ["nnUNet_raw"])
    else:
        raw = root / "data" / "processed"

    # Preprocessed
    if preprocessed_dir is not None:
        preprocessed = Path(preprocessed_dir)
    elif (
        "nnUNet_preprocessed" in os.environ
        and os.environ["nnUNet_preprocessed"].strip()
    ):
        preprocessed = Path(os.environ["nnUNet_preprocessed"])
    else:
        preprocessed = root / "data" / "nnunet_preprocessed"

    # Results
    if results_dir is not None:
        results = Path(results_dir)
    elif "nnUNet_results" in os.environ and os.environ["nnUNet_results"].strip():
        results = Path(os.environ["nnUNet_results"])
    else:
        results = root / "outputs" / "nnunet_results"

    # Make absolute if relative
    if not raw.is_absolute():
        raw = (root / raw).resolve()
    else:
        raw = raw.resolve()

    if not preprocessed.is_absolute():
        preprocessed = (root / preprocessed).resolve()
    else:
        preprocessed = preprocessed.resolve()

    if not results.is_absolute():
        results = (root / results).resolve()
    else:
        results = results.resolve()

    return NNUNetPaths(
        project_root=root,
        raw=raw,
        preprocessed=preprocessed,
        results=results,
    )


def ensure_nnunet_directories(paths: NNUNetPaths) -> None:
    """Ensure preprocessed and results directories exist."""
    paths.preprocessed.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)
