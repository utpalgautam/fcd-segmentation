"""Registration boundary for the FCD preprocessing package.

Resampling is not registration: this module intentionally does not claim to
align anatomy. Implement/validate registration against the project's chosen
reference modality and dataset before using multimodal inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def register_to_reference(
    moving_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "rigid",
    **kwargs: Any,
) -> Path:
    """Explicitly fail rather than silently pretending to register images.

    The repository README identifies registration as incomplete. Choose and
    validate a registration implementation (for example, ANTs or SimpleITK)
    before enabling this function in a pipeline.
    """
    raise NotImplementedError(
        "Registration is not implemented. Resampling only aligns voxel grids; "
        "it does not anatomically register MRI scans."
    )
