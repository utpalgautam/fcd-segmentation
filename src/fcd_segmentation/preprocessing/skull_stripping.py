from __future__ import annotations

import csv
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from fcd_segmentation.preprocessing.geometry import (
    validate_image_geometry,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SUPPORTED_DEVICES = {"cuda", "cpu", "mps"}

# Conservative screening limits for QC, not diagnostic thresholds.
MIN_BRAIN_FRACTION = 0.05
MAX_BRAIN_FRACTION = 0.25

MANIFEST_NAME = "hd_bet_manifest.json"


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def _find_hd_bet_executable() -> str:
    """Return the installed HD-BET executable or raise an error."""
    executable = shutil.which("hd-bet")

    if executable is None:
        venv_candidate = Path(sys.executable).parent / "hd-bet"
        if venv_candidate.is_file() and os.access(venv_candidate, os.X_OK):
            executable = str(venv_candidate)

    if executable is None:
        raise RuntimeError(
            "HD-BET was not found on PATH. Activate the project "
            "virtual environment and install hd-bet==2.0.1."
        )

    return executable


def _load_image(path: str | Path) -> nib.spatialimages.SpatialImage:
    """Load and validate a three-dimensional NIfTI image."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    image = nib.load(str(path))
    validate_image_geometry(image)

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D image, got shape {image.shape}: {path}"
        )

    return image


def _image_data(
    image: nib.spatialimages.SpatialImage,
    *,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Read an image into a finite NumPy array."""
    data = np.asarray(image.dataobj, dtype=dtype)

    if not np.isfinite(data).all():
        raise ValueError("Image contains NaN or infinite values.")

    return data


def _same_geometry(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    tolerance: float = 1e-4,
) -> bool:
    """Check image dimensions and spatial affine."""
    return (
        first.shape == second.shape
        and np.allclose(
            first.affine,
            second.affine,
            atol=tolerance,
            rtol=0,
        )
    )


def _validate_binary_mask(
    mask_image: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    """Validate a binary brain mask and return it as Boolean data."""
    mask = _image_data(mask_image)

    if not np.isin(np.unique(mask), [0.0, 1.0]).all():
        raise ValueError(
            "HD-BET brain mask contains values other than 0 and 1."
        )

    binary = mask > 0

    if not binary.any():
        raise ValueError("HD-BET produced an empty brain mask.")

    return binary


def _mask_path_from_output(output_path: Path) -> Path:
    """
    Derive the HD-BET 2.0.1 brain-mask filename.

    For:
        subject_FLAIR_skullstripped.nii.gz

    HD-BET creates:
        subject_FLAIR_skullstripped_bet.nii.gz
    """
    if output_path.name.endswith(".nii.gz"):
        stem = output_path.name[:-7]
        return output_path.with_name(f"{stem}_bet.nii.gz")

    raise ValueError(
        "HD-BET output must have a .nii.gz extension."
    )


def _copy_nifti(
    source: Path,
    destination: Path,
    overwrite: bool,
) -> Path:
    """Copy a NIfTI file without silently overwriting existing results."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. "
            "Use overwrite=True to replace it."
        )

    shutil.copy2(source, destination)
    return destination


# ---------------------------------------------------------------------
# HD-BET brain extraction
# ---------------------------------------------------------------------

def run_hd_bet(
    input_path: str | Path,
    output_dir: str | Path,
    subject_id: str | None = None,
    device: str = "cuda",
    tta: bool = True,
    overwrite: bool = False,
    modality: str = "FLAIR",
    output_image_path: str | Path | None = None,
    output_mask_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run HD-BET 2.0.1 on one FLAIR or T1 scan.

    Returns a dictionary containing:
        subject_id
        input_path
        skullstripped_path
        brain_mask_path
        device
        modality
        status

    HD-BET is the only brain-extraction method used here.
    """
    executable = _find_hd_bet_executable()

    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")

    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device {device!r}. "
            f"Choose one of {sorted(SUPPORTED_DEVICES)}."
        )

    input_image = _load_image(input_path)
    _image_data(input_image)

    if subject_id is None:
        subject_id = input_path.parent.parent.name

    output_dir.mkdir(parents=True, exist_ok=True)

    modality_upper = modality.upper()
    if output_image_path is not None:
        stripped_path = Path(output_image_path).resolve()
    elif modality_upper == "T1":
        stripped_path = output_dir / f"{subject_id}_T1_skullstripped.nii.gz"
    else:
        stripped_path = output_dir / f"{subject_id}_FLAIR_skullstripped.nii.gz"

    if output_mask_path is not None:
        brain_mask_path = Path(output_mask_path).resolve()
    elif modality_upper == "T1":
        brain_mask_path = output_dir / f"{subject_id}_T1_brain_mask.nii.gz"
    else:
        brain_mask_path = output_dir / f"{subject_id}_brain_mask.nii.gz"

    if not overwrite:
        existing = [
            path
            for path in (stripped_path, brain_mask_path)
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Output already exists: "
                + ", ".join(str(path) for path in existing)
                + ". Use overwrite=True to replace existing outputs."
            )

    # Use a temporary directory so intermediate HD-BET files do not
    # pollute the final output directory.
    with tempfile.TemporaryDirectory(
        prefix=f"{subject_id}_hdbet_"
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)

        temporary_output = (
            temporary_directory
            / f"{subject_id}_FLAIR_skullstripped.nii.gz"
        )
        temporary_mask = _mask_path_from_output(temporary_output)

        command = [
            executable,
            "-i",
            str(input_path),
            "-o",
            str(temporary_output),
            "-device",
            device,
            "--save_bet_mask",
            "--verbose",
        ]

        if not tta:
            command.append("--disable_tta")

        print(f"\nRunning HD-BET for {subject_id}")
        print(f"Input:  {input_path}")
        print(f"Device: {device}")

        completed = subprocess.run(
            command,
            check=False,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"HD-BET failed for {subject_id} "
                f"with exit code {completed.returncode}."
            )

        if not temporary_output.is_file():
            raise RuntimeError(
                "HD-BET finished without creating the expected "
                f"output image: {temporary_output}"
            )

        if not temporary_mask.is_file():
            raise RuntimeError(
                "HD-BET finished without creating the expected "
                f"brain mask: {temporary_mask}"
            )

        stripped_image = _load_image(temporary_output)
        mask_image = _load_image(temporary_mask)

        if not _same_geometry(input_image, stripped_image):
            raise ValueError(
                "HD-BET output geometry does not match the input."
            )

        if not _same_geometry(stripped_image, mask_image):
            raise ValueError(
                "HD-BET brain-mask geometry does not match its output."
            )

        stripped_data = _image_data(stripped_image)
        mask = _validate_binary_mask(mask_image)

        # HD-BET should set voxels outside the mask to zero.
        outside_nonzero = int(np.count_nonzero(stripped_data[~mask]))

        if outside_nonzero:
            raise ValueError(
                "Skull-stripped image contains nonzero voxels outside "
                f"the brain mask ({outside_nonzero} voxels)."
            )

        # Save only validated outputs.
        _copy_nifti(
            temporary_output,
            stripped_path,
            overwrite=overwrite,
        )
        _copy_nifti(
            temporary_mask,
            brain_mask_path,
            overwrite=overwrite,
        )

    print(f"Skull-stripped image: {stripped_path}")
    print(f"Brain mask:           {brain_mask_path}")

    return {
        "subject_id": subject_id,
        "input_path": str(input_path),
        "skullstripped_path": str(stripped_path),
        "brain_mask_path": str(brain_mask_path),
        "device": device,
        "status": "success",
    }


# ---------------------------------------------------------------------
# ROI alignment for visualization
# ---------------------------------------------------------------------

def _roi_on_reference_grid(
    roi_path: str | Path,
    reference_image: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    """
    Load a lesion ROI and align it to the reference image grid.

    Nearest-neighbour interpolation is used to preserve label values.
    The source ROI file is never modified.
    """
    from nibabel.processing import resample_from_to

    roi_image = _load_image(roi_path)

    if not _same_geometry(roi_image, reference_image):
        roi_image = resample_from_to(
            roi_image,
            (reference_image.shape, reference_image.affine),
            order=0,
        )

    roi_data = _image_data(roi_image)

    unique_values = np.unique(roi_data)

    if not np.isin(unique_values, [0.0, 1.0]).all():
        raise ValueError(
            f"ROI mask must be binary; found values {unique_values[:10]}."
        )

    return roi_data > 0


# ---------------------------------------------------------------------
# Visual QC image generation
# ---------------------------------------------------------------------

def _choose_slices(
    brain_mask: np.ndarray,
    roi_mask: np.ndarray | None,
) -> tuple[int, int, int]:
    """
    Choose central brain slices.

    If a lesion ROI exists, choose slices near its centroid to improve
    the chance of displaying the lesion.
    """
    target = (
        roi_mask
        if roi_mask is not None and roi_mask.any()
        else brain_mask
    )

    coordinates = np.argwhere(target)

    if coordinates.size == 0:
        raise ValueError(
            "Cannot choose QC slices: both target and brain mask are empty."
        )

    center = np.round(coordinates.mean(axis=0)).astype(int)

    x = int(np.clip(center[0], 0, brain_mask.shape[0] - 1))
    y = int(np.clip(center[1], 0, brain_mask.shape[1] - 1))
    z = int(np.clip(center[2], 0, brain_mask.shape[2] - 1))

    return x, y, z


def _display_slice(
    axis: Any,
    image_slice: np.ndarray,
    title: str,
    mask_slice: np.ndarray | None = None,
    roi_slice: np.ndarray | None = None,
) -> None:
    """Render a single image slice with optional mask and ROI contours."""
    axis.imshow(
        image_slice.T,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
    )

    if mask_slice is not None and mask_slice.any():
        axis.contour(
            mask_slice.T.astype(np.uint8),
            levels=[0.5],
            colors=["lime"],
            linewidths=0.8,
        )

    if roi_slice is not None and roi_slice.any():
        axis.contour(
            roi_slice.T.astype(np.uint8),
            levels=[0.5],
            colors=["red"],
            linewidths=1.0,
        )

    axis.set_title(title, fontsize=9)
    axis.axis("off")


def create_visual_qc(
    original_flair_path: str | Path,
    skullstripped_path: str | Path,
    brain_mask_path: str | Path,
    output_png: str | Path | None = None,
    roi_path: str | Path | None = None,
    subject_id: str | None = None,
    modality: str = "FLAIR",
    return_base64: bool = True,
) -> dict[str, Any]:
    """
    Create a six-panel QC image.

    Top row:
        Original axial, coronal, sagittal views.

    Bottom row:
        Skull-stripped axial, coronal, sagittal views.

    Green contour:
        HD-BET brain-mask boundary.

    Red contour:
        FCD lesion ROI boundary, when available.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_flair_path = Path(original_flair_path)
    skullstripped_path = Path(skullstripped_path)
    brain_mask_path = Path(brain_mask_path)

    original_image = _load_image(original_flair_path)
    stripped_image = _load_image(skullstripped_path)
    mask_image = _load_image(brain_mask_path)

    if not _same_geometry(original_image, stripped_image):
        raise ValueError(
            f"Original and skull-stripped {modality} geometry do not match."
        )

    if not _same_geometry(stripped_image, mask_image):
        raise ValueError(
            f"Skull-stripped image and brain-mask geometry do not match "
            f"(modality: {modality})."
        )

    original = _image_data(original_image)
    stripped = _image_data(stripped_image)
    brain_mask = _validate_binary_mask(mask_image)

    roi_mask = None

    if roi_path is not None and Path(roi_path).is_file():
        roi_mask = _roi_on_reference_grid(
            roi_path,
            stripped_image,
        )

    x, y, z = _choose_slices(brain_mask, roi_mask)

    # Transpose the slices for consistent display with the NIfTI
    # voxel-axis convention.
    views = [
        (
            original[:, :, z],
            stripped[:, :, z],
            brain_mask[:, :, z],
            roi_mask[:, :, z] if roi_mask is not None else None,
            f"Axial (z={z})",
        ),
        (
            original[:, y, :],
            stripped[:, y, :],
            brain_mask[:, y, :],
            roi_mask[:, y, :] if roi_mask is not None else None,
            f"Coronal (y={y})",
        ),
        (
            original[x, :, :],
            stripped[x, :, :],
            brain_mask[x, :, :],
            roi_mask[x, :, :] if roi_mask is not None else None,
            f"Sagittal (x={x})",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
        constrained_layout=True,
    )

    for column, view in enumerate(views):
        original_slice, stripped_slice, mask_slice, roi_slice, title = view

        _display_slice(
            axes[0, column],
            original_slice,
            f"Original {modality} — {title}",
            mask_slice=mask_slice,
            roi_slice=roi_slice,
        )

        _display_slice(
            axes[1, column],
            stripped_slice,
            f"Skull-stripped {modality} — {title}",
            mask_slice=mask_slice,
            roi_slice=roi_slice,
        )

    fig.suptitle(
        f"HD-BET Visual QC ({modality}) — {subject_id or original_flair_path.stem}",
        fontsize=14,
    )

    base64_str = ""
    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        base64_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
        buf.close()

    if output_png is not None:
        p_out = Path(output_png)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            p_out,
            dpi=160,
            bbox_inches="tight",
        )
    plt.close(fig)

    brain_voxels = int(brain_mask.sum())
    brain_fraction = float(brain_mask.mean())

    outside_nonzero = int(np.count_nonzero(stripped[~brain_mask]))

    roi_voxels = int(roi_mask.sum()) if roi_mask is not None else 0
    roi_inside_brain = (
        int(np.count_nonzero(roi_mask & brain_mask))
        if roi_mask is not None
        else 0
    )

    roi_overlap_fraction = (
        float(roi_inside_brain / roi_voxels)
        if roi_voxels
        else None
    )

    flags: list[str] = []

    if not MIN_BRAIN_FRACTION <= brain_fraction <= MAX_BRAIN_FRACTION:
        flags.append("REVIEW_BRAIN_FRACTION")

    if outside_nonzero:
        flags.append("REVIEW_NONZERO_OUTSIDE_MASK")

    if roi_mask is not None and roi_voxels == 0:
        flags.append("REVIEW_EMPTY_ROI")

    if roi_voxels and roi_overlap_fraction is not None:
        if roi_overlap_fraction < 0.95:
            flags.append("REVIEW_ROI_MASK_OVERLAP")

    return {
        "subject_id": subject_id or original_flair_path.stem,
        "status": "success",
        "qc_image": str(output_png) if output_png else "",
        "base64_image": base64_str,
        "brain_voxels": brain_voxels,
        "brain_fraction": brain_fraction,
        "stripped_nonzero_outside_mask": outside_nonzero,
        "roi_voxels": roi_voxels,
        "roi_inside_brain_voxels": roi_inside_brain,
        "roi_inside_brain_fraction": roi_overlap_fraction,
        "review_flags": ";".join(flags),
        "qc_status": "REVIEW" if flags else "PASS",
    }


# ---------------------------------------------------------------------
# HTML index and CSV summary
# ---------------------------------------------------------------------

def generate_qc_index(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Generate an HTML page and CSV summary for visual QC.

    QC images may be stored in subject-specific subdirectories under
    output_dir. The HTML links are made relative to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.html"
    csv_path = output_dir / "qc_summary.csv"

    fieldnames = [
        "subject_id",
        "status",
        "qc_status",
        "qc_image",
        "brain_voxels",
        "brain_fraction",
        "stripped_nonzero_outside_mask",
        "roi_voxels",
        "roi_inside_brain_voxels",
        "roi_inside_brain_fraction",
        "review_flags",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow({
                key: record.get(key, "")
                for key in fieldnames
            })

    cards: list[str] = []

    for record in records:
        subject_id = html.escape(str(record.get("subject_id", "unknown")))
        status = str(record.get("status", "unknown"))
        qc_status = str(record.get("qc_status", "NOT_AVAILABLE"))

        flags = html.escape(str(record.get("review_flags", "")))
        error = html.escape(str(record.get("error", "")))

        if status != "success":
            cards.append(
                f"""
                <section class="card error">
                  <h2>{subject_id}</h2>
                  <p><strong>Processing status:</strong> FAILED</p>
                  <pre>{error}</pre>
                </section>
                """
            )
            continue

        base64_data = record.get("base64_image")
        if base64_data:
            img_src = base64_data
        else:
            qc_path = Path(str(record.get("qc_image", "")))
            try:
                relative_image = qc_path.resolve().relative_to(
                    output_dir.resolve()
                ).as_posix()
            except Exception:
                relative_image = f"images/{qc_path.name}"
            img_src = html.escape(relative_image)

        css_class = "review" if qc_status == "REVIEW" else "pass"

        brain_fraction = record.get("brain_fraction")
        brain_fraction_text = (
            f"{float(brain_fraction):.4f}"
            if brain_fraction is not None
            else "N/A"
        )

        roi_fraction = record.get("roi_inside_brain_fraction")
        roi_fraction_text = (
            f"{float(roi_fraction):.4f}"
            if roi_fraction is not None
            else "N/A"
        )

        cards.append(
            f"""
            <section class="card {css_class}">
              <h2>{subject_id}</h2>
              <p><strong>QC status:</strong> {html.escape(qc_status)}</p>
              <p><strong>Brain fraction:</strong> {brain_fraction_text}</p>
              <p><strong>ROI inside brain mask:</strong> {roi_fraction_text}</p>
              <p><strong>Review flags:</strong> {flags or "None"}</p>
              <img src="{img_src}" alt="QC image for {subject_id}">
            </section>
            """
        )

    total = len(records)
    successful = sum(r.get("status") == "success" for r in records)
    failed = total - successful
    review = sum(r.get("qc_status") == "REVIEW" for r in records)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HD-BET Visual Quality Control</title>
<style>
  body {{
    margin: 0;
    padding: 24px;
    background: #f4f6f8;
    color: #20252b;
    font-family: Arial, sans-serif;
  }}
  header {{
    max-width: 1200px;
    margin: 0 auto 24px;
  }}
  h1 {{ margin-bottom: 8px; }}
  .summary {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 20px 0;
  }}
  .summary div {{
    background: white;
    border: 1px solid #d8dee5;
    border-radius: 8px;
    padding: 12px 18px;
  }}
  .grid {{
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
    gap: 20px;
  }}
  .card {{
    min-width: 0;
    background: white;
    border: 2px solid #d8dee5;
    border-radius: 10px;
    padding: 16px;
    overflow-wrap: anywhere;
  }}
  .card.pass {{ border-color: #238636; }}
  .card.review {{ border-color: #d29922; }}
  .card.error {{ border-color: #cf222e; }}
  .card img {{
    display: block;
    width: 100%;
    height: auto;
    margin-top: 12px;
    border: 1px solid #d8dee5;
    background: black;
  }}
  pre {{
    white-space: pre-wrap;
    color: #cf222e;
  }}
  .legend {{
    line-height: 1.8;
    color: #57606a;
  }}
</style>
</head>
<body>
<header>
  <h1>HD-BET Skull-Stripping — Visual QC</h1>
  <p>Review each subject's original and skull-stripped FLAIR images.</p>
  <p class="legend">
    <strong>Green:</strong> brain-mask boundary &nbsp; | &nbsp;
    <strong>Red:</strong> FCD lesion ROI boundary
  </p>
  <div class="summary">
    <div><strong>Total:</strong> {total}</div>
    <div><strong>Successful:</strong> {successful}</div>
    <div><strong>Failed:</strong> {failed}</div>
    <div><strong>Flagged for review:</strong> {review}</div>
  </div>
  <p>
    The automatic QC flags are screening aids, not clinical validation.
    Inspect the images, especially the lesion boundary and the inferior
    and superior brain margins.
  </p>
  <p><a href="qc_summary.csv">Open CSV summary</a></p>
</header>
<main class="grid">
{''.join(cards)}
</main>
</body>
</html>
"""

    index_path.write_text(page, encoding="utf-8")

    return index_path, csv_path


# ---------------------------------------------------------------------
# One-subject processing
# ---------------------------------------------------------------------

def process_subject(
    flair_path: str | Path,
    output_root: str | Path,
    device: str = "cuda",
    tta: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run HD-BET and visual QC for one subject."""
    flair_path = Path(flair_path).resolve()
    output_root = Path(output_root).resolve()

    subject_id = flair_path.parent.parent.name

    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "subject_id": subject_id,
        "input_path": str(flair_path),
        "status": "failed",
        "qc_status": "NOT_AVAILABLE",
        "qc_image": "",
        "review_flags": "",
        "error": "",
    }

    try:
        # Find the subject's lesion ROI independently of the FLAIR
        # acquisition name. ROI and FLAIR acquisitions may differ.
        roi_candidates = sorted(
            path
            for pattern in ("*_FLAIR_roi.nii", "*_FLAIR_roi.nii.gz")
            for path in flair_path.parent.glob(pattern)
            if path.is_file()
        )

        # Deduplicate paths in case filename patterns overlap.
        roi_candidates = sorted(set(roi_candidates))

        if len(roi_candidates) > 1:
            raise RuntimeError(
                f"Multiple ROI files found for {subject_id}: "
                f"{[p.name for p in roi_candidates]}"
            )

        roi_path = roi_candidates[0] if roi_candidates else None

        result = run_hd_bet(
            input_path=flair_path,
            output_dir=subject_output_dir,
            subject_id=subject_id,
            device=device,
            tta=tta,
            overwrite=overwrite,
        )

        qc_png = subject_output_dir / f"{subject_id}_qc.png"

        qc_result = create_visual_qc(
            original_flair_path=flair_path,
            skullstripped_path=result["skullstripped_path"],
            brain_mask_path=result["brain_mask_path"],
            output_png=qc_png,
            roi_path=roi_path,
            subject_id=subject_id,
        )

        record.update(result)
        record.update(qc_result)
        record["status"] = "success"
        record["qc_image"] = str(qc_png)

    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR processing {subject_id}: {record['error']}")

    return record


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------

def discover_flair_files(
    data_root: str | Path,
) -> list[Path]:
    """
    Find FLAIR NIfTI files without accidentally including lesion masks.

    Supports both .nii and .nii.gz filenames.
    """
    data_root = Path(data_root)

    if not data_root.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: {data_root}"
        )

    candidates = list(data_root.glob("sub-*/anat/*_FLAIR.nii"))
    candidates += list(data_root.glob("sub-*/anat/*_FLAIR.nii.gz"))

    flair_files = sorted(
        path
        for path in candidates
        if not path.name.endswith("_FLAIR_roi.nii")
        and not path.name.endswith("_FLAIR_roi.nii.gz")
    )

    if not flair_files:
        raise FileNotFoundError(
            f"No FLAIR files found under {data_root}/sub-*/anat/"
        )

    return flair_files


# ---------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------

def run_dataset_hd_bet(
    data_root: str | Path,
    output_root: str | Path = "outputs/skull_stripping",
    device: str = "cuda",
    limit: int | None = None,
    tta: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Run HD-BET sequentially over discovered FLAIR scans.

    Parameters
    ----------
    data_root:
        Dataset directory, e.g. data/raw/dataset.
    output_root:
        Destination for skull-stripped images and visual QC.
    device:
        cuda, cpu, or mps.
    limit:
        Optional maximum number of subjects to process.
    tta:
        Keep HD-BET test-time augmentation enabled by default.
        Set False to use --disable_tta.
    overwrite:
        Allow existing skull-stripping outputs to be replaced.

    Returns
    -------
    Path to the JSON manifest.
    """
    _find_hd_bet_executable()

    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device {device!r}. "
            f"Choose one of {sorted(SUPPORTED_DEVICES)}."
        )

    flair_files = discover_flair_files(data_root)

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1 or None.")

        flair_files = flair_files[:limit]

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    print(f"FLAIR files selected: {len(flair_files)}")
    print(f"Output directory: {output_root}")
    print(f"Device: {device}")
    print(f"Test-time augmentation: {tta}")

    for index, flair_path in enumerate(flair_files, start=1):
        print(
            f"\n[{index}/{len(flair_files)}] "
            f"{flair_path.parent.parent.name}: {flair_path.name}"
        )

        record = process_subject(
            flair_path=flair_path,
            output_root=output_root,
            device=device,
            tta=tta,
            overwrite=overwrite,
        )
        records.append(record)

        # Refresh the HTML and CSV after each subject so partial results
        # remain available even if a later subject fails.
        generate_qc_index(
            records,
            output_root,
        )

        manifest_path = output_root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )

    index_path, csv_path = generate_qc_index(
        records,
        output_root,
    )

    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    successful = sum(
        record.get("status") == "success"
        for record in records
    )
    failed = len(records) - successful
    flagged = sum(
        record.get("qc_status") == "REVIEW"
        for record in records
    )

    print("\n" + "=" * 65)
    print("HD-BET DATASET RUN FINISHED")
    print("=" * 65)
    print(f"Subjects attempted: {len(records)}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"QC review flags:    {flagged}")
    print(f"HTML index:         {index_path}")
    print(f"CSV summary:        {csv_path}")
    print(f"JSON manifest:      {manifest_path}")

    return manifest_path