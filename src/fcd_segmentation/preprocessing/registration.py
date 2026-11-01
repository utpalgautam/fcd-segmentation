"""MRI registration utilities for the FCD preprocessing pipeline.

Registers a moving MRI image to a fixed reference image using SimpleITK.

For multimodal FCD preprocessing:
    Fixed image: T1-weighted MRI
    Moving image: FLAIR MRI

Registration estimates an anatomical transformation. Resampling then applies
that transformation to place the moving image on the reference image grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import SimpleITK as sitk


def register_to_reference(
    moving_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "rigid",
    transform_output_path: str | Path | None = None,
    number_of_iterations: int = 200,
    number_of_bins: int = 50,
    sampling_percentage: float = 0.20,
    **kwargs: Any,
) -> Path:
    """Register a moving MRI image to a fixed anatomical reference.

    Parameters
    ----------
    moving_path:
        Path to the moving image, such as FLAIR MRI.

    reference_path:
        Path to the fixed reference image, such as T1-weighted MRI.

    output_path:
        Destination for the registered moving image.

    method:
        Registration transformation. Supported values are "rigid" and
        "affine". Rigid registration is the recommended starting point.

    transform_output_path:
        Optional destination for saving the estimated transformation.

    number_of_iterations:
        Maximum number of optimizer iterations.

    number_of_bins:
        Number of histogram bins used by the Mattes mutual information metric.

    sampling_percentage:
        Fraction of spatial samples used to evaluate the registration metric.
        Must be greater than 0 and at most 1.

    kwargs:
        Reserved for future registration options. Unknown options raise
        TypeError rather than being silently ignored.

    Returns
    -------
    Path
        Path to the registered image.

    Raises
    ------
    FileNotFoundError
        If either input image does not exist.

    ValueError
        If the registration parameters are invalid.

    NotImplementedError
        If an unsupported registration method is requested.
    """

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected registration arguments: {unexpected}")

    moving_path = Path(moving_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    # Validate input files.
    if not moving_path.is_file():
        raise FileNotFoundError(f"Moving image not found: {moving_path}")

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference image not found: {reference_path}"
        )

    if number_of_iterations <= 0:
        raise ValueError("number_of_iterations must be positive.")

    if number_of_bins < 5:
        raise ValueError("number_of_bins must be at least 5.")

    if not 0 < sampling_percentage <= 1:
        raise ValueError("sampling_percentage must be in (0, 1].")

    method = method.lower()

    if method not in {"rigid", "affine"}:
        raise NotImplementedError(
            f"Unsupported registration method: {method!r}. "
            "Supported methods are 'rigid' and 'affine'."
        )

    # Load images. SimpleITK reads the physical image geometry from NIfTI.
    fixed = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        raise ValueError("Registration requires 3D MRI images.")

    if any(size < 2 for size in fixed.GetSize() + moving.GetSize()):
        raise ValueError("Input images must contain at least 2 voxels per axis.")

    # Initialize a transformation using image geometry.
    if method == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    else:
        initial_rigid = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        # Optimize an affine transform initialized from the rigid transform.
        initial_transform = sitk.AffineTransform(3)
        initial_transform.SetCenter(initial_rigid.GetCenter())
        initial_transform.SetMatrix(initial_rigid.GetMatrix())
        initial_transform.SetTranslation(initial_rigid.GetTranslation())

    # Configure the registration metric and optimizer.
    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=number_of_bins
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )
    registration.SetMetricSamplingPercentage(sampling_percentage, seed=42)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=number_of_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    # Use a multiresolution pyramid to improve convergence.
    registration.SetShrinkFactorsPerLevel(
        shrinkFactors=[4, 2, 1]
    )
    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[2, 1, 0]
    )
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False,
    )

    # Estimate the transformation aligning FLAIR with T1.
    final_transform = registration.Execute(fixed, moving)

    # Resample FLAIR onto the T1 reference grid.
    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    # Create the output directory if necessary.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(registered, str(output_path), True)

    # Optionally save the transformation for reuse with the lesion mask.
    if transform_output_path is not None:
        transform_path = Path(transform_output_path)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(final_transform, str(transform_path))

    print("Registration completed.")
    print(f"Method: {method}")
    print(f"Reference (T1): {reference_path}")
    print(f"Moving (FLAIR): {moving_path}")
    print(f"Registered output: {output_path}")
    print(f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration.GetMetricValue():.6f}")

    return output_path