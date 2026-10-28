"""Baseline foreground percentile clipping and z-score normalization."""
import numpy as np

def normalize_intensity(data, mask=None, *, clip_percentiles=(0.5, 99.5), eps=1e-8):
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {values.ndim} dimensions")
    finite = np.isfinite(values)
    foreground = finite & (values != 0) if mask is None else finite & np.asarray(mask, dtype=bool)
    if foreground.shape != values.shape:
        raise ValueError("Mask shape must match image shape")
    if not foreground.any():
        return np.zeros_like(values, dtype=np.float32)
    low, high = clip_percentiles
    if not 0 <= low < high <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    lower, upper = np.percentile(values[foreground], [low, high])
    clipped = np.clip(values, lower, upper)
    selected = clipped[foreground]
    mean, std = float(selected.mean()), float(selected.std())
    result = np.zeros_like(values, dtype=np.float32)
    if std > eps:
        result[foreground] = (selected - mean) / std
    return result
