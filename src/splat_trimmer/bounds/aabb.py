from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_aabb(
    positions: NDArray[np.floating],
    percentile: float | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Axis-aligned bbox of (N, 3) points. `percentile` clips outliers per axis."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {positions.shape}")
    if positions.shape[0] == 0:
        raise ValueError("Cannot compute AABB on an empty point set")

    pts = np.asarray(positions, dtype=np.float64)

    if percentile is None:
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
    else:
        if not 50.0 < percentile <= 100.0:
            raise ValueError(f"percentile must be in (50, 100]; got {percentile}")
        lo = 100.0 - percentile
        hi = percentile
        mn = np.percentile(pts, lo, axis=0)
        mx = np.percentile(pts, hi, axis=0)

    return mn.astype(np.float32), mx.astype(np.float32)


def aabb_diagonal(mn: NDArray[np.floating], mx: NDArray[np.floating]) -> float:
    return float(np.linalg.norm(np.asarray(mx) - np.asarray(mn)))


def aabb_center(mn: NDArray[np.floating], mx: NDArray[np.floating]) -> NDArray[np.float32]:
    return ((np.asarray(mn) + np.asarray(mx)) * 0.5).astype(np.float32)
