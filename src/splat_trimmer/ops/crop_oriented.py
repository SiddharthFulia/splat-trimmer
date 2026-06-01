from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from splat_trimmer.models import Cloud


def pca_axes(
    positions: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """`(centroid, axes)` — rows of `axes` are unit eigenvectors, descending variance."""
    pts = np.asarray(positions, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T
    return centroid.astype(np.float32), axes.astype(np.float32)


def crop_oriented_bbox(cloud: Cloud, margin: float = 0.0) -> Cloud:
    """Trim `cloud` to its PCA-aligned bbox. `margin` shrinks per-axis extents on each side."""
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5); got {margin}")

    centroid, axes = pca_axes(cloud.positions)
    centred = cloud.positions - centroid
    projected = centred @ axes.T

    mn = projected.min(axis=0)
    mx = projected.max(axis=0)
    span = mx - mn
    mn_eff = mn + margin * span
    mx_eff = mx - margin * span

    keep = np.all(
        (projected >= mn_eff) & (projected <= mx_eff),
        axis=1,
    ).astype(np.bool_)
    return cloud.mask(keep)
