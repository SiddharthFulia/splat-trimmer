from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from splat_trimmer.models import Cloud


def outlier_mask(
    positions: NDArray[np.floating],
    percentile: float,
) -> NDArray[np.bool_]:
    """Keep points whose centroid distance is below the p-th percentile."""
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100); got {percentile}")
    pts = np.asarray(positions, dtype=np.float64)
    centroid = pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid, axis=1)
    cutoff = float(np.percentile(d, percentile))
    mask: NDArray[np.bool_] = (d <= cutoff).astype(np.bool_)
    return mask


def drop_outliers(cloud: Cloud, percentile: float = 99.0) -> Cloud:
    """Return a new Cloud with the outermost `100 - percentile`% of points removed."""
    keep = outlier_mask(cloud.positions, percentile)
    return cloud.mask(keep)
