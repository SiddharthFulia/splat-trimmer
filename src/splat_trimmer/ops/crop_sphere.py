from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from splat_trimmer.models import Cloud


def sphere_mask(
    positions: NDArray[np.floating],
    center: NDArray[np.floating],
    radius: float,
) -> NDArray[np.bool_]:
    """(N,) bool mask: True where ||p - center|| <= radius."""
    center_a = np.asarray(center, dtype=np.float64).reshape(1, 3)
    d2 = np.sum((np.asarray(positions, dtype=np.float64) - center_a) ** 2, axis=1)
    return (d2 <= float(radius) ** 2).astype(np.bool_)


def crop_sphere(
    cloud: Cloud,
    center: NDArray[np.floating],
    radius: float,
) -> Cloud:
    """Return a new Cloud containing only splats inside the sphere."""
    keep = sphere_mask(cloud.positions, center, radius)
    return cloud.mask(keep)
