from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from splat_trimmer.models import Cloud


def aabb_mask(
    positions: NDArray[np.floating],
    mn: NDArray[np.floating],
    mx: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """(N,) bool mask: True where the point is inside the bbox."""
    mn_a = np.asarray(mn)
    mx_a = np.asarray(mx)
    inside = np.all((positions >= mn_a) & (positions <= mx_a), axis=1)
    mask: NDArray[np.bool_] = inside.astype(np.bool_)
    return mask


def crop_aabb(
    cloud: Cloud,
    mn: NDArray[np.floating],
    mx: NDArray[np.floating],
) -> Cloud:
    """Return a new Cloud containing only splats inside the AABB."""
    keep = aabb_mask(cloud.positions, mn, mx)
    return cloud.mask(keep)
