from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bounding_sphere(
    positions: NDArray[np.floating],
) -> tuple[NDArray[np.float32], float]:
    """Ritter bounding sphere — O(N), ~5-10% larger than the minimum enclosing sphere."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {positions.shape}")
    if positions.shape[0] == 0:
        raise ValueError("Cannot compute bounding sphere on an empty point set")

    pts = np.asarray(positions, dtype=np.float64)

    p0 = pts[0]
    d2 = np.sum((pts - p0) ** 2, axis=1)
    p1 = pts[int(np.argmax(d2))]
    d2 = np.sum((pts - p1) ** 2, axis=1)
    p2 = pts[int(np.argmax(d2))]

    center = (p1 + p2) * 0.5
    radius = float(np.sqrt(np.sum((p2 - center) ** 2)))

    for p in pts:
        diff = p - center
        dist = float(np.sqrt(diff @ diff))
        if dist > radius:
            new_radius = (radius + dist) * 0.5
            if dist > 0.0:
                center = center + diff * ((dist - new_radius) / dist)
            radius = new_radius

    # epsilon so fp round-off doesn't push points outside
    radius *= 1.0 + 1e-6
    return center.astype(np.float32), float(radius)
