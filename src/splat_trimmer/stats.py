from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from splat_trimmer.models import Cloud


@dataclass(slots=True)
class CloudStats:
    """Summary statistics for a splat cloud."""

    count: int
    mn: NDArray[np.float32]
    mx: NDArray[np.float32]
    centroid: NDArray[np.float32]
    std: NDArray[np.float32]
    diagonal: float

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "min": self.mn.tolist(),
            "max": self.mx.tolist(),
            "centroid": self.centroid.tolist(),
            "std": self.std.tolist(),
            "diagonal": self.diagonal,
        }


def summarise(cloud: Cloud) -> CloudStats:
    """Compute per-axis min/max/centroid/std for a Cloud."""
    pts = np.asarray(cloud.positions, dtype=np.float64)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    centroid = pts.mean(axis=0)
    std = pts.std(axis=0)
    diagonal = float(np.linalg.norm(mx - mn))
    return CloudStats(
        count=cloud.count,
        mn=mn.astype(np.float32),
        mx=mx.astype(np.float32),
        centroid=centroid.astype(np.float32),
        std=std.astype(np.float32),
        diagonal=diagonal,
    )
