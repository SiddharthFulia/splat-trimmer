from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Format = Literal["ply", "splat", "ksplat", "spz"]


@dataclass(slots=True)
class Splat:
    """A single Gaussian splat."""

    position: NDArray[np.float32]
    scale: NDArray[np.float32]
    rotation: NDArray[np.float32]  # quat convention varies by source format
    opacity: float
    sh: NDArray[np.float32] = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))


@dataclass(slots=True)
class Cloud:
    """Struct-of-arrays splat scene — all arrays share length N."""

    positions: NDArray[np.float32]
    scales: NDArray[np.float32]
    rotations: NDArray[np.float32]
    opacities: NDArray[np.float32]
    sh: NDArray[np.float32]
    source_format: Format = "ply"
    extra: dict[str, NDArray[np.float32]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def count(self) -> int:
        return len(self)

    def mask(self, keep: NDArray[np.bool_]) -> Cloud:
        """Return a new Cloud containing only splats where `keep` is True."""
        if keep.shape[0] != self.count:
            raise ValueError(f"mask length {keep.shape[0]} != cloud size {self.count}")
        return Cloud(
            positions=self.positions[keep],
            scales=self.scales[keep],
            rotations=self.rotations[keep],
            opacities=self.opacities[keep],
            sh=self.sh[keep],
            source_format=self.source_format,
            extra={k: v[keep] for k, v in self.extra.items()},
        )
