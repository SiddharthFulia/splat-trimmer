from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from splat_trimmer.models import Cloud

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_ply_path() -> Path:
    return FIXTURES / "tiny.ply"


@pytest.fixture(scope="session")
def tiny_splat_path() -> Path:
    return FIXTURES / "tiny.splat"


@pytest.fixture()
def gridded_cloud() -> Cloud:
    """Deterministic 5x5x5 = 125 splat cube spanning [-1, 1]^3."""
    coords = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float32)
    n = pts.shape[0]
    return Cloud(
        positions=pts,
        scales=np.zeros((n, 3), dtype=np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        opacities=np.zeros(n, dtype=np.float32),
        sh=np.zeros((n, 3), dtype=np.float32),
    )
