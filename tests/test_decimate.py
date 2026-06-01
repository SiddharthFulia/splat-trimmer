from __future__ import annotations

import numpy as np

from splat_trimmer.models import Cloud
from splat_trimmer.ops.decimate import decimate


def test_decimate_exact_count(gridded_cloud: Cloud) -> None:
    out = decimate(gridded_cloud, target=50, seed=0)
    assert out.count == 50


def test_decimate_returns_original_if_target_large(gridded_cloud: Cloud) -> None:
    out = decimate(gridded_cloud, target=10_000)
    assert out.count == gridded_cloud.count


def test_decimate_is_deterministic(gridded_cloud: Cloud) -> None:
    a = decimate(gridded_cloud, target=30, seed=1234)
    b = decimate(gridded_cloud, target=30, seed=1234)
    np.testing.assert_array_equal(a.positions, b.positions)


def test_decimate_preserves_distribution() -> None:
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((20_000, 3)).astype(np.float32)
    cloud = Cloud(
        positions=pts,
        scales=np.zeros((20_000, 3), dtype=np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (20_000, 1)),
        opacities=np.zeros(20_000, dtype=np.float32),
        sh=np.zeros((20_000, 3), dtype=np.float32),
    )
    sub = decimate(cloud, target=5_000, seed=0)
    np.testing.assert_allclose(sub.positions.mean(axis=0), pts.mean(axis=0), atol=0.06)
    np.testing.assert_allclose(sub.positions.std(axis=0), pts.std(axis=0), atol=0.08)
