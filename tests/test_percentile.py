from __future__ import annotations

import numpy as np
import pytest

from splat_trimmer.models import Cloud
from splat_trimmer.ops.percentile import drop_outliers, outlier_mask


def _cloud_from_positions(pts: np.ndarray) -> Cloud:
    n = pts.shape[0]
    return Cloud(
        positions=pts.astype(np.float32),
        scales=np.zeros((n, 3), dtype=np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        opacities=np.zeros(n, dtype=np.float32),
        sh=np.zeros((n, 3), dtype=np.float32),
    )


def test_outlier_mask_drops_far_points() -> None:
    pts = np.zeros((100, 3), dtype=np.float32)
    pts[0] = [100.0, 0.0, 0.0]
    keep = outlier_mask(pts, percentile=95.0)
    assert not bool(keep[0])
    assert keep.sum() >= 90


def test_drop_outliers_returns_cloud() -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1000, 3)).astype(np.float32)
    pts[0] = [1e6, 0.0, 0.0]
    cloud = _cloud_from_positions(pts)
    trimmed = drop_outliers(cloud, percentile=99.0)
    assert trimmed.count < cloud.count
    assert float(trimmed.positions[:, 0].max()) < 1e3


def test_percentile_validates() -> None:
    with pytest.raises(ValueError):
        outlier_mask(np.zeros((10, 3), dtype=np.float32), percentile=0.0)
    with pytest.raises(ValueError):
        outlier_mask(np.zeros((10, 3), dtype=np.float32), percentile=150.0)
