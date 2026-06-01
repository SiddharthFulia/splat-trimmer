from __future__ import annotations

import numpy as np
import pytest

from splat_trimmer.bounds.aabb import aabb_center, aabb_diagonal, compute_aabb
from splat_trimmer.models import Cloud
from splat_trimmer.ops.crop_aabb import aabb_mask, crop_aabb


def test_compute_aabb_exact() -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
        ],
        dtype=np.float32,
    )
    mn, mx = compute_aabb(pts)
    np.testing.assert_array_equal(mn, np.array([-1.0, -2.0, -3.0], dtype=np.float32))
    np.testing.assert_array_equal(mx, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_compute_aabb_percentile_rejects_outlier() -> None:
    pts = np.zeros((101, 3), dtype=np.float32)
    pts[100] = [1000.0, 1000.0, 1000.0]
    mn, mx = compute_aabb(pts, percentile=99.0)
    assert float(mx[0]) < 100.0
    mn2, mx2 = compute_aabb(pts)
    assert mx2[0] == 1000.0


def test_compute_aabb_validates() -> None:
    with pytest.raises(ValueError):
        compute_aabb(np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        compute_aabb(np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        compute_aabb(np.zeros((3, 3), dtype=np.float32), percentile=10.0)


def test_aabb_center_diagonal() -> None:
    mn = np.array([0.0, 0.0, 0.0])
    mx = np.array([2.0, 4.0, 6.0])
    np.testing.assert_array_equal(aabb_center(mn, mx), [1.0, 2.0, 3.0])
    assert aabb_diagonal(mn, mx) == pytest.approx(np.sqrt(4 + 16 + 36))


def test_crop_aabb_on_cube(gridded_cloud: Cloud) -> None:
    mn = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    mx = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    keep = aabb_mask(gridded_cloud.positions, mn, mx)
    trimmed = crop_aabb(gridded_cloud, mn, mx)
    assert trimmed.count == int(keep.sum())
    # inner 3^3 grid points only
    assert trimmed.count == 27
