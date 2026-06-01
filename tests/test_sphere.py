from __future__ import annotations

import numpy as np
import pytest

from splat_trimmer.bounds.sphere import bounding_sphere
from splat_trimmer.models import Cloud
from splat_trimmer.ops.crop_sphere import crop_sphere, sphere_mask


def _all_inside(pts: np.ndarray, center: np.ndarray, radius: float) -> bool:
    d = np.linalg.norm(pts - center, axis=1)
    return bool(np.all(d <= radius + 1e-5))


def test_sphere_contains_random_cloud() -> None:
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    center, radius = bounding_sphere(pts)
    assert radius > 0
    assert _all_inside(pts, center, radius)


def test_sphere_handles_collinear_points() -> None:
    pts = np.stack(
        [np.linspace(-3.0, 3.0, 20), np.zeros(20), np.zeros(20)],
        axis=1,
    ).astype(np.float32)
    center, radius = bounding_sphere(pts)
    assert radius >= 3.0 - 1e-3
    assert _all_inside(pts, center, radius)


def test_sphere_grid_cloud(gridded_cloud: Cloud) -> None:
    center, radius = bounding_sphere(gridded_cloud.positions)
    assert _all_inside(gridded_cloud.positions, center, radius)


def test_sphere_validates() -> None:
    with pytest.raises(ValueError):
        bounding_sphere(np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        bounding_sphere(np.zeros((4, 2), dtype=np.float32))


def test_crop_sphere(gridded_cloud: Cloud) -> None:
    keep = sphere_mask(gridded_cloud.positions, np.zeros(3, dtype=np.float32), 0.6)
    trimmed = crop_sphere(gridded_cloud, np.zeros(3, dtype=np.float32), 0.6)
    assert trimmed.count == int(keep.sum())
    assert trimmed.count >= 1
