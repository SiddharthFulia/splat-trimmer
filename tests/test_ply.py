from __future__ import annotations

from pathlib import Path

import numpy as np

from splat_trimmer.io.ply import load_ply, write_ply


def test_load_tiny_ply(tiny_ply_path: Path) -> None:
    cloud = load_ply(tiny_ply_path)
    assert cloud.count == 10
    np.testing.assert_allclose(cloud.positions[0], [-1.0, -2.0, -0.5], rtol=1e-6)
    np.testing.assert_allclose(cloud.positions[-1], [1.0, 2.0, 0.5], rtol=1e-6)


def test_ply_round_trip(tiny_ply_path: Path, tmp_path: Path) -> None:
    cloud = load_ply(tiny_ply_path)
    out = tmp_path / "round.ply"
    write_ply(out, cloud)
    again = load_ply(out)
    np.testing.assert_array_equal(cloud.positions, again.positions)
    np.testing.assert_array_equal(cloud.scales, again.scales)
    np.testing.assert_array_equal(cloud.rotations, again.rotations)
    np.testing.assert_array_equal(cloud.opacities, again.opacities)
    np.testing.assert_array_equal(cloud.sh, again.sh)
