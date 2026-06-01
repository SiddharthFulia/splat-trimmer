from __future__ import annotations

from pathlib import Path

import numpy as np

from splat_trimmer.io.splat import load_splat, write_splat


def test_load_tiny_splat(tiny_splat_path: Path) -> None:
    cloud = load_splat(tiny_splat_path)
    assert cloud.count == 10
    np.testing.assert_allclose(cloud.positions[0], [-1.0, -2.0, -0.5], rtol=1e-6)
    np.testing.assert_allclose(cloud.positions[-1], [1.0, 2.0, 0.5], rtol=1e-6)


def test_splat_round_trip_positions(tiny_splat_path: Path, tmp_path: Path) -> None:
    cloud = load_splat(tiny_splat_path)
    out = tmp_path / "round.splat"
    write_splat(out, cloud)
    again = load_splat(out)
    # positions are lossless float32; rgba/quat go via uint8 quantisation
    np.testing.assert_array_equal(cloud.positions, again.positions)
    np.testing.assert_allclose(cloud.scales, again.scales, atol=1e-5)
