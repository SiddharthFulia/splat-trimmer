"""Regenerate tiny.ply / tiny.splat. Re-run when on-disk schemas change."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).parent


def make_ply() -> None:
    rng = np.random.default_rng(0)
    n = 10
    pts = np.stack(
        [
            np.linspace(-1.0, 1.0, n, dtype=np.float32),
            np.linspace(-2.0, 2.0, n, dtype=np.float32),
            np.linspace(-0.5, 0.5, n, dtype=np.float32),
        ],
        axis=1,
    )
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["f_dc_0"] = rng.random(n).astype(np.float32)
    arr["f_dc_1"] = rng.random(n).astype(np.float32)
    arr["f_dc_2"] = rng.random(n).astype(np.float32)
    arr["opacity"] = rng.random(n).astype(np.float32)
    arr["scale_0"] = rng.random(n).astype(np.float32) - 2.0
    arr["scale_1"] = rng.random(n).astype(np.float32) - 2.0
    arr["scale_2"] = rng.random(n).astype(np.float32) - 2.0
    arr["rot_0"] = 1.0
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(ROOT / "tiny.ply"))


def make_splat() -> None:
    n = 10
    pts = np.stack(
        [
            np.linspace(-1.0, 1.0, n, dtype=np.float32),
            np.linspace(-2.0, 2.0, n, dtype=np.float32),
            np.linspace(-0.5, 0.5, n, dtype=np.float32),
        ],
        axis=1,
    )
    scales = np.full((n, 3), 0.1, dtype=np.float32)
    rgba = np.tile(np.array([200, 100, 50, 220], dtype=np.uint8), (n, 1))
    # quat ~ (1, 0, 0, 0) encoded as bytes: (b - 128) / 128
    rot = np.tile(np.array([255, 128, 128, 128], dtype=np.uint8), (n, 1))

    buf = np.zeros((n, 32), dtype=np.uint8)
    buf[:, 0:12] = pts.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scales.view(np.uint8).reshape(n, 12)
    buf[:, 24:28] = rgba
    buf[:, 28:32] = rot
    (ROOT / "tiny.splat").write_bytes(buf.tobytes())


if __name__ == "__main__":
    make_ply()
    make_splat()
    print("wrote tiny.ply and tiny.splat")
