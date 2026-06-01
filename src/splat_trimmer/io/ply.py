from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from splat_trimmer.models import Cloud


def _field_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def _gather(vertex: object, names: list[str]) -> np.ndarray:
    cols = [np.asarray(vertex[n], dtype=np.float32) for n in names]  # type: ignore[index]
    return np.stack(cols, axis=1)


def load_ply(path: str | Path) -> Cloud:
    """Load an INRIA-format Gaussian-splat PLY into a Cloud."""
    data = PlyData.read(str(path))
    vertex = data["vertex"]

    n = vertex.count
    positions = np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
        axis=1,
    ).astype(np.float32)

    scales = _gather(vertex, _field_names("scale", 3))
    rotations = _gather(vertex, _field_names("rot", 4))
    opacities = np.asarray(vertex["opacity"], dtype=np.float32).reshape(n)

    dc = _gather(vertex, _field_names("f_dc", 3))
    rest_names = [
        name
        for name in vertex.data.dtype.names
        if name.startswith("f_rest_")
    ]
    rest_names.sort(key=lambda s: int(s.split("_")[-1]))
    rest = _gather(vertex, rest_names) if rest_names else np.zeros((n, 0), dtype=np.float32)
    sh = np.concatenate([dc, rest], axis=1)

    return Cloud(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh=sh,
        source_format="ply",
    )


def write_ply(path: str | Path, cloud: Cloud) -> None:
    """Write a Cloud back to an INRIA-format binary PLY."""
    n = cloud.count
    sh_rest = cloud.sh.shape[1] - 3
    if sh_rest < 0:
        raise ValueError("Cloud.sh must have at least 3 columns (f_dc_0..2)")

    dtype: list[tuple[str, str]] = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    dtype += [(f"f_rest_{i}", "f4") for i in range(sh_rest)]
    dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = (
        cloud.positions[:, 0],
        cloud.positions[:, 1],
        cloud.positions[:, 2],
    )
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = (
        cloud.sh[:, 0],
        cloud.sh[:, 1],
        cloud.sh[:, 2],
    )
    for i in range(sh_rest):
        arr[f"f_rest_{i}"] = cloud.sh[:, 3 + i]
    arr["opacity"] = cloud.opacities
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = (
        cloud.scales[:, 0],
        cloud.scales[:, 1],
        cloud.scales[:, 2],
    )
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
        cloud.rotations[:, 0],
        cloud.rotations[:, 1],
        cloud.rotations[:, 2],
        cloud.rotations[:, 3],
    )

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
