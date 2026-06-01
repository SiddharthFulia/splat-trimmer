from __future__ import annotations

from pathlib import Path

import numpy as np

from splat_trimmer.models import Cloud

SPLAT_STRIDE = 32  # bytes per splat (pos12 + scale12 + rgba4 + quat4)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def load_splat(path: str | Path) -> Cloud:
    """Load an antimatter15 `.splat` file into a Cloud."""
    raw = Path(path).read_bytes()
    if len(raw) % SPLAT_STRIDE != 0:
        raise ValueError(
            f"{path}: file size {len(raw)} is not a multiple of {SPLAT_STRIDE} bytes"
        )
    n = len(raw) // SPLAT_STRIDE
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, SPLAT_STRIDE)

    positions = buf[:, 0:12].copy().view(np.float32).reshape(n, 3)
    scales_lin = buf[:, 12:24].copy().view(np.float32).reshape(n, 3)
    # convert to log-scale to match the PLY representation
    scales = np.log(np.clip(scales_lin, 1e-12, None)).astype(np.float32)

    rgba = buf[:, 24:28].astype(np.float32) / 255.0
    rot_bytes = buf[:, 28:32].astype(np.float32)
    rotations = (rot_bytes - 128.0) / 128.0

    opacities = _logit(rgba[:, 3]).astype(np.float32)
    sh = rgba[:, :3].astype(np.float32)

    return Cloud(
        positions=positions.astype(np.float32),
        scales=scales,
        rotations=rotations.astype(np.float32),
        opacities=opacities,
        sh=sh,
        source_format="splat",
    )


def write_splat(path: str | Path, cloud: Cloud) -> None:
    """Write a Cloud to antimatter15 `.splat`. Higher-order SH is dropped (format limit)."""
    n = cloud.count
    buf = np.zeros((n, SPLAT_STRIDE), dtype=np.uint8)

    buf[:, 0:12] = cloud.positions.astype(np.float32).view(np.uint8).reshape(n, 12)
    lin_scales = np.exp(cloud.scales).astype(np.float32)
    buf[:, 12:24] = lin_scales.view(np.uint8).reshape(n, 12)

    rgb = np.clip(cloud.sh[:, :3], 0.0, 1.0)
    alpha = _sigmoid(cloud.opacities)
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)
    buf[:, 24:28] = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)

    rot = np.clip(cloud.rotations, -1.0, 1.0)
    buf[:, 28:32] = np.clip(rot * 128.0 + 128.0, 0, 255).astype(np.uint8)

    Path(path).write_bytes(buf.tobytes())
