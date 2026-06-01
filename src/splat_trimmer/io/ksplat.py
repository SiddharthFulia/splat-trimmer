from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from splat_trimmer.models import Cloud

KSPLAT_HEADER_BYTES = 4096
KSPLAT_MAGIC = b"KSPL"


def load_ksplat(path: str | Path) -> Cloud:
    """Load a mkkellogg KSplat file (compression level 0 only)."""
    # TODO: support compressionLevel 1 (fp16) and 2 (int8)
    raw = Path(path).read_bytes()
    if len(raw) < KSPLAT_HEADER_BYTES:
        raise ValueError(f"{path}: too small to be a KSplat file")
    if raw[:4] != KSPLAT_MAGIC:
        raise ValueError(f"{path}: missing KSPL magic header")

    (
        _magic,
        version,
        _max_sections,
        section_count,
        _max_splats,
        splat_count,
        compression,
    ) = struct.unpack_from("<4sIIIIII", raw, 0)

    if version not in (1, 2):
        raise ValueError(f"Unsupported KSplat version {version}")
    if compression != 0:
        raise NotImplementedError(
            f"KSplat compressionLevel={compression} not yet supported; please re-export "
            "with compression level 0 (raw float32) using the upstream KSplat tools."
        )

    # Section format: pos12 + scale12 + rot16 + rgba4 at compression 0 (bps=44)
    cursor = KSPLAT_HEADER_BYTES
    sections: list[np.ndarray] = []
    for _ in range(section_count):
        s_count, s_bps, _s_off = struct.unpack_from("<III", raw, cursor)
        cursor += 12
        block = np.frombuffer(raw, dtype=np.uint8, count=s_count * s_bps, offset=cursor)
        sections.append(block.reshape(s_count, s_bps))
        cursor += s_count * s_bps

    if not sections:
        empty = np.zeros((0, 3), dtype=np.float32)
        return Cloud(
            positions=empty,
            scales=empty.copy(),
            rotations=np.zeros((0, 4), dtype=np.float32),
            opacities=np.zeros((0,), dtype=np.float32),
            sh=np.zeros((0, 3), dtype=np.float32),
            source_format="ksplat",
        )

    buf = np.concatenate(sections, axis=0)
    n = buf.shape[0]
    assert n == splat_count, f"section splat count {n} != header {splat_count}"

    positions = buf[:, 0:12].copy().view(np.float32).reshape(n, 3)
    scales_lin = buf[:, 12:24].copy().view(np.float32).reshape(n, 3)
    scales = np.log(np.clip(scales_lin, 1e-12, None)).astype(np.float32)
    rotations = buf[:, 24:40].copy().view(np.float32).reshape(n, 4)
    rgba = buf[:, 40:44].astype(np.float32) / 255.0

    eps = 1e-6
    a = np.clip(rgba[:, 3], eps, 1 - eps)
    opacities = np.log(a / (1 - a)).astype(np.float32)
    sh = rgba[:, :3].astype(np.float32)

    return Cloud(
        positions=positions.astype(np.float32),
        scales=scales,
        rotations=rotations.astype(np.float32),
        opacities=opacities,
        sh=sh,
        source_format="ksplat",
    )
