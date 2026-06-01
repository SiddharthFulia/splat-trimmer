from __future__ import annotations

from pathlib import Path


def load_spz(path: str | Path) -> None:
    """Stub: PlayCanvas `.spz` decoder not yet implemented."""
    # TODO: implement variable-bit position decoder; reuse splat.py for dc-SH output
    raise NotImplementedError(
        f"{path}: .spz (PlayCanvas compressed splats) is not yet supported. "
        "Please decode to .ply or .splat first using the upstream supersplat tools, "
        "then run splat-trimmer on the decoded file."
    )
