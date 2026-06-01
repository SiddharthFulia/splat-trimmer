from __future__ import annotations

from pathlib import Path

from splat_trimmer.io.registry import write
from splat_trimmer.models import Cloud


def write_same_format(out_path: str | Path, cloud: Cloud) -> None:
    """Write `cloud` to `out_path` via the registered writer for its extension."""
    write(out_path, cloud)


def suggest_output_name(in_path: str | Path, suffix: str = "trimmed") -> Path:
    """`scene.ply` -> `scene.trimmed.ply` (sibling path)."""
    p = Path(in_path)
    return p.with_name(f"{p.stem}.{suffix}{p.suffix}")
