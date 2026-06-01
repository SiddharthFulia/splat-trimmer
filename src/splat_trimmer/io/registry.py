from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from splat_trimmer.io.ksplat import load_ksplat
from splat_trimmer.io.ply import load_ply, write_ply
from splat_trimmer.io.splat import load_splat, write_splat
from splat_trimmer.io.spz import load_spz
from splat_trimmer.models import Cloud

Loader = Callable[[str], Cloud]
Writer = Callable[[str, Cloud], None]

LOADERS: dict[str, Loader] = {
    ".ply": load_ply,
    ".splat": load_splat,
    ".ksplat": load_ksplat,
    ".spz": load_spz,  # type: ignore[dict-item]  # stub raises NotImplementedError
}

WRITERS: dict[str, Writer] = {
    ".ply": write_ply,
    ".splat": write_splat,
}


def _ext(path: str | Path) -> str:
    return Path(path).suffix.lower()


def load(path: str | Path) -> Cloud:
    """Dispatch loader by file extension."""
    ext = _ext(path)
    if ext not in LOADERS:
        raise ValueError(f"Unsupported extension {ext!r}; known: {sorted(LOADERS)}")
    return LOADERS[ext](str(path))


def write(path: str | Path, cloud: Cloud) -> None:
    """Dispatch writer by file extension."""
    ext = _ext(path)
    if ext not in WRITERS:
        raise ValueError(f"No writer for extension {ext!r}; supported: {sorted(WRITERS)}")
    WRITERS[ext](str(path), cloud)
