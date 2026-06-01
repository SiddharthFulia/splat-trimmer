"""Subsample a cloud to half its size."""

from __future__ import annotations

import sys
from pathlib import Path

from splat_trimmer.io.registry import load, write
from splat_trimmer.io.write import suggest_output_name
from splat_trimmer.ops.decimate import decimate


def main(in_path: str) -> None:
    cloud = load(in_path)
    target = max(cloud.count // 2, 1)
    print(f"decimating {cloud.count:,} -> {target:,}")
    sub = decimate(cloud, target=target, seed=42)
    out = suggest_output_name(in_path, suffix="half")
    write(out, sub)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/decimate-half.py path/to/scene.ply")
        sys.exit(1)
    main(str(Path(sys.argv[1])))
