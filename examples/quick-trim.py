"""Trim a captured scene by clipping its 99th-percentile AABB."""

from __future__ import annotations

import sys
from pathlib import Path

from splat_trimmer.bounds.aabb import compute_aabb
from splat_trimmer.io.registry import load, write
from splat_trimmer.io.write import suggest_output_name
from splat_trimmer.ops.crop_aabb import crop_aabb


def main(in_path: str) -> None:
    cloud = load(in_path)
    print(f"loaded {cloud.count:,} splats")
    mn, mx = compute_aabb(cloud.positions, percentile=99.0)
    print(f"clipped AABB: {mn.tolist()} -> {mx.tolist()}")
    trimmed = crop_aabb(cloud, mn, mx)
    out = suggest_output_name(in_path)
    write(out, trimmed)
    print(f"wrote {out}  ({trimmed.count:,} splats kept)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/quick-trim.py path/to/scene.ply")
        sys.exit(1)
    main(str(Path(sys.argv[1])))
