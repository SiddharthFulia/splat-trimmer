"""Crop a cloud to its PCA-aligned oriented bounding box."""

from __future__ import annotations

import sys
from pathlib import Path

from splat_trimmer.io.registry import load, write
from splat_trimmer.io.write import suggest_output_name
from splat_trimmer.ops.crop_oriented import crop_oriented_bbox


def main(in_path: str) -> None:
    cloud = load(in_path)
    print(f"loaded {cloud.count:,} splats")
    trimmed = crop_oriented_bbox(cloud, margin=0.02)
    out = suggest_output_name(in_path, suffix="obb")
    write(out, trimmed)
    print(f"wrote {out}  ({trimmed.count:,} kept)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/oriented-bbox.py path/to/scene.ply")
        sys.exit(1)
    main(str(Path(sys.argv[1])))
