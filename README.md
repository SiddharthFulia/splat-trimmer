# splat-trimmer

A Python CLI and library for trimming Gaussian splat clouds.

`splat-trimmer` loads `.ply` (INRIA format) and `.splat` (antimatter15 packed
format) files, computes the axis-aligned bounding box of the populated splats
(with optional percentile clipping to ignore stray outliers), applies user
supplied AABB or sphere crops, and writes the trimmed cloud back out.

It's useful for cleaning captured 3D Gaussian splat scenes before publishing —
you almost always want to drop the "background sky" splats that drift far away
from the scene of interest and bloat the file.

## Why?

After running a 3DGS optimization, the trained scene often contains:

- Floaters: low-opacity splats hanging in empty space far from any surface.
- Sky / boundary splats: huge, low-opacity bubbles approximating the
  far-field. Cheap to skip when you only care about an object or room.
- Outlier blow-up: a handful of splats with positions in the 1e4 range.

Trimming these out can shrink a `.ply` from 350 MB to 60 MB with no perceptual
loss inside the region of interest. `splat-trimmer` is designed to make that
trim a one-liner.

## Supported formats

| Extension | Format                         | Read | Write |
| --------- | ------------------------------ | ---- | ----- |
| `.ply`    | INRIA 3DGS                     | yes  | yes   |
| `.splat`  | antimatter15 packed            | yes  | yes   |
| `.ksplat` | mkkellogg KSplat               | yes  | no    |
| `.spz`    | PlayCanvas compressed (stub)   | no   | no    |

See [docs/FORMATS.md](docs/FORMATS.md) for the byte layouts.

## Install

```bash
pip install splat-trimmer
```

Or from source:

```bash
git clone https://github.com/SiddharthFulia/splat-trimmer
cd splat-trimmer
pip install -e .
```

## CLI usage

### Inspect

```bash
splat-trimmer info scene.ply
```

Prints splat count, AABB, centroid, std-dev per axis and which fields are
present.

### Trim with auto AABB + percentile clip

The 99th-percentile clip drops the outermost 1% of points along each axis
before computing the AABB. This is the most common "throw out the floaters"
workflow.

```bash
splat-trimmer trim scene.ply --output scene.trimmed.ply --percentile 99
```

### Trim to an explicit AABB

```bash
splat-trimmer trim scene.ply \
  --output scene.trimmed.ply \
  --aabb "-1.5,-1.5,-1.5,1.5,1.5,1.5"
```

### Trim to a sphere

```bash
splat-trimmer trim scene.ply \
  --output scene.trimmed.ply \
  --sphere "0,0,0,2.0"
```

### Decimate (random subsample)

```bash
splat-trimmer decimate scene.ply --output half.ply --target 500000
```

## Library usage

```python
from splat_trimmer.io.registry import load, write
from splat_trimmer.bounds.aabb import compute_aabb
from splat_trimmer.ops.crop_aabb import crop_aabb

cloud = load("scene.ply")
mn, mx = compute_aabb(cloud.positions, percentile=99.0)
trimmed = crop_aabb(cloud, mn, mx)
write("scene.trimmed.ply", trimmed)
```

### Sphere crop

```python
from splat_trimmer.bounds.sphere import bounding_sphere
from splat_trimmer.ops.crop_sphere import crop_sphere

center, radius = bounding_sphere(cloud.positions)
trimmed = crop_sphere(cloud, center, radius * 0.95)
```

### Oriented bounding box

```python
from splat_trimmer.ops.crop_oriented import crop_oriented_bbox

trimmed = crop_oriented_bbox(cloud, margin=0.05)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
