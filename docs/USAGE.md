# Usage guide

## Install

```bash
pip install splat-trimmer
```

Verify:

```bash
splat-trimmer --version
```

## Common workflows

### "I just want the floaters gone"

```bash
splat-trimmer trim scene.ply --percentile 99
```

This computes the 99th-percentile-clipped AABB and writes
`scene.trimmed.ply` next to the input.

### "I know exactly which box I want"

```bash
splat-trimmer trim scene.ply \
  --output scene.trimmed.ply \
  --aabb "-2,-2,-1,2,2,3"
```

### "Crop to a region of interest"

```bash
splat-trimmer trim scene.ply \
  --output object.ply \
  --sphere "0.3,0.2,1.5,0.8"
```

### "Drop loud outliers before computing the bbox"

```bash
splat-trimmer trim scene.ply \
  --drop-outliers 99 \
  --percentile 99
```

### "Subsample for the web viewer"

```bash
splat-trimmer decimate scene.ply --target 500000 --output scene.web.ply
```

## Library

See `examples/` for runnable snippets. The public API surface is small:

```python
from splat_trimmer.io.registry import load, write
from splat_trimmer.bounds.aabb import compute_aabb
from splat_trimmer.bounds.sphere import bounding_sphere
from splat_trimmer.ops.crop_aabb import crop_aabb
from splat_trimmer.ops.crop_sphere import crop_sphere
from splat_trimmer.ops.crop_oriented import crop_oriented_bbox
from splat_trimmer.ops.percentile import drop_outliers
from splat_trimmer.ops.decimate import decimate
from splat_trimmer.stats import summarise
```

Every op returns a new `Cloud` and never mutates the input.
