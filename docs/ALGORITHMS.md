# Algorithms

## Axis-aligned bounding box (AABB)

The literal AABB is `min` and `max` along each axis. In a captured 3DGS
scene this bbox is almost always blown out by a handful of floaters at
extreme distances. The percentile-clipped variant addresses this:

1. Compute the `(100 - p)`-th and `p`-th percentile of each coordinate.
2. Use those as the bbox corners.

With `p = 99` this rejects the outermost 1% of points along every axis. Cost
is O(N) — `numpy.percentile` handles it cheaply.

## Bounding sphere — Ritter's algorithm

Given an unordered set of N points in R^3, find a sphere `(center, radius)`
that contains all of them. The minimum enclosing sphere (MEC) is solvable
exactly in O(N) via Welzl, but Welzl is fiddly and recursive. Ritter's
algorithm gives a sphere typically ~5-10% larger than the MEC with O(N) time
and trivial implementation:

1. **Seed.** Pick any point P0. Find P1 farthest from P0. Find P2 farthest
   from P1. The initial sphere has center `(P1 + P2) / 2` and radius
   `||P1 - P2|| / 2`.
2. **Expand.** For each remaining point P:
   - if `||P - center|| <= radius`: continue.
   - else: grow the sphere just enough to include P, shifting `center`
     toward P by half the slack.

After one pass the sphere contains every input point. We add a tiny
relative epsilon to `radius` to absorb float round-off.

## Oriented bounding box via PCA

The PCA-aligned OBB is the AABB of the cloud after rotating into the
principal-component frame:

1. Centre the points (subtract centroid).
2. Build the 3x3 covariance matrix `C = X^T X / (N - 1)`.
3. Eigen-decompose `C`. The eigenvectors are the OBB axes.
4. Project the points onto those axes; the per-axis min/max is the OBB
   extent in the rotated frame.

This is *not* the minimum-volume OBB (which is NP-hard), but it's a great
default and runs in O(N) after the 3x3 eigen-decomposition. For most
captured scenes the PCA OBB is within a few percent of the optimal volume.

## Decimation

Pure random subsampling — `numpy.random.Generator.choice(N, target,
replace=False)`. We deliberately do *not* do voxel-grid downsampling
because it can collapse multiple distinct Gaussians into one voxel cell,
producing visible artefacts in the rendered output. For visualisation /
QA the random version is the right default; voxel-aware decimation could
be added later behind a `--mode voxel` flag.
