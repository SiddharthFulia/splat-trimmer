# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-01

### Added

- `.ply` (INRIA 3DGS) and `.splat` (antimatter15) read + write.
- `.ksplat` (mkkellogg) read at compression level 0.
- `.spz` (PlayCanvas) loader stub with a helpful error message.
- AABB compute with optional percentile clipping.
- Ritter's bounding-sphere implementation.
- AABB, sphere and PCA-oriented bbox crop ops.
- Centroid-distance outlier drop and uniform-random decimation.
- `splat-trimmer info | trim | decimate | fit-sphere` CLI.
- Stats summary (count, AABB, centroid, std, diagonal).
- Tiny hand-crafted PLY and `.splat` fixtures.
- pytest suite covering round-trip IO, bounds, and ops.
- GitHub Actions CI: ruff, mypy strict, pytest on 3.11 and 3.12.
