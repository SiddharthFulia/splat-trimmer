# File formats

`splat-trimmer` aims to be format-agnostic. This document describes the
on-disk layouts of every format we touch — useful both for contributors and
for anyone trying to debug a malformed file.

## INRIA `.ply`

Binary little-endian PLY with one vertex element per splat. Field order is
not enforced by the PLY spec, but the INRIA training code emits them in a
predictable sequence:

| Field            | Type | Meaning                                                    |
| ---------------- | ---- | ---------------------------------------------------------- |
| `x`, `y`, `z`    | f32  | World-space position                                       |
| `nx`, `ny`, `nz` | f32  | Unused normals (kept for downstream tool compatibility)    |
| `f_dc_0..2`      | f32  | Degree-0 spherical harmonic coeffs (per RGB channel)       |
| `f_rest_0..44`   | f32  | Higher-order SH coeffs (degrees 1-3, 3 channels x 15 each) |
| `opacity`        | f32  | Pre-sigmoid logit                                          |
| `scale_0..2`     | f32  | Log-space per-axis scale                                   |
| `rot_0..3`       | f32  | Quaternion (w, x, y, z)                                    |

Some scenes train at SH degree < 3 and emit fewer `f_rest_*` fields. The
loader gracefully accepts whatever's present.

## antimatter15 `.splat`

A 32-byte-per-splat flat binary blob. No header — just `N * 32` bytes.

```
offset   bytes   type      field
0        12      f32 x 3   position
12       12      f32 x 3   scale (linear, not log)
24       4       u8 x 4    rgba (8-bit, premultiplied)
28       4       u8 x 4    quaternion (b - 128) / 128 in [-1, 1]
```

This is the format used by https://github.com/antimatter15/splat for web
viewers — its big draw is incremental streaming. The trade-off is loss of
higher-order SH and log-scale precision.

## mkkellogg `.ksplat`

Used by https://github.com/mkkellogg/GaussianSplats3D. Has a 4096-byte fixed
header followed by N sections. Each section has its own splat count and
stride, allowing different compression levels per section.

Compression levels:

- `0`: raw float32, ~44 bytes/splat (pos f32x3 + scale f32x3 + rot f32x4 + rgba u8x4)
- `1`: float16 positions / scales, ~22 bytes/splat
- `2`: int8 quantised, ~14 bytes/splat (with scene center stored in header)

`splat-trimmer` currently only reads level 0. Levels 1 and 2 raise
`NotImplementedError` — convert your file using the upstream KSplat tools.

## PlayCanvas `.spz` (stub)

Compressed 3DGS format with variable-bit encoding for positions, scales,
rotations, and SH. The `.spz` loader is intentionally a stub for v0.1;
implementing the full bit-level decoder is on the roadmap.

If you have a `.spz` file today, decode it to `.ply` with the upstream
[supersplat](https://github.com/playcanvas/supersplat) tooling, then run
`splat-trimmer` on the decoded file.
