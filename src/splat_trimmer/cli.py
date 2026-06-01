from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from splat_trimmer.bounds.aabb import compute_aabb
from splat_trimmer.bounds.sphere import bounding_sphere
from splat_trimmer.io.registry import load, write
from splat_trimmer.io.write import suggest_output_name
from splat_trimmer.ops.crop_aabb import crop_aabb
from splat_trimmer.ops.crop_sphere import crop_sphere
from splat_trimmer.ops.decimate import decimate as decimate_op
from splat_trimmer.ops.percentile import drop_outliers
from splat_trimmer.stats import summarise
from splat_trimmer.version import __version__

app = typer.Typer(
    help="Trim Gaussian splat clouds (AABB, sphere, OBB, decimation).",
    add_completion=False,
)
console = Console()


def _parse_aabb(spec: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 6:
        raise typer.BadParameter("AABB must be 'minx,miny,minz,maxx,maxy,maxz'")
    mn = np.asarray(parts[:3], dtype=np.float32)
    mx = np.asarray(parts[3:], dtype=np.float32)
    if np.any(mn >= mx):
        raise typer.BadParameter(f"AABB min must be strictly < max; got {mn=} {mx=}")
    return mn, mx


def _parse_sphere(spec: str) -> tuple[np.ndarray, float]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("Sphere must be 'cx,cy,cz,radius'")
    center = np.asarray(parts[:3], dtype=np.float32)
    radius = float(parts[3])
    if radius <= 0:
        raise typer.BadParameter(f"Sphere radius must be positive; got {radius}")
    return center, radius


@app.command()
def info(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Print summary statistics for a splat file."""
    cloud = load(path)
    stats = summarise(cloud)
    t = Table(title=f"{path.name}  ({cloud.source_format})")
    t.add_column("key")
    t.add_column("value")
    for k, v in stats.to_dict().items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def trim(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    percentile: Annotated[
        float | None,
        typer.Option(
            "--percentile",
            "-p",
            help="Auto AABB clipped at the p-th / (100-p)-th percentile per axis (e.g. 99).",
        ),
    ] = None,
    aabb: Annotated[
        str | None,
        typer.Option("--aabb", help="Explicit AABB: 'minx,miny,minz,maxx,maxy,maxz'."),
    ] = None,
    sphere: Annotated[
        str | None,
        typer.Option("--sphere", help="Sphere crop: 'cx,cy,cz,radius'."),
    ] = None,
    drop: Annotated[
        float | None,
        typer.Option(
            "--drop-outliers",
            help="Drop the outermost (100 - X)% of points by centroid distance first.",
        ),
    ] = None,
) -> None:
    """Trim a splat cloud by AABB, sphere or percentile, and write the result."""
    cloud = load(path)
    n_before = cloud.count
    console.print(f"Loaded {n_before:,} splats from {path}")

    if drop is not None:
        cloud = drop_outliers(cloud, percentile=drop)
        console.print(f"After --drop-outliers {drop}: {cloud.count:,} splats")

    if aabb is not None:
        mn, mx = _parse_aabb(aabb)
    elif percentile is not None:
        mn, mx = compute_aabb(cloud.positions, percentile=percentile)
        console.print(f"Auto AABB (p={percentile}): {mn.tolist()} -> {mx.tolist()}")
    else:
        mn, mx = None, None  # type: ignore[assignment]

    if mn is not None:
        cloud = crop_aabb(cloud, mn, mx)
        console.print(f"After AABB crop: {cloud.count:,} splats")

    if sphere is not None:
        c, r = _parse_sphere(sphere)
        cloud = crop_sphere(cloud, c, r)
        console.print(f"After sphere crop: {cloud.count:,} splats")

    out_path = output or suggest_output_name(path)
    write(out_path, cloud)
    pct = (cloud.count / n_before * 100.0) if n_before else 0.0
    console.print(
        f"[green]Wrote[/green] {out_path}  "
        f"({cloud.count:,} / {n_before:,} splats kept, {pct:.1f}%)"
    )


@app.command()
def decimate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    target: Annotated[int, typer.Option("--target", "-t", help="Target splat count.")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="RNG seed.")] = None,
) -> None:
    """Randomly subsample a splat cloud to `target` points."""
    cloud = load(path)
    n_before = cloud.count
    cloud = decimate_op(cloud, target=target, seed=seed)
    out_path = output or suggest_output_name(path, suffix="decimated")
    write(out_path, cloud)
    console.print(
        f"[green]Wrote[/green] {out_path}  ({cloud.count:,} / {n_before:,} splats kept)"
    )


@app.command()
def fit_sphere(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Print the Ritter bounding sphere for the cloud."""
    cloud = load(path)
    c, r = bounding_sphere(cloud.positions)
    console.print(f"center: {c.tolist()}")
    console.print(f"radius: {r}")


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":  # pragma: no cover
    app()
