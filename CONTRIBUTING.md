# Contributing to splat-trimmer

Thanks for taking the time! This project is small and intentionally so —
keeping the dependency surface tight (`numpy`, `plyfile`, `typer`, `rich`) is
a deliberate design goal.

## Dev setup

```bash
git clone https://github.com/SiddharthFulia/splat-trimmer
cd splat-trimmer
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Quality bar

Before opening a PR please run:

```bash
ruff check src tests
mypy src
pytest -q
```

CI runs the same three steps on Python 3.11 and 3.12. PRs are gated on
green CI.

## Adding a format

1. Add a `src/splat_trimmer/io/<fmt>.py` exposing `load_<fmt>` and
   (optionally) `write_<fmt>`.
2. Register it in `src/splat_trimmer/io/registry.py`.
3. Document the on-disk layout in `docs/FORMATS.md`.
4. Add a tiny fixture under `fixtures/` and a round-trip test under
   `tests/`.

## Adding an op

Crops and decimations live in `src/splat_trimmer/ops/`. They should:

- Accept a `Cloud` and return a new `Cloud` (never mutate).
- Use `Cloud.mask(keep_mask)` to apply the selection — that handles all
  the per-field reindexing for you.
- Have a numpy-only fast path (no Python loops for million-splat clouds).

## Reporting bugs

Please attach the smallest possible repro file (under 1 MB) plus the exact
command you ran. If the file is sensitive, a synthetic cloud with the same
shape / range usually triggers the same path.
