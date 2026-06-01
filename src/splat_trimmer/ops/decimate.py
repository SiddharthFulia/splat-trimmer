from __future__ import annotations

import numpy as np

from splat_trimmer.models import Cloud


def decimate(
    cloud: Cloud,
    target: int,
    seed: int | None = None,
) -> Cloud:
    """Uniformly subsample `cloud` to at most `target` splats. Deterministic with `seed`."""
    if target < 0:
        raise ValueError(f"target must be non-negative; got {target}")
    n = cloud.count
    if target >= n:
        return cloud

    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(n, size=target, replace=False)
    mask = np.zeros(n, dtype=np.bool_)
    mask[keep_idx] = True
    return cloud.mask(mask)
