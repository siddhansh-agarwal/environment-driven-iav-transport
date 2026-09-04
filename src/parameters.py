"""Particle ligand layouts used by the simulations."""

from __future__ import annotations

import numpy as np


def distribute_ligands(n_total: int, n_binders: int, pattern: str) -> np.ndarray:
    """Return the fixed axial binder/cleaver layout.

    ``True`` denotes a binder and ``False`` a cleaver. Polarized particles
    contain one contiguous binder block followed by one cleaver block. In the
    mixed layout, cleavers occupy both endpoints and any remaining cleavers
    are distributed approximately uniformly over the interior sites. The
    layout is deterministic, so architecture does not add another source of
    trajectory-to-trajectory variation.
    """

    if n_total < 1:
        raise ValueError("n_total must be positive")
    if not 0 <= n_binders <= n_total:
        raise ValueError(f"Invalid n_binders ({n_binders}) for n_total={n_total}")
    if pattern not in {"polarized", "mixed"}:
        raise ValueError("pattern must be 'polarized' or 'mixed'")

    n_cleavers = n_total - n_binders
    if pattern == "polarized":
        return np.array([True] * n_binders + [False] * n_cleavers, dtype=np.bool_)
    if n_binders == 0:
        return np.zeros(n_total, dtype=np.bool_)
    if n_cleavers == 0:
        return np.ones(n_total, dtype=np.bool_)

    ligands = np.ones(n_total, dtype=np.bool_)
    if n_cleavers == 1:
        ligands[-1] = False
        return ligands

    cleaver_positions = np.array([0, n_total - 1], dtype=int)
    if n_cleavers > 2:
        interior = np.linspace(1, n_total - 2, n_cleavers)[1:-1]
        cleaver_positions = np.concatenate(
            (cleaver_positions, np.rint(interior).astype(int))
        )
    cleaver_positions = np.unique(cleaver_positions)
    if len(cleaver_positions) < n_cleavers:
        available = np.setdiff1d(
            np.arange(1, n_total - 1, dtype=int),
            cleaver_positions,
            assume_unique=True,
        )
        cleaver_positions = np.concatenate(
            (cleaver_positions, available[: n_cleavers - len(cleaver_positions)])
        )
    ligands[cleaver_positions] = False
    return ligands
