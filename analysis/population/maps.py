"""Rebuild the explore, escape and exploit maps from the mean-field model.

The continuous population calculation uses the uniform-landscape transport
equations. Gradient guidance is calculated on a compact grid of stopped-path
solutions and interpolated with shape-preserving cubic splines; held-out
solutions measure the interpolation accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from analysis.uniform_3d.mean_field import GAUSSIAN_BOUND_WEIGHT, MU, load_inputs

from .transport_mean_field import OUTPUT_NAMES, evaluate_batch, params_to_array


LENGTH = 20.0
PHI_GRID = np.unique(
    np.r_[np.geomspace(0.0025, 0.99, 96), 1.0 - np.geomspace(0.01, 0.08, 13)]
)
CLEAVAGE_GRID = np.geomspace(0.001, 2.5, 90)
ALLOCATION_GRID = np.linspace(0.0, 1.0, 65)
ARCHITECTURES = ("polarized", "mixed")
SITE_COUNT_FLOOR = 0.2

OUTPUT = {name: index for index, name in enumerate(OUTPUT_NAMES)}


@dataclass(frozen=True)
class MapFields:
    values: dict[str, dict[str, np.ndarray]]
    references: dict[str, float]


def _inputs() -> SimpleNamespace:
    uniform = load_inputs()
    return SimpleNamespace(
        L=uniform.L,
        alpha=uniform.alpha,
        d_rec=uniform.d_rec,
        spring_k=uniform.spring_k,
        kbt=uniform.kbt,
        cleavage_exposure_factor=uniform.cleavage_exposure_factor,
        gamma_parallel=uniform.gamma_parallel,
        gamma_perp=uniform.gamma_perp,
        gamma_rot=uniform.gamma_rot,
        D_parallel=uniform.D_parallel,
        D_perp=uniform.D_perp,
        D_rot0=uniform.D_rot0,
        nu_b=uniform.nu_b,
        nu_c=uniform.nu_c,
        bound_weight=GAUSSIAN_BOUND_WEIGHT,
        mu=MU,
        observation_time=1_000_000.0,
    )


def _physical_coordinates(
    binder_fraction: float, nu_b: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = max(LENGTH * binder_fraction, SITE_COUNT_FLOOR)
    upper = LENGTH - SITE_COUNT_FLOOR
    span = max(upper - lower, 0.0)
    binder_line = lower + span * ALLOCATION_GRID
    n_binders = np.broadcast_to(
        binder_line[None, :], (len(CLEAVAGE_GRID), len(ALLOCATION_GRID))
    ).copy()
    valid = np.full(n_binders.shape, upper >= lower, dtype=bool)
    k_d = np.maximum(
        nu_b * (n_binders / (LENGTH * binder_fraction) - 1.0), 1.0e-9
    )
    n_cleavers = np.maximum(LENGTH - n_binders, 1.0e-9)
    k_c = CLEAVAGE_GRID[:, None] * LENGTH / n_cleavers
    return (
        np.where(valid, n_binders, np.nan),
        np.where(valid, k_d, np.nan),
        np.where(valid, k_c, np.nan),
        valid,
    )


def _guidance_tensor(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    extrema = pd.read_csv(path.with_name("gradient_ci_extremum_evaluations.csv"))
    required = {
        "architecture",
        "phi_index",
        "chi_C_index",
        "allocation_index",
        "exact_CI_path",
    }
    if not required.issubset(extrema.columns):
        raise ValueError("Exact guidance-extremum table has an invalid schema")
    log_phi = np.log10(PHI_GRID)
    log_cleavage = np.log10(CLEAVAGE_GRID)
    result: dict[str, np.ndarray] = {}
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        values = data["values"][architecture_index]
        values = PchipInterpolator(
            data["log_phi_grid"], values, axis=0, extrapolate=False
        )(log_phi)
        values = PchipInterpolator(
            data["log_writer_grid"], values, axis=1, extrapolate=False
        )(log_cleavage)
        values = PchipInterpolator(
            data["allocation_grid"], values, axis=2, extrapolate=False
        )(ALLOCATION_GRID)
        # Direct evaluations at neighboring points preserve the broad maximum
        # used to normalize the guidance score.
        values = np.clip(values, 0.0, 1.0).astype(np.float32)
        selected = extrema.loc[extrema["architecture"].eq(architecture)]
        for row in selected.itertuples(index=False):
            values[
                int(row.phi_index),
                int(row.chi_C_index),
                int(row.allocation_index),
            ] = float(row.exact_CI_path)
        result[architecture] = values
    return result


def calculate_map_fields(guidance_tensor: Path) -> MapFields:
    """Evaluate all molecular states and normalize the displayed functions."""

    inputs = _inputs()
    parameters = params_to_array(inputs)
    guidance = _guidance_tensor(guidance_tensor)
    shape = (len(PHI_GRID), len(CLEAVAGE_GRID), len(ALLOCATION_GRID))
    coordinate_rows = [
        _physical_coordinates(float(phi), inputs.nu_b) for phi in PHI_GRID
    ]
    all_n_binders = np.stack([row[0] for row in coordinate_rows])
    all_valid = np.stack([row[3] for row in coordinate_rows])
    fields: dict[str, dict[str, np.ndarray]] = {}

    for architecture_index, architecture in enumerate(ARCHITECTURES):
        block = {
            name: np.full(shape, np.nan, dtype=np.float32)
            for name in (
                "n_binders",
                "range_shift",
                "guidance",
                "attachment_probability",
                "cleavage_probability",
                "local_support_loss_probability",
                "escape_probability",
                "passive_probability",
                "exploit_raw",
            )
        }
        for phi_index, phi in enumerate(PHI_GRID):
            n_binders, k_d, k_c, valid = coordinate_rows[phi_index]
            flat_valid = valid.ravel()
            architecture_codes = np.full(
                int(np.sum(flat_valid)), architecture_index, dtype=np.int8
            )
            active = np.full((flat_valid.size, len(OUTPUT_NAMES)), np.nan)
            inactive = np.full_like(active, np.nan)
            active[flat_valid] = evaluate_batch(
                architecture_codes,
                k_d.ravel()[flat_valid],
                k_c.ravel()[flat_valid],
                n_binders.ravel()[flat_valid],
                parameters,
            )
            inactive[flat_valid] = evaluate_batch(
                architecture_codes,
                k_d.ravel()[flat_valid],
                np.zeros(int(np.sum(flat_valid))),
                n_binders.ravel()[flat_valid],
                parameters,
            )
            active = active.reshape(len(CLEAVAGE_GRID), len(ALLOCATION_GRID), -1)
            inactive = inactive.reshape(
                len(CLEAVAGE_GRID), len(ALLOCATION_GRID), -1
            )
            range_shift = (
                active[..., OUTPUT["effective_diffusivity"]]
                - inactive[..., OUTPUT["effective_diffusivity"]]
            )
            mean_attachments = np.full_like(n_binders, LENGTH * float(phi))
            attached_fraction = np.clip(
                mean_attachments / np.maximum(n_binders, 1.0e-12), 0.0, 1.0
            )
            no_attachment = np.exp(
                n_binders * np.log(np.maximum(1.0 - attached_fraction, 1.0e-300))
            )
            attachment = 1.0 - no_attachment
            n_cleavers = np.maximum(LENGTH - n_binders, 0.0)
            total_cleaving_rate = (
                inputs.cleavage_exposure_factor
                * k_c
                * math.sqrt(math.pi)
                * inputs.alpha
                * n_cleavers
            )
            cleavage = np.clip(
                -np.expm1(
                    -np.minimum(
                        total_cleaving_rate
                        * active[..., OUTPUT["persistence_time"]],
                        700.0,
                    )
                ),
                0.0,
                1.0,
            )
            support_loss = (
                np.clip(
                    active[..., OUTPUT["local_support_loss_probability"]], 0.0, 1.0
                )
                if architecture == "mixed"
                else np.zeros_like(n_binders)
            )
            attachment_load = np.clip(
                active[..., OUTPUT["attachment_friction"]]
                / np.maximum(active[..., OUTPUT["total_friction"]], 1.0e-300),
                0.0,
                1.0,
            )
            escape = no_attachment * cleavage + attachment * support_loss
            passive = no_attachment * (1.0 - cleavage)
            exploit = (
                attachment
                * (1.0 - support_loss)
                * attachment_load
                * (1.0 - cleavage)
            )
            block["n_binders"][phi_index] = n_binders
            block["range_shift"][phi_index] = range_shift
            block["guidance"][phi_index] = guidance[architecture][phi_index]
            block["attachment_probability"][phi_index] = attachment
            block["cleavage_probability"][phi_index] = cleavage
            block["local_support_loss_probability"][phi_index] = support_loss
            block["escape_probability"][phi_index] = escape
            block["passive_probability"][phi_index] = passive
            block["exploit_raw"][phi_index] = exploit
        fields[architecture] = block

    controlled = (
        (all_n_binders >= 1.0)
        & (LENGTH - all_n_binders >= 1.0)
        & (np.broadcast_to(ALLOCATION_GRID[None, None, :], shape) > 0.0)
        & all_valid
    )
    range_reference = max(
        float(
            np.nanmax(
                np.where(
                    controlled,
                    np.maximum(fields[architecture]["range_shift"], 0.0),
                    np.nan,
                )
            )
        )
        for architecture in ARCHITECTURES
    )
    exploit_reference = max(
        float(
            np.nanmax(
                np.where(controlled, fields[architecture]["exploit_raw"], np.nan)
            )
        )
        for architecture in ARCHITECTURES
    )
    guidance_reference = max(
        float(
            np.nanmax(
                np.where(controlled, fields[architecture]["guidance"], np.nan)
            )
        )
        for architecture in ARCHITECTURES
    )
    for architecture in ARCHITECTURES:
        block = fields[architecture]
        survival = np.clip(1.0 - block["escape_probability"], 0.0, 1.0)
        block["exploration_range_score"] = (
            np.clip(
                np.maximum(block["range_shift"], 0.0) / range_reference, 0.0, 1.0
            )
            * survival
        )
        block["gradient_guidance_score"] = (
            np.clip(block["guidance"] / guidance_reference, 0.0, 1.0) * survival
        )
        block["escape_score"] = np.clip(
            block["escape_probability"], 0.0, 1.0
        )
        block["exploit_score"] = np.clip(
            block["exploit_raw"] / exploit_reference, 0.0, 1.0
        )
        block["passive_score"] = np.clip(
            block["passive_probability"], 0.0, 1.0
        )
    return MapFields(
        values=fields,
        references={
            "range": range_reference,
            "guidance": guidance_reference,
            "exploit": exploit_reference,
        },
    )


def _continuous_envelope(
    values: np.ndarray, n_binders: np.ndarray
) -> np.ndarray:
    result = np.full(values.shape[:2], np.nan, dtype=float)
    for phi_index in range(values.shape[0]):
        finite = np.all(np.isfinite(values[phi_index]), axis=0)
        if np.count_nonzero(finite) < 2:
            continue
        lower_count = float(n_binders[phi_index, 0, 0])
        upper_count = float(n_binders[phi_index, 0, -1])
        span = upper_count - lower_count
        if span <= 0.0:
            continue
        lower = max(0.0, (1.0 - lower_count) / span)
        upper = min(1.0, ((LENGTH - 1.0) - lower_count) / span)
        if upper < lower:
            continue
        dense_allocation = np.linspace(lower, upper, 257)
        curves = PchipInterpolator(
            ALLOCATION_GRID[finite],
            np.asarray(values[phi_index, :, finite], dtype=float).T,
            axis=1,
        )(dense_allocation)
        result[phi_index] = np.max(curves, axis=1)
    return np.clip(result, 0.0, 1.0)


def rebuild_function_maps(guidance_tensor: Path) -> pd.DataFrame:
    """Return the exact table used for the two population function maps."""

    calculated = calculate_map_fields(guidance_tensor)
    rows: list[dict[str, float | str]] = []
    score_names = (
        "exploration_range_score",
        "gradient_guidance_score",
        "escape_score",
        "exploit_score",
        "passive_score",
    )
    for architecture in ARCHITECTURES:
        block = calculated.values[architecture]
        envelopes = {
            name: _continuous_envelope(block[name], block["n_binders"])
            for name in score_names
        }
        for cleavage_index, cleavage in enumerate(CLEAVAGE_GRID):
            for phi_index, phi in enumerate(PHI_GRID):
                scores = {
                    name: float(envelopes[name][phi_index, cleavage_index])
                    for name in score_names
                }
                if not all(np.isfinite(value) for value in scores.values()):
                    scores = {
                        "exploration_range_score": 0.0,
                        "gradient_guidance_score": 0.0,
                        "escape_score": 0.0,
                        "exploit_score": 0.0,
                        "passive_score": 1.0,
                    }
                rows.append(
                    {
                        "architecture": architecture,
                        "phi_b": float(phi),
                        "chi_C": float(cleavage),
                        **scores,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_representative_states(
    guidance_tensor: Path, states: pd.DataFrame
) -> pd.DataFrame:
    """Recalculate all function scores at the declared panel-c support states."""

    calculated = calculate_map_fields(guidance_tensor)
    rows: list[dict[str, float | str]] = []
    for state in states.itertuples(index=False):
        phi_matches = np.flatnonzero(np.isclose(PHI_GRID, float(state.phi_b)))
        cleavage_matches = np.flatnonzero(
            np.isclose(CLEAVAGE_GRID, float(state.chi_C))
        )
        if phi_matches.size != 1 or cleavage_matches.size != 1:
            raise ValueError("A representative state does not lie on the map grid")
        phi_index = int(phi_matches[0])
        cleavage_index = int(cleavage_matches[0])
        block = calculated.values[str(state.architecture)]
        allocation_index = int(
            np.nanargmin(
                np.abs(
                    block["n_binders"][phi_index, cleavage_index]
                    - float(state.N_b)
                )
            )
        )
        rebuilt_n_binders = float(
            block["n_binders"][phi_index, cleavage_index, allocation_index]
        )
        if not math.isclose(rebuilt_n_binders, float(state.N_b), abs_tol=2.0e-6):
            raise ValueError("A representative binder allocation is not on the map grid")
        spreading = float(
            block["exploration_range_score"][
                phi_index, cleavage_index, allocation_index
            ]
        )
        sensing = float(
            block["gradient_guidance_score"][
                phi_index, cleavage_index, allocation_index
            ]
        )
        rows.append(
            {
                "function": str(state.function),
                "architecture": str(state.architecture),
                "phi_b": float(state.phi_b),
                "chi_C": float(state.chi_C),
                "N_b": rebuilt_n_binders,
                "spreading_score": spreading,
                "sensing_score": sensing,
                "explore_score": max(spreading, sensing),
                "escape_score": float(
                    block["escape_score"][phi_index, cleavage_index, allocation_index]
                ),
                "exploit_score": float(
                    block["exploit_score"][phi_index, cleavage_index, allocation_index]
                ),
            }
        )
    return pd.DataFrame(rows)
