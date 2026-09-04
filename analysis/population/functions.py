"""Population-function definitions from the manuscript Methods."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def attachment_probability(n_binders: float, k_d: float, nu_b: float) -> float:
    """Probability that at least one binder is attached."""

    if n_binders < 0.0 or k_d < 0.0 or nu_b < 0.0:
        raise ValueError("n_binders, k_d and nu_b must be nonnegative")
    if k_d + nu_b == 0.0:
        return 0.0
    return float(1.0 - (k_d / (k_d + nu_b)) ** n_binders)


def cleavage_probability(rate: float, persistence_time: float) -> float:
    """Probability of at least one cleavage event during a persistent run."""

    if rate < 0.0 or persistence_time < 0.0:
        raise ValueError("rate and persistence_time must be nonnegative")
    return float(-math.expm1(-rate * persistence_time))


def escape_probability(
    p_attachment: float,
    p_cleavage: float,
    p_local_depletion: float,
) -> float:
    """Combine weak-attachment and local-depletion routes to escape."""

    values = np.asarray([p_attachment, p_cleavage, p_local_depletion], dtype=float)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie between zero and one")
    return float((1.0 - p_attachment) * p_cleavage + p_attachment * p_local_depletion)


def exploit_score(
    p_attachment: float,
    p_local_depletion: float,
    attachment_friction: float,
    total_friction: float,
    p_cleavage: float,
) -> float:
    """Persistent-attachment score before normalization over parameter space."""

    if total_friction <= 0.0 or attachment_friction < 0.0:
        raise ValueError(
            "friction coefficients must be nonnegative and total_friction positive"
        )
    probabilities = np.asarray(
        [p_attachment, p_local_depletion, p_cleavage], dtype=float
    )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie between zero and one")
    return float(
        p_attachment
        * (1.0 - p_local_depletion)
        * (attachment_friction / total_friction)
        * (1.0 - p_cleavage)
    )


def optimal_allocation(
    capacities: Mapping[str, float],
    demands: Mapping[str, float],
) -> dict[str, float]:
    """Allocate population fractions to maximize the least-covered function."""

    if set(capacities) != set(demands) or not capacities:
        raise ValueError(
            "capacities and demands must contain the same nonempty set of functions"
        )
    ratios = {}
    for name, capacity in capacities.items():
        demand = float(demands[name])
        capacity = float(capacity)
        if capacity <= 0.0 or demand <= 0.0:
            raise ValueError("capacities and demands must be positive")
        ratios[name] = demand / capacity
    total = sum(ratios.values())
    return {name: value / total for name, value in ratios.items()}


def demand_sweep_allocation(
    capacities: Mapping[str, float],
    explore_to_exploit_ratios: np.ndarray,
) -> list[dict[str, float]]:
    """Population allocations for the demand sweep shown in the manuscript.

    Escape demand is held at one. Explore and exploit demands are the square
    root and inverse square root of the requested ratio, so only their relative
    demand changes.
    """

    rows: list[dict[str, float]] = []
    for ratio in np.asarray(explore_to_exploit_ratios, dtype=float):
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("explore-to-exploit demand ratios must be positive")
        demands = {
            "explore": math.sqrt(float(ratio)),
            "escape": 1.0,
            "exploit": 1.0 / math.sqrt(float(ratio)),
        }
        fractions = optimal_allocation(capacities, demands)
        rows.append(
            {
                "explore_to_exploit_demand_ratio": float(ratio),
                **{f"{name}_fraction": fractions[name] for name in capacities},
                **{
                    f"{name}_contribution": fractions[name] * float(capacities[name])
                    for name in capacities
                },
                **{f"{name}_demand": demands[name] for name in capacities},
            }
        )
    return rows


def projected_population_density(
    fractions: Mapping[str, float],
    shared_explore_escape_center: float,
    exploit_center: float,
    coordinate: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Project a simplex allocation onto the displayed binder--cleaver axis.

    Explore and escape share one molecular coordinate but differ in particle
    architecture. Exploit uses the second coordinate. The common Gaussian
    width is fixed by setting its full width at half maximum to the separation
    between the two molecular coordinates; the width affects only display.
    Every component is normalized over the finite displayed interval, so its
    area equals the corresponding simplex weight.
    """

    if set(fractions) != {"explore", "escape", "exploit"}:
        raise ValueError("fractions must contain explore, escape and exploit")
    x = np.asarray(coordinate, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("coordinate must be a strictly increasing 1D array")
    separation = abs(float(exploit_center) - float(shared_explore_escape_center))
    sigma = separation / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if sigma <= 0.0:
        raise ValueError("the two molecular coordinates must differ")

    def kernel(center: float) -> np.ndarray:
        values = np.exp(-0.5 * np.square((x - float(center)) / sigma))
        return values / max(float(np.trapz(values, x)), 1.0e-300)

    shared = kernel(shared_explore_escape_center)
    exploit = kernel(exploit_center)
    components = {
        "explore_density": float(fractions["explore"]) * shared,
        "escape_density": float(fractions["escape"]) * shared,
        "exploit_density": float(fractions["exploit"]) * exploit,
    }
    return {
        **components,
        "total_density": sum(components.values()),
        "display_width": sigma,
    }


def reader_cleaver_coordinate(
    phi_b: float, chi_c: float, offset: float = 0.003
) -> float:
    """Return the logarithmic coordinate used to display population states."""

    if phi_b < 0.0 or chi_c < 0.0 or offset <= 0.0:
        raise ValueError("phi_b and chi_c must be nonnegative and offset positive")
    return float(math.log10((phi_b + offset) / (chi_c + offset)))
