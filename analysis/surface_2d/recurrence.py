"""Geometry-resolved trail writing and long-time planar recurrence."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import lambertw

from .transport_mean_field import Geometry, SurfaceInputs


def cleaved_trail_width(
    kc: float,
    directional_time: float,
    persistence_length: float,
    geometry: Geometry,
    inputs: SurfaceInputs,
) -> dict[str, float]:
    """Effective width of the cleaved trail used in the main-text theory.

    The total cleavage accumulated during one directional interval is shared
    over the binder regions that can support a returning particle.  Independent
    cleavage events give Poisson coverage of the finite geometric footprint,
    retaining the linear weak-cleavage limit and saturating at the width set by
    the ligand layout and interaction range.
    """

    if directional_time < 0.0 or persistence_length <= 0.0 or kc < 0.0:
        raise ValueError(
            "kc and directional_time must be nonnegative and "
            "persistence_length positive"
        )
    n_binders = int(geometry.n_binders)
    n_cleavers = int(geometry.x_cleavers.size)
    if n_binders <= 0:
        raise ValueError("the surface recurrence theory requires at least one binder")

    if geometry.pattern == "polarized":
        footprint_width = 2.0 * inputs.alpha
    elif geometry.pattern == "mixed":
        span = (
            float(np.ptp(geometry.x_cleavers))
            if geometry.x_cleavers.size > 1
            else 0.0
        )
        footprint_width = span + 2.0 * inputs.alpha
    else:
        raise ValueError(f"unsupported architecture: {geometry.pattern}")

    integrated_cleavage_area = (
        inputs.cleavage_exposure_factor
        * float(kc)
        * n_cleavers
        * math.pi
        * inputs.alpha**2
        * float(directional_time)
    )
    linear_stopping_area = integrated_cleavage_area / n_binders
    footprint_area = float(persistence_length) * footprint_width
    coverage = -math.expm1(-linear_stopping_area / footprint_area)
    effective_width = footprint_width * coverage
    return {
        "integrated_cleavage_area": integrated_cleavage_area,
        "linear_stopping_area": linear_stopping_area,
        "footprint_width": footprint_width,
        "footprint_coverage": coverage,
        "effective_trail_width": effective_width,
        "detachment_area": float(persistence_length) * effective_width,
    }


def recurrence_prediction(
    diffusion: float,
    tau: float,
    area: float,
    final_time: float,
) -> dict[str, float]:
    """Finite-time mean terminal range and its long-time Lambert limit."""
    diffusion = max(float(diffusion), 1.0e-300)
    tau = max(float(tau), 1.0e-300)
    area = max(float(area), 0.0)
    n_final = max(float(final_time) / tau, 0.0)
    free_range = math.sqrt(math.pi * diffusion * float(final_time))
    if area <= 0.0 or n_final <= 0.0:
        return {
            "epsilon_return": 0.0,
            "H_final": 0.0,
            "S_final": 1.0,
            "n_star": math.inf,
            "R_finite": free_range,
            "R_asymptotic": math.inf,
        }

    epsilon = area / (4.0 * math.pi * diffusion * tau)

    def exposure(n: float) -> float:
        return epsilon * ((n + 1.0) * math.log1p(n) - n)

    h_final = exposure(n_final)
    survival_final = math.exp(-min(h_final, 745.0))
    z_final = math.log1p(n_final)

    def integrand(z: float) -> float:
        n = math.expm1(z)
        if n <= 0.0:
            return 0.0
        h = exposure(n)
        return math.sqrt(n) * math.exp(-min(h, 745.0)) * epsilon * z * math.exp(z)

    stopped_part, _ = quad(
        integrand, 0.0, z_final, epsabs=1.0e-8, epsrel=2.0e-7, limit=300
    )
    scale = math.sqrt(math.pi * diffusion * tau)
    finite_range = scale * (survival_final * math.sqrt(n_final) + stopped_part)

    inverse = 1.0 / epsilon
    n_star = inverse / float(np.real(lambertw(inverse)))
    ell = math.sqrt(2.0 * diffusion * tau)
    b = math.sqrt(area)
    argument = 2.0 * math.pi * (ell / b) ** 2
    asymptotic = math.pi * ell**2 / (b * math.sqrt(float(np.real(lambertw(argument)))))
    return {
        "epsilon_return": epsilon,
        "H_final": h_final,
        "S_final": survival_final,
        "n_star": n_star,
        "R_finite": finite_range,
        "R_asymptotic": asymptotic,
        "R_free": free_range,
        "N_intervals": n_final,
    }


def predict_surface_range(
    pattern: str,
    kd: float,
    kc: float,
    n_binders: int,
    final_time: float,
    inputs: SurfaceInputs,
    geometries: dict[tuple[str, int], Geometry],
) -> dict[str, float | str]:
    """Evaluate the molecular transport and surface-return theory together."""

    from .transport_mean_field import surface_transport

    transport = surface_transport(
        pattern,
        kd,
        kc,
        n_binders,
        inputs,
        geometries,
    )
    geometry = geometries[(pattern, int(n_binders))]
    trail = cleaved_trail_width(
        kc=kc,
        directional_time=float(transport["tau_direction_2d"]),
        persistence_length=float(transport["ell_one_clock"]),
        geometry=geometry,
        inputs=inputs,
    )
    recurrence = recurrence_prediction(
        diffusion=float(transport["D_mobile_2d"]),
        tau=float(transport["tau_direction_2d"]),
        area=trail["detachment_area"],
        final_time=final_time,
    )
    return {
        **transport,
        **trail,
        **recurrence,
        "mean_field_range": recurrence["R_asymptotic"],
    }
