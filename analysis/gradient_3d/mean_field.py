"""Local mean-field response to a receptor-density gradient."""

from __future__ import annotations

import math


def attached_fraction(k_d: float, accessible_receptors: float, density: float) -> float:
    """Equilibrium attached fraction at the specified receptor density."""

    if k_d < 0.0 or accessible_receptors < 0.0 or density < 0.0:
        raise ValueError("k_d, accessible_receptors and density must be nonnegative")
    scale = accessible_receptors * density
    return 0.0 if k_d + scale == 0.0 else scale / (k_d + scale)


def binding_order(
    k_d: float,
    accessible_receptors: float,
    dense_density: float,
    sparse_density: float,
) -> float:
    """Normalized difference in attachment probability across the particle."""

    dense = attached_fraction(k_d, accessible_receptors, dense_density)
    sparse = attached_fraction(k_d, accessible_receptors, sparse_density)
    return (dense - sparse) / max(dense + sparse, 1.0e-300)


def binding_free_energy_force(
    n_binders: int,
    bond_work: float,
    particle_length: float,
    k_d: float,
    accessible_receptors: float,
    dense_density: float,
    sparse_density: float,
) -> float:
    """Force obtained from the binding free-energy difference across a particle."""

    if n_binders < 0 or bond_work < 0.0 or particle_length <= 0.0:
        raise ValueError(
            "n_binders and bond_work must be nonnegative and particle_length positive"
        )
    numerator = k_d + accessible_receptors * dense_density
    denominator = k_d + accessible_receptors * sparse_density
    if numerator <= 0.0 or denominator <= 0.0:
        return 0.0
    return float(
        n_binders * bond_work * math.log(numerator / denominator) / particle_length
    )


def binding_drift(
    n_binders: int,
    mean_force: float,
    attached_probability: float,
    particle_friction: float,
    attachment_friction: float,
) -> float:
    """Average drift after accounting for the finite number of attachments."""

    if n_binders <= 0:
        return 0.0
    if not 0.0 <= attached_probability <= 1.0:
        raise ValueError("attached_probability must lie between zero and one")
    if particle_friction <= 0.0 or attachment_friction < 0.0:
        raise ValueError(
            "particle_friction must be positive and attachment_friction nonnegative"
        )
    if attached_probability == 0.0:
        return 0.0
    force_per_attachment = mean_force / (n_binders * attached_probability)
    velocity = 0.0
    for n_attached in range(1, n_binders + 1):
        probability = (
            math.comb(n_binders, n_attached)
            * attached_probability**n_attached
            * (1.0 - attached_probability) ** (n_binders - n_attached)
        )
        velocity += (
            probability
            * n_attached
            * force_per_attachment
            / (particle_friction + n_attached * attachment_friction)
        )
    return float(velocity)


def polarized_active_drift(uniform_speed: float, order: float) -> float:
    """Project the uniform polarized speed onto the receptor gradient."""

    if uniform_speed < 0.0 or not -1.0 <= order <= 1.0:
        raise ValueError(
            "uniform_speed must be nonnegative and order must lie between -1 and 1"
        )
    return float(uniform_speed * order)
