"""Brownian-motion primitives for the particle model.

The spring energy is ``U = 0.5 * |r_ligand - r_receptor|^2`` in simulation
units.  The reversible rate ratio has the corresponding Boltzmann distance
dependence when the reference energy is ``alpha^2 / 2``.  The imposed
background diffusivities are specified separately and may be reduced by the
factor ``f_T`` used in the manuscript.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def reversible_thermal_energy_from_alpha(alpha: float) -> float:
    """Return the reference energy implied by the reversible rate ratio."""
    return 0.5 * alpha * alpha


@njit(cache=True)
def thermal_diffusion_constants(
    thermal_energy: float,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_rot: float,
) -> Tuple[float, float, float]:
    """Return translational and rotational diffusivities from kBT/gamma."""
    return (
        thermal_energy / gamma_parallel,
        thermal_energy / gamma_perp,
        thermal_energy / gamma_rot,
    )


@njit(cache=True)
def _standard_normal_pair_from_uniforms() -> Tuple[float, float]:
    """Return two standard normals while consuming exactly two uniform draws."""
    u1 = np.random.random()
    if u1 <= 1.0e-300:
        u1 = 1.0e-300
    u2 = np.random.random()
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    return radius * np.cos(theta), radius * np.sin(theta)


@njit(cache=True)
def _six_standard_normals_from_six_uniforms() -> Tuple[
    float, float, float, float, float, float
]:
    """
    Return six standard normals while consuming exactly six uniform draws.

    Constructing the normals explicitly keeps each trajectory reproducible
    from its seed.
    """
    z0, z1 = _standard_normal_pair_from_uniforms()
    z2, z3 = _standard_normal_pair_from_uniforms()
    z4, z5 = _standard_normal_pair_from_uniforms()
    return z0, z1, z2, z3, z4, z5


@njit(cache=True)
def _normalize3(x: float, y: float, z: float) -> Tuple[float, float, float]:
    norm = np.sqrt(x * x + y * y + z * z)
    if norm <= 0.0:
        return 1.0, 0.0, 0.0
    inv = 1.0 / norm
    return x * inv, y * inv, z * inv


@njit(cache=True)
def _rotate_unit_vector_by_rotation_vector(
    nx: float,
    ny: float,
    nz: float,
    ox: float,
    oy: float,
    oz: float,
) -> Tuple[float, float, float]:
    angle = np.sqrt(ox * ox + oy * oy + oz * oz)
    if angle <= 1e-15:
        return _normalize3(nx, ny, nz)

    ax = ox / angle
    ay = oy / angle
    az = oz / angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dot = ax * nx + ay * ny + az * nz
    cross_x = ay * nz - az * ny
    cross_y = az * nx - ax * nz
    cross_z = ax * ny - ay * nx

    rx = nx * cos_a + cross_x * sin_a + ax * dot * (1.0 - cos_a)
    ry = ny * cos_a + cross_y * sin_a + ay * dot * (1.0 - cos_a)
    rz = nz * cos_a + cross_z * sin_a + az * dot * (1.0 - cos_a)
    return _normalize3(rx, ry, rz)


@njit(cache=True)
def bound_force_and_weighted_force_rod_3d(
    x: float,
    y: float,
    z: float,
    nx: float,
    ny: float,
    nz: float,
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    """Return total force F and weighted force A = sum_i s_i F_i."""
    nx, ny, nz = _normalize3(nx, ny, nz)
    fx = 0.0
    fy = 0.0
    fz = 0.0
    ax = 0.0
    ay = 0.0
    az = 0.0
    for k in range(len(bound_ligand_idx)):
        lig_idx = int(bound_ligand_idx[k])
        s = float(r_i[lig_idx])
        lx = x + s * nx
        ly = y + s * ny
        lz = z + s * nz
        dx = float(bound_rx[k]) - lx
        dy = float(bound_ry[k]) - ly
        dz = float(bound_rz[k]) - lz
        fx += dx
        fy += dy
        fz += dz
        ax += s * dx
        ay += s * dy
        az += s * dz
    return fx, fy, fz, ax, ay, az


@njit(cache=True)
def brownian_kick_free_rod_3d(
    x: float,
    y: float,
    z: float,
    nx: float,
    ny: float,
    nz: float,
    dt: float,
    diffusion_parallel: float,
    diffusion_perp: float,
    diffusion_rot: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Apply one overdamped Brownian kick to a free rigid rod.

    Translation samples covariance
    ``2 dt [D_perp I + (D_parallel - D_perp) n n^T]``. Orientation is updated by
    a Gaussian small rotation and Rodrigues' formula, preserving unit norm.
    """
    nx, ny, nz = _normalize3(nx, ny, nz)
    if dt <= 0.0:
        return x, y, z, nx, ny, nz

    g_tx, g_ty, g_tz, g_rx, g_ry, g_rz = _six_standard_normals_from_six_uniforms()
    g_parallel = g_tx * nx + g_ty * ny + g_tz * nz
    sqrt_parallel = np.sqrt(max(0.0, 2.0 * diffusion_parallel * dt))
    sqrt_perp = np.sqrt(max(0.0, 2.0 * diffusion_perp * dt))

    dx = sqrt_perp * (g_tx - g_parallel * nx) + sqrt_parallel * g_parallel * nx
    dy = sqrt_perp * (g_ty - g_parallel * ny) + sqrt_parallel * g_parallel * ny
    dz = sqrt_perp * (g_tz - g_parallel * nz) + sqrt_parallel * g_parallel * nz

    sqrt_rot = np.sqrt(max(0.0, 2.0 * diffusion_rot * dt))
    ox = sqrt_rot * g_rx
    oy = sqrt_rot * g_ry
    oz = sqrt_rot * g_rz
    rx, ry, rz = _rotate_unit_vector_by_rotation_vector(nx, ny, nz, ox, oy, oz)

    return x + dx, y + dy, z + dz, rx, ry, rz


@njit(cache=True)
def _ou_relaxation_mean_factor(dt: float, rate: float) -> float:
    """Return (1 - exp(-rate * dt)) / rate with the rate -> 0 limit."""
    if rate <= 1.0e-300:
        return dt
    arg = rate * dt
    if arg > 700.0:
        return 1.0 / rate
    return (1.0 - np.exp(-arg)) / rate


@njit(cache=True)
def _ou_relaxation_noise_sigma(dt: float, rate: float, diffusion: float) -> float:
    """Return the exact OU noise standard deviation for dX=-rate X dt + sqrt(2D)dW."""
    if diffusion <= 0.0 or dt <= 0.0:
        return 0.0
    if rate <= 1.0e-300:
        return np.sqrt(max(0.0, 2.0 * diffusion * dt))
    arg = 2.0 * rate * dt
    if arg > 700.0:
        variance = diffusion / rate
    else:
        variance = diffusion * (1.0 - np.exp(-arg)) / rate
    return np.sqrt(max(0.0, variance))


@njit(cache=True)
def brownian_dynamics_step_bound_rod_relaxation_ou_3d(
    x: float,
    y: float,
    z: float,
    nx: float,
    ny: float,
    nz: float,
    dt: float,
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    thermal_energy: float,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_rot: float,
    min_gamma: float,
    passive_diffusion_parallel: float = -1.0,
    passive_diffusion_perp: float = -1.0,
    passive_diffusion_rot: float = -1.0,
) -> Tuple[float, float, float, float, float, float]:
    """
    Stable overdamped Brownian step for a rod tethered by fixed harmonic bonds.

    For fixed anchors, the translational spring modes are linear over a frozen
    orientation, and the small-angle rotational spring mode has stiffness
    ``sum_i s_i^2``. The update integrates these relaxation factors exactly
    over ``dt`` and uses the corresponding Ornstein--Uhlenbeck noise variance.
    """
    nx, ny, nz = _normalize3(nx, ny, nz)
    if dt <= 0.0:
        return x, y, z, nx, ny, nz

    n_bound = len(bound_ligand_idx)
    if n_bound <= 0:
        diffusion_parallel, diffusion_perp, diffusion_rot = thermal_diffusion_constants(
            thermal_energy,
            max(float(gamma_parallel), float(min_gamma)),
            max(float(gamma_perp), float(min_gamma)),
            max(float(gamma_rot), float(min_gamma)),
        )
        if passive_diffusion_parallel >= 0.0:
            diffusion_parallel = passive_diffusion_parallel
        if passive_diffusion_perp >= 0.0:
            diffusion_perp = passive_diffusion_perp
        if passive_diffusion_rot >= 0.0:
            diffusion_rot = passive_diffusion_rot
        return brownian_kick_free_rod_3d(
            x,
            y,
            z,
            nx,
            ny,
            nz,
            dt,
            diffusion_parallel,
            diffusion_perp,
            diffusion_rot,
        )

    gamma_parallel_eff = max(float(gamma_parallel), float(min_gamma))
    gamma_perp_eff = max(float(gamma_perp), float(min_gamma))
    gamma_rot_eff = max(float(gamma_rot), float(min_gamma))
    diffusion_parallel, diffusion_perp, diffusion_rot = thermal_diffusion_constants(
        thermal_energy,
        gamma_parallel_eff,
        gamma_perp_eff,
        gamma_rot_eff,
    )
    if passive_diffusion_parallel >= 0.0:
        diffusion_parallel = passive_diffusion_parallel
    if passive_diffusion_perp >= 0.0:
        diffusion_perp = passive_diffusion_perp
    if passive_diffusion_rot >= 0.0:
        diffusion_rot = passive_diffusion_rot

    fx, fy, fz, ax, ay, az = bound_force_and_weighted_force_rod_3d(
        x,
        y,
        z,
        nx,
        ny,
        nz,
        r_i,
        bound_ligand_idx,
        bound_rx,
        bound_ry,
        bound_rz,
    )

    g_tx, g_ty, g_tz, g_rx, g_ry, g_rz = _six_standard_normals_from_six_uniforms()
    g_parallel = g_tx * nx + g_ty * ny + g_tz * nz
    n_bound_float = float(n_bound)

    f_dot_n = fx * nx + fy * ny + fz * nz
    f_par_x = f_dot_n * nx
    f_par_y = f_dot_n * ny
    f_par_z = f_dot_n * nz
    f_perp_x = fx - f_par_x
    f_perp_y = fy - f_par_y
    f_perp_z = fz - f_par_z

    rate_parallel = n_bound_float / gamma_parallel_eff
    rate_perp = n_bound_float / gamma_perp_eff
    mean_factor_parallel = (
        _ou_relaxation_mean_factor(dt, rate_parallel) / gamma_parallel_eff
    )
    mean_factor_perp = _ou_relaxation_mean_factor(dt, rate_perp) / gamma_perp_eff
    sigma_parallel = _ou_relaxation_noise_sigma(dt, rate_parallel, diffusion_parallel)
    sigma_perp = _ou_relaxation_noise_sigma(dt, rate_perp, diffusion_perp)

    noise_x = sigma_perp * (g_tx - g_parallel * nx) + sigma_parallel * g_parallel * nx
    noise_y = sigma_perp * (g_ty - g_parallel * ny) + sigma_parallel * g_parallel * ny
    noise_z = sigma_perp * (g_tz - g_parallel * nz) + sigma_parallel * g_parallel * nz

    x_next = x + mean_factor_parallel * f_par_x + mean_factor_perp * f_perp_x + noise_x
    y_next = y + mean_factor_parallel * f_par_y + mean_factor_perp * f_perp_y + noise_y
    z_next = z + mean_factor_parallel * f_par_z + mean_factor_perp * f_perp_z + noise_z

    sum_s2 = 0.0
    for k in range(n_bound):
        lig_idx = int(bound_ligand_idx[k])
        if 0 <= lig_idx < len(r_i):
            s = float(r_i[lig_idx])
            sum_s2 += s * s
    rate_rot = sum_s2 / gamma_rot_eff
    rot_mean_factor = _ou_relaxation_mean_factor(dt, rate_rot) / gamma_rot_eff
    sigma_rot = _ou_relaxation_noise_sigma(dt, rate_rot, diffusion_rot)
    torque_x = ny * az - nz * ay
    torque_y = nz * ax - nx * az
    torque_z = nx * ay - ny * ax
    rot_x = rot_mean_factor * torque_x + sigma_rot * g_rx
    rot_y = rot_mean_factor * torque_y + sigma_rot * g_ry
    rot_z = rot_mean_factor * torque_z + sigma_rot * g_rz
    nx_next, ny_next, nz_next = _rotate_unit_vector_by_rotation_vector(
        nx,
        ny,
        nz,
        rot_x,
        rot_y,
        rot_z,
    )

    return x_next, y_next, z_next, nx_next, ny_next, nz_next


def validate_thermal_inputs(
    thermal_energy: float,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_rot: float,
) -> None:
    """Fail early for invalid physical parameters before entering numba kernels."""
    if thermal_energy <= 0.0:
        raise ValueError("thermal_energy must be positive")
    if gamma_parallel <= 0.0:
        raise ValueError("gamma_parallel must be positive")
    if gamma_perp <= 0.0:
        raise ValueError("gamma_perp must be positive")
    if gamma_rot <= 0.0:
        raise ValueError("gamma_rot must be positive")
