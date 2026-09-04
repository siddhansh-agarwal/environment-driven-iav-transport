"""
Overdamped rigid-particle dynamics for binder--cleaver simulations.

Physics for a rigid rod:
- Anisotropic translational friction: γ_∥ (parallel to rod), γ_⊥ (perpendicular)
- Rotational friction: γ_R (rotation about any axis perpendicular to rod)
- Orientation represented by unit vector n̂

Equations of motion:
    F_∥ = (F · n̂) n̂
    F_⊥ = F - F_∥
    v = F_∥/γ_∥ + F_⊥/γ_⊥

    τ = Σ (s_i n̂) × F_i   (torque from bonds)
    ω = τ / γ_R
    dn̂/dt = ω × n̂
"""

import numpy as np
from numba import njit
from typing import Tuple

from .geometry3d import normalize_vector


@njit(cache=True)
def _build_bound_pose_coupling_terms_3d(
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
) -> Tuple[int, float, float, float, float, float, float, float]:
    """
    Build pose-coupled invariants for the fixed-bound overdamped rod ODE.

    For a fixed bound set during one Gillespie interval:
      C = Σ R_k
      D = Σ r_k R_k
      s1 = Σ r_k
    """
    n_bonds = len(bound_ligand_idx)
    c_x = 0.0
    c_y = 0.0
    c_z = 0.0
    d_x = 0.0
    d_y = 0.0
    d_z = 0.0
    s1 = 0.0

    for k in range(n_bonds):
        lig_idx = int(bound_ligand_idx[k])
        rx = float(bound_rx[k])
        ry = float(bound_ry[k])
        rz = float(bound_rz[k])
        weight = float(r_i[lig_idx])

        c_x += rx
        c_y += ry
        c_z += rz
        d_x += weight * rx
        d_y += weight * ry
        d_z += weight * rz
        s1 += weight

    return n_bonds, c_x, c_y, c_z, d_x, d_y, d_z, s1


@njit(cache=True)
def _orientation_alignment_exact_step(
    n_hat: np.ndarray,
    ax_total: float,
    ay_total: float,
    az_total: float,
    dt: float,
    gamma_R: float,
    min_gamma: float,
) -> np.ndarray:
    """
    Exact orientation step for dn/dt = (A - (A·n) n) / gamma_R with constant A.
    """
    n_next = n_hat.copy()
    if dt <= 0.0:
        return n_next

    a_mag_sq = ax_total * ax_total + ay_total * ay_total + az_total * az_total
    if a_mag_sq <= 1e-30:
        return n_next

    a_mag = np.sqrt(a_mag_sq)
    inv_a_mag = 1.0 / a_mag
    a_x = ax_total * inv_a_mag
    a_y = ay_total * inv_a_mag
    a_z = az_total * inv_a_mag
    u0 = n_hat[0] * a_x + n_hat[1] * a_y + n_hat[2] * a_z
    if u0 >= 1.0 - 1e-15:
        n_next[0] = a_x
        n_next[1] = a_y
        n_next[2] = a_z
        return n_next
    if u0 <= -1.0 + 1e-15:
        n_next[0] = -a_x
        n_next[1] = -a_y
        n_next[2] = -a_z
        return n_next

    p_x = n_hat[0] - u0 * a_x
    p_y = n_hat[1] - u0 * a_y
    p_z = n_hat[2] - u0 * a_z
    p_mag_sq = p_x * p_x + p_y * p_y + p_z * p_z
    kappa = a_mag * dt / max(gamma_R, min_gamma)
    exp_term = np.exp(2.0 * kappa)
    one_plus = 1.0 + u0
    one_minus = 1.0 - u0
    denom = one_plus * exp_term + one_minus
    if denom <= 1e-300:
        return n_next
    u1 = (one_plus * exp_term - one_minus) / denom
    u1 = min(1.0, max(-1.0, u1))
    p_scale = 0.0
    if p_mag_sq > 1e-30:
        p_scale = np.sqrt(max(0.0, 1.0 - u1 * u1) / p_mag_sq)

    n_next[0] = u1 * a_x + p_scale * p_x
    n_next[1] = u1 * a_y + p_scale * p_y
    n_next[2] = u1 * a_z + p_scale * p_z
    normalize_vector(n_next)
    return n_next


@njit(cache=True)
def _translation_step_fixed_orientation_pose_coupled(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    dt: float,
    n_bonds: int,
    c_x: float,
    c_y: float,
    c_z: float,
    s1: float,
    gamma_parallel: float,
    gamma_perp: float,
    min_gamma: float,
) -> Tuple[float, float, float]:
    """
    Exact translation for fixed orientation under the true spring-coupled ODE.

    x_dot = [mu_perp I + (mu_par - mu_perp) n n^T] (C - N_B x - s1 n)
    """
    if dt <= 0.0 or n_bonds <= 0:
        return x, y, z

    n_bonds_f = float(n_bonds)
    x_eq_x = (c_x - s1 * float(n_hat[0])) / n_bonds_f
    x_eq_y = (c_y - s1 * float(n_hat[1])) / n_bonds_f
    x_eq_z = (c_z - s1 * float(n_hat[2])) / n_bonds_f

    dx_eq_x = x - x_eq_x
    dx_eq_y = y - x_eq_y
    dx_eq_z = z - x_eq_z
    dx_eq_dot_n = (
        dx_eq_x * float(n_hat[0])
        + dx_eq_y * float(n_hat[1])
        + dx_eq_z * float(n_hat[2])
    )
    dx_par_x = dx_eq_dot_n * float(n_hat[0])
    dx_par_y = dx_eq_dot_n * float(n_hat[1])
    dx_par_z = dx_eq_dot_n * float(n_hat[2])
    dx_perp_x = dx_eq_x - dx_par_x
    dx_perp_y = dx_eq_y - dx_par_y
    dx_perp_z = dx_eq_z - dx_par_z

    exp_par = np.exp(-n_bonds_f * dt / max(gamma_parallel, min_gamma))
    exp_perp = np.exp(-n_bonds_f * dt / max(gamma_perp, min_gamma))
    return (
        x_eq_x + exp_par * dx_par_x + exp_perp * dx_perp_x,
        x_eq_y + exp_par * dx_par_y + exp_perp * dx_perp_y,
        x_eq_z + exp_par * dx_par_z + exp_perp * dx_perp_z,
    )


@njit(cache=True)
def _all_bonds_valid_sq(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    max_bond_length_sq: float,
) -> bool:
    for k in range(len(bound_ligand_idx)):
        lig_idx = int(bound_ligand_idx[k])
        x_lig = x + r_i[lig_idx] * n_hat[0]
        y_lig = y + r_i[lig_idx] * n_hat[1]
        z_lig = z + r_i[lig_idx] * n_hat[2]
        ddx = bound_rx[k] - x_lig
        ddy = bound_ry[k] - y_lig
        ddz = bound_rz[k] - z_lig
        if ddx * ddx + ddy * ddy + ddz * ddz > max_bond_length_sq:
            return False
    return True


@njit(cache=True)
def _evaluate_bound_vectors_strang_split_state(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    tau: float,
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_R: float,
    min_gamma: float,
) -> Tuple[float, float, float, np.ndarray]:
    """Evaluate one pose-coupled Strang interval for the fixed-bound ODE."""
    (
        n_bonds,
        c_x,
        c_y,
        c_z,
        d_x,
        d_y,
        d_z,
        s1,
    ) = _build_bound_pose_coupling_terms_3d(
        r_i,
        bound_ligand_idx,
        bound_rx,
        bound_ry,
        bound_rz,
    )
    if n_bonds <= 0 or tau <= 0.0:
        return x, y, z, n_hat.copy()

    n_half = _orientation_alignment_exact_step(
        n_hat,
        d_x - s1 * x,
        d_y - s1 * y,
        d_z - s1 * z,
        0.5 * tau,
        gamma_R,
        min_gamma,
    )
    x_mid, y_mid, z_mid = _translation_step_fixed_orientation_pose_coupled(
        x,
        y,
        z,
        n_half,
        tau,
        n_bonds,
        c_x,
        c_y,
        c_z,
        s1,
        gamma_parallel,
        gamma_perp,
        min_gamma,
    )
    n_next = _orientation_alignment_exact_step(
        n_half,
        d_x - s1 * x_mid,
        d_y - s1 * y_mid,
        d_z - s1 * z_mid,
        0.5 * tau,
        gamma_R,
        min_gamma,
    )
    return x_mid, y_mid, z_mid, n_next


@njit(cache=True)
def _advance_bound_vectors_strang_split_with_time(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    tau: float,
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_R: float,
    min_gamma: float,
    max_bond_length_sq: float,
) -> Tuple[float, float, float, np.ndarray, bool, float]:
    """
    Advance the pose-coupled fixed-bound interval with Strang splitting and
    bracket the first bond-validity crossing, if any.
    """
    x_end, y_end, z_end, n_end = _evaluate_bound_vectors_strang_split_state(
        x,
        y,
        z,
        n_hat,
        tau,
        r_i,
        bound_ligand_idx,
        bound_dx,
        bound_dy,
        bound_dz,
        bound_rx,
        bound_ry,
        bound_rz,
        gamma_parallel,
        gamma_perp,
        gamma_R,
        min_gamma,
    )
    if _all_bonds_valid_sq(
        x_end,
        y_end,
        z_end,
        n_end,
        r_i,
        bound_ligand_idx,
        bound_rx,
        bound_ry,
        bound_rz,
        max_bond_length_sq,
    ):
        return x_end, y_end, z_end, n_end, False, tau

    lo = 0.0
    hi = tau
    x_hi = x_end
    y_hi = y_end
    z_hi = z_end
    n_hi = n_end.copy()
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        x_mid, y_mid, z_mid, n_mid = _evaluate_bound_vectors_strang_split_state(
            x,
            y,
            z,
            n_hat,
            mid,
            r_i,
            bound_ligand_idx,
            bound_dx,
            bound_dy,
            bound_dz,
            bound_rx,
            bound_ry,
            bound_rz,
            gamma_parallel,
            gamma_perp,
            gamma_R,
            min_gamma,
        )
        if _all_bonds_valid_sq(
            x_mid,
            y_mid,
            z_mid,
            n_mid,
            r_i,
            bound_ligand_idx,
            bound_rx,
            bound_ry,
            bound_rz,
            max_bond_length_sq,
        ):
            lo = mid
        else:
            hi = mid
            x_hi = x_mid
            y_hi = y_mid
            z_hi = z_mid
            n_hi = n_mid.copy()

    return x_hi, y_hi, z_hi, n_hi, True, hi


@njit(cache=True)
def _advance_bound_vectors_strang_split_with_flag(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    tau: float,
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_R: float,
    min_gamma: float,
    max_bond_length_sq: float,
) -> Tuple[float, float, float, np.ndarray, bool]:
    x_new, y_new, z_new, n_new, broke_early, _ = (
        _advance_bound_vectors_strang_split_with_time(
            x,
            y,
            z,
            n_hat,
            tau,
            r_i,
            bound_ligand_idx,
            bound_dx,
            bound_dy,
            bound_dz,
            bound_rx,
            bound_ry,
            bound_rz,
            gamma_parallel,
            gamma_perp,
            gamma_R,
            min_gamma,
            max_bond_length_sq,
        )
    )
    return x_new, y_new, z_new, n_new, broke_early


@njit(cache=True)
def update_position_3d_pending_frozen_with_flag(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    tau: float,
    r_i: np.ndarray,
    ligand_indices: np.ndarray,
    force_x: np.ndarray,
    force_y: np.ndarray,
    force_z: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_R: float,
    min_gamma: float = 1e-6,
    max_bond_length_sq: float = 1.0,
) -> Tuple[float, float, float, np.ndarray, bool]:
    """Continue a split event interval using its saved bond geometry."""
    if len(ligand_indices) == 0:
        return x, y, z, n_hat.copy(), False
    return _advance_bound_vectors_strang_split_with_flag(
        x,
        y,
        z,
        n_hat,
        tau,
        r_i,
        ligand_indices,
        force_x,
        force_y,
        force_z,
        bound_rx,
        bound_ry,
        bound_rz,
        gamma_parallel,
        gamma_perp,
        gamma_R,
        min_gamma,
        max_bond_length_sq,
    )


@njit(cache=True)
def update_position_3d_bound_vectors(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    tau: float,
    r_i: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_R: float,
    min_gamma: float = 1e-6,
    max_bond_length_sq: float = 1.0,
) -> Tuple[float, float, float, np.ndarray]:
    """Advance the particle pose over one event interval with fixed bonds."""
    if len(bound_ligand_idx) == 0:
        return x, y, z, n_hat.copy()
    x_new, y_new, z_new, n_new, _ = _advance_bound_vectors_strang_split_with_flag(
        x,
        y,
        z,
        n_hat,
        tau,
        r_i,
        bound_ligand_idx,
        bound_dx,
        bound_dy,
        bound_dz,
        bound_rx,
        bound_ry,
        bound_rz,
        gamma_parallel,
        gamma_perp,
        gamma_R,
        min_gamma,
        max_bond_length_sq,
    )
    return x_new, y_new, z_new, n_new
