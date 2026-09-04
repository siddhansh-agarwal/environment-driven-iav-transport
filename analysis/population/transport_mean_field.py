"""Mean-field transport model used for the population calculation.

The model combines the molecular inputs, binder and cleaver geometry,
attachment-induced friction, thermal motion, cleavage-driven motion and local
receptor depletion. Integer site counts describe the simulated particles;
continuous counts provide the interpolation used in the population map. The
cleavage-exposure factor is fixed by the uniform-environment calculation and
is unchanged in the population map.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.interpolate import PchipInterpolator


(
    P_L,
    P_ALPHA,
    P_DREC,
    P_SPRING,
    P_KBT,
    P_CLEAVAGE_EXPOSURE,
    P_GPAR,
    P_GPERP,
    P_GROT,
    P_DPAR,
    P_DPERP,
    P_DROT,
    P_NU_B,
    P_NU_C,
    P_BOUND_WEIGHT,
    P_MU,
    P_OBSERVATION_TIME,
) = range(17)


OUTPUT_NAMES = (
    "speed",
    "persistence_time",
    "coherent_speed_squared",
    "fluctuating_speed_variance",
    "binding_unbinding_diffusivity",
    "background_diffusivity",
    "active_diffusivity",
    "mobile_state_diffusivity",
    "effective_diffusivity",
    "mean_squared_displacement",
    "exploration_range",
    "cleavage_exposure",
    "local_depletion_entry_rate",
    "local_depletion_exit_rate",
    "mobile_fraction",
    "attachment_friction",
    "total_friction",
    "trail_segment_cleaving_probability",
    "expected_local_depletion_events",
    "cleaved_region_width",
    "remaining_receptor_fraction",
    "local_support_loss_probability",
    "locally_depleted_fraction",
)
N_OUTPUTS = len(OUTPUT_NAMES)


def params_to_array(inputs) -> np.ndarray:
    """Convert the microscopic inputs to the vector used by the solver."""

    observation_time = float(getattr(inputs, "observation_time", 1_000_000.0))
    return np.asarray(
        [
            inputs.L,
            inputs.alpha,
            inputs.d_rec,
            inputs.spring_k,
            inputs.kbt,
            inputs.cleavage_exposure_factor,
            inputs.gamma_parallel,
            inputs.gamma_perp,
            inputs.gamma_rot,
            inputs.D_parallel,
            inputs.D_perp,
            inputs.D_rot0,
            inputs.nu_b,
            inputs.nu_c,
            inputs.bound_weight,
            inputs.mu,
            observation_time,
        ],
        dtype=np.float64,
    )


# The mixed geometries are inversion symmetric. Their exact second
# moments at the three manuscript site counts, together with the empty and
# all-binder endpoints, define the continuous allocation.
_MIXED_NODES = np.asarray([0.0, 4.0, 10.0, 16.0, 20.0])
_MIXED_S2_NODES = np.asarray([0.0, 185.0, 346.5, 472.0, 665.0])
_MIXED_S2_TABLE = PchipInterpolator(_MIXED_NODES, _MIXED_S2_NODES)(
    np.arange(21.0)
).astype(np.float64)


@njit(cache=True, inline="always")
def _clip_exp(x: float) -> float:
    return math.exp(-min(max(x, 0.0), 700.0))


@njit(cache=True, inline="always")
def _n_cleavers(nb: float, p: np.ndarray) -> float:
    return max(p[P_L] - nb, 0.0)


@njit(cache=True, inline="always")
def _polarized_integer_moments(count: int, length: float) -> tuple[float, float]:
    """Exact first and second axial moments for an integer end block."""

    if count <= 0:
        return 0.0, 0.0
    n = float(count)
    a = -0.5 * length + 0.5
    s1 = n * a + 0.5 * n * (n - 1.0)
    s2 = n * a * a + a * n * (n - 1.0) + n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    return s1, max(s2, 0.0)


@njit(cache=True, inline="always")
def _geometry_moments(pattern: int, nb: float, p: np.ndarray) -> tuple[float, float]:
    n = min(max(nb, 0.0), p[P_L])
    if pattern == 0:
        # A continuous allocation represents an ensemble mixture of the two
        # neighbouring integer rods.  Convexly averaging their exact moment
        # matrices preserves positive attachment friction.
        low = int(math.floor(n))
        high = min(low + 1, int(round(p[P_L])))
        fraction = n - low
        s1_low, s2_low = _polarized_integer_moments(low, p[P_L])
        if high == low:
            return s1_low, s2_low
        s1_high, s2_high = _polarized_integer_moments(high, p[P_L])
        s1 = (1.0 - fraction) * s1_low + fraction * s1_high
        s2 = (1.0 - fraction) * s2_low + fraction * s2_high
        return s1, max(s2, 0.0)
    low = int(math.floor(n))
    high = min(low + 1, 20)
    fraction = n - low
    s2 = (1.0 - fraction) * _MIXED_S2_TABLE[low] + fraction * _MIXED_S2_TABLE[high]
    return 0.0, max(s2, 0.0)


@njit(cache=True, inline="always")
def _molecular_state(
    pattern: int, kd: float, nb: float, p: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    kd = max(kd, 1.0e-300)
    attached_fraction = p[P_NU_B] / (kd + p[P_NU_B])
    tau_off = p[P_BOUND_WEIGHT] / kd
    per_binder_friction = p[P_SPRING] * attached_fraction * tau_off
    s1, s2 = _geometry_moments(pattern, nb, p)

    bt = per_binder_friction * nb
    br = per_binder_friction * s2
    btr = per_binder_friction * s1
    zpar = p[P_GPAR] + bt
    f00 = p[P_GPERP] + bt
    f01 = btr
    f11 = p[P_GROT] + br
    determinant = max(f00 * f11 - f01 * f01, 1.0e-300)
    m00 = f11 / determinant
    m01 = -f01 / determinant
    m11 = f00 / determinant

    # Bond-noise matrix is (force variance / spring constant) times bond
    # friction.  The bath covariance is diagonal in translation and rotation.
    noise_scale = p[P_SPRING] * p[P_ALPHA] * p[P_ALPHA] / 2.0
    q00 = noise_scale * bt
    q01 = noise_scale * btr
    q11 = noise_scale * br
    dperp_bond = m00 * (q00 * m00 + q01 * m01) + m01 * (q01 * m00 + q11 * m01)
    drot_bond = m01 * (q00 * m01 + q01 * m11) + m11 * (q01 * m01 + q11 * m11)
    bath_q00 = p[P_DPERP] * p[P_GPERP] * p[P_GPERP]
    bath_q11 = p[P_DROT] * p[P_GROT] * p[P_GROT]
    dperp_bath = m00 * m00 * bath_q00 + m01 * m01 * bath_q11
    drot_bath = m01 * m01 * bath_q00 + m11 * m11 * bath_q11

    force_variance = p[P_SPRING] * p[P_SPRING] * p[P_ALPHA] * p[P_ALPHA] / 2.0
    force_covariance = nb * attached_fraction * force_variance
    dpar_bond = force_covariance * tau_off / max(zpar * zpar, 1.0e-300)
    dpar_bath = p[P_DPAR] * p[P_GPAR] * p[P_GPAR] / max(zpar * zpar, 1.0e-300)
    d_bond = (dpar_bond + 2.0 * dperp_bond) / 3.0
    d_bath = (dpar_bath + 2.0 * dperp_bath) / 3.0
    d_rot = max(drot_bond + drot_bath, 1.0e-300)
    tau = 1.0 / (2.0 * d_rot)
    zeta_direction = zpar if pattern == 0 else f00
    return (
        attached_fraction,
        tau_off,
        bt,
        zeta_direction,
        d_bond,
        d_bath,
        tau,
        dperp_bath,
        br,
        btr,
    )


@njit(cache=True, inline="always")
def _cleavage_frequency(pattern: int, kc: float, nb: float, p: np.ndarray) -> float:
    nc = _n_cleavers(nb, p)
    line_integral = math.sqrt(math.pi) * p[P_ALPHA]
    tracks = nc if pattern == 0 else nc / p[P_L]
    return p[P_CLEAVAGE_EXPOSURE] * max(kc, 0.0) * line_integral * tracks


@njit(cache=True, inline="always")
def _drive(
    pattern: int, kd: float, nb: float, exposure: float, p: np.ndarray
) -> tuple[float, float, float, float]:
    x = max(exposure, 0.0)
    kd = max(kd, 1.0e-300)
    if pattern == 0:
        rho_high = 1.0
        rho_low = _clip_exp(x)
        delta_g = math.log((kd + p[P_NU_B] * rho_high) / (kd + p[P_NU_B] * rho_low))
        phi_b = nb / p[P_L]
        phi_c = _n_cleavers(nb, p) / p[P_L]
        pair_availability = 4.0 * phi_b * phi_c
        force = p[P_KBT] * delta_g * pair_availability / (2.0 * p[P_ALPHA])
    else:
        rho_high = _clip_exp((1.0 - p[P_MU]) * x)
        rho_low = _clip_exp((1.0 + p[P_MU]) * x)
        delta_g = math.log((kd + p[P_NU_B] * rho_high) / (kd + p[P_NU_B] * rho_low))
        force = nb * p[P_KBT] * delta_g / (2.0 * p[P_ALPHA] * p[P_MU])
    return force, delta_g, rho_high, rho_low


@njit(cache=True, inline="always")
def _residual(
    pattern: int,
    velocity: float,
    kd: float,
    nb: float,
    omega: float,
    zeta: float,
    p: np.ndarray,
) -> float:
    force, _, _, _ = _drive(pattern, kd, nb, omega / max(velocity, 1.0e-300), p)
    return zeta * velocity - force


@njit(cache=True, inline="always")
def _bisect(
    pattern: int,
    low: float,
    high: float,
    kd: float,
    nb: float,
    omega: float,
    zeta: float,
    p: np.ndarray,
) -> float:
    f_low = _residual(pattern, low, kd, nb, omega, zeta, p)
    for _ in range(90):
        mid = 0.5 * (low + high)
        f_mid = _residual(pattern, mid, kd, nb, omega, zeta, p)
        if f_low * f_mid <= 0.0:
            high = mid
        else:
            low = mid
            f_low = f_mid
        if high - low <= 1.0e-12 + 1.0e-10 * abs(mid):
            break
    return 0.5 * (low + high)


@njit(cache=True, inline="always")
def _active_velocity(
    pattern: int, kd: float, kc: float, nb: float, zeta: float, p: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    omega = _cleavage_frequency(pattern, kc, nb, p)
    if omega <= 0.0 or _n_cleavers(nb, p) <= 0.0:
        _, delta_g, rho_high, rho_low = _drive(pattern, kd, nb, 0.0, p)
        return 0.0, 0.0, omega, delta_g, rho_high, rho_low

    previous_v = 1.0e-14
    previous_f = _residual(pattern, previous_v, kd, nb, omega, zeta, p)
    chosen = 0.0
    first_root = 0.0
    for i in range(1, 180):
        exponent = -14.0 + 17.0 * i / 179.0
        current_v = 10.0**exponent
        current_f = _residual(pattern, current_v, kd, nb, omega, zeta, p)
        if (
            math.isfinite(previous_f)
            and math.isfinite(current_f)
            and previous_f * current_f < 0.0
        ):
            root = _bisect(
                pattern,
                previous_v,
                current_v,
                kd,
                nb,
                omega,
                zeta,
                p,
            )
            if first_root <= 0.0 or root < first_root:
                first_root = root
            step = max(root * 1.0e-5, 1.0e-14)
            low = max(root - step, 1.0e-300)
            high = root + step
            slope = (
                _residual(pattern, high, kd, nb, omega, zeta, p)
                - _residual(pattern, low, kd, nb, omega, zeta, p)
            ) / (high - low)
            if slope > 0.0 and (chosen <= 0.0 or root < chosen):
                chosen = root
        previous_v = current_v
        previous_f = current_f
    velocity = chosen if chosen > 0.0 else first_root
    if velocity <= 0.0:
        _, delta_g, rho_high, rho_low = _drive(pattern, kd, nb, 0.0, p)
        return 0.0, 0.0, omega, delta_g, rho_high, rho_low
    exposure = omega / velocity
    _, delta_g, rho_high, rho_low = _drive(pattern, kd, nb, exposure, p)
    return velocity, exposure, omega, delta_g, rho_high, rho_low


@njit(cache=True, inline="always")
def evaluate_state(
    pattern: int, kd: float, kc: float, nb: float, p: np.ndarray
) -> np.ndarray:
    out = np.empty(N_OUTPUTS, dtype=np.float64)
    (
        attached_fraction,
        tau_off,
        bt,
        zeta,
        d_bond,
        d_bath,
        tau,
        _,
        _,
        _,
    ) = _molecular_state(pattern, kd, nb, p)
    velocity, exposure, omega, _, rho_high, rho_low = _active_velocity(
        pattern, kd, kc, nb, zeta, p
    )

    phi_b = nb / p[P_L]
    phi_c = _n_cleavers(nb, p) / p[P_L]
    interface = (2.0 * min(phi_b, phi_c)) ** 3
    if pattern == 0:
        speed_gate = 1.0
        response_time = p[P_GPAR] / p[P_SPRING]
        renewal_probability = -math.expm1(-response_time / max(tau_off, 1.0e-300))
        fluctuation_factor = 1.0 + (
            4.0
            * attached_fraction
            * (1.0 - attached_fraction)
            * interface
            * renewal_probability
        )
    else:
        speed_gate = 1.0 - (1.0 - attached_fraction) ** max(nb, 0.0)
        fluctuation_factor = 1.0 + (
            4.0 * attached_fraction * (1.0 - attached_fraction) * interface
        )
    coherent = velocity * velocity * speed_gate
    fluctuating = coherent * (fluctuation_factor - 1.0)

    # The cleaved-region overlap follows the uniform-environment calculation.
    p0_fresh = (max(kd, 1.0e-300) / (max(kd, 1.0e-300) + p[P_NU_B])) ** max(nb, 0.0)
    if kc > 0.0 and _n_cleavers(nb, p) > 0.0:
        coverage = phi_c**3
        written_width = 6.0 * p[P_ALPHA] * coverage
        residence = (
            written_width
            * (p[P_L] + 2.0 * p[P_ALPHA])
            / max(8.0 * (d_bath + d_bond), 1.0e-300)
        )
        cleavage_extent = p[P_CLEAVAGE_EXPOSURE] * kc * p[P_NU_C] * residence
    else:
        coverage = 0.0
        cleavage_extent = 0.0
    rho_support = _clip_exp(cleavage_extent)
    p0_cleaved = (
        max(kd, 1.0e-300) / (max(kd, 1.0e-300) + p[P_NU_B] * rho_support)
    ) ** max(nb, 0.0)
    support_loss = max(
        min((p0_cleaved - p0_fresh) / max(1.0 - p0_fresh, 1.0e-300), 1.0),
        0.0,
    )

    if pattern == 1 and kc > 0.0:
        span = p[P_L] + 2.0 * p[P_ALPHA]
        coherent_speed = math.sqrt(max(coherent, 0.0))
        directed_rate = coherent_speed / span
        diffusive_rate = 8.0 * p[P_DPERP] / (span * span)
        active_lifetime = 1.0 / max(
            1.0 / max(tau, 1.0e-300) + directed_rate + diffusive_rate,
            1.0e-300,
        )
        persistence = coherent_speed * active_lifetime
        crossing = _clip_exp(span / max(persistence, 1.0e-300))
        k_in = crossing * support_loss / max(active_lifetime, 1.0e-300)
        k_out = diffusive_rate
        mobile = k_out / max(k_in + k_out, 1.0e-300)
    else:
        span = 2.0 * p[P_ALPHA]
        active_lifetime = tau
        k_in = 0.0
        k_out = 0.0
        mobile = 1.0

    coherent_d = coherent * active_lifetime / 3.0
    fluctuating_d = fluctuating * tau / 3.0
    d_active = coherent_d + fluctuating_d
    active_excess = mobile * d_active - (1.0 - mobile) * d_bond if kc > 0.0 else 0.0
    d_mobile = d_bath + d_bond + d_active
    d_eff = d_bath + d_bond + active_excess
    cleavage_probability = -math.expm1(-min(max(omega * tau, 0.0), 700.0))

    out[0] = velocity
    out[1] = tau
    out[2] = coherent
    out[3] = fluctuating
    out[4] = d_bond
    out[5] = d_bath
    out[6] = d_active
    out[7] = d_mobile
    out[8] = d_eff
    out[9] = 6.0 * d_eff * p[P_OBSERVATION_TIME]
    out[10] = math.sqrt(max(d_eff * p[P_OBSERVATION_TIME], 0.0))
    out[11] = exposure
    out[12] = k_in
    out[13] = k_out
    out[14] = mobile
    out[15] = bt
    out[16] = zeta
    out[17] = cleavage_probability
    out[18] = k_in * p[P_OBSERVATION_TIME]
    out[19] = span
    out[20] = math.sqrt(max(rho_high * rho_low, 0.0))
    out[21] = support_loss
    out[22] = 1.0 - mobile
    return out


@njit(cache=True, parallel=True)
def evaluate_batch(
    patterns: np.ndarray,
    kd: np.ndarray,
    kc: np.ndarray,
    nb: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """Evaluate independent walker states in parallel."""

    count = kd.size
    out = np.empty((count, N_OUTPUTS), dtype=np.float64)
    for index in prange(count):
        values = evaluate_state(
            int(patterns[index]), kd[index], kc[index], nb[index], p
        )
        for field in range(N_OUTPUTS):
            out[index, field] = values[field]
    return out
