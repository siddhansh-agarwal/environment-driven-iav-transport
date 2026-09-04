"""Mean-field transport coefficients for a particle on a receptor surface.

These equations are the two-dimensional counterparts of Supplementary
Information Eqs. S1--S9. Surface simulations contain no added thermal
displacements, so their random motion comes from attachment and detachment.
The calculation uses only the molecular inputs and ligand geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config"
CHI_B = 1.0 + 2.0 ** (-1.5)
MU = 4.0 / (3.0 * math.pi)


@dataclass(frozen=True)
class SurfaceInputs:
    length: float
    alpha: float
    receptor_spacing: float
    spring_k: float
    kbt: float
    cleavage_exposure_factor: float
    gamma_parallel: float
    gamma_perp: float
    gamma_rot: float

    @property
    def nu_kernel(self) -> float:
        """Continuum sum of the shared Gaussian receptor kernel on a plane."""
        return math.pi * (self.alpha / self.receptor_spacing) ** 2

    @property
    def nu_b(self) -> float:
        """Weighted receptor count sampled by one binder."""
        return self.nu_kernel

    @property
    def nu_c(self) -> float:
        """Weighted receptor count sampled by one cleaver."""
        return self.nu_kernel

    @property
    def spring_force_variance(self) -> float:
        return self.spring_k**2 * self.alpha**2 / 2.0


@dataclass(frozen=True)
class Geometry:
    pattern: str
    n_binders: int
    x_binders: np.ndarray
    x_cleavers: np.ndarray


def load_inputs() -> SurfaceInputs:
    raw = json.loads((CONFIG / "microscopic_inputs.json").read_text())
    return SurfaceInputs(
        length=float(raw["rod_length"]),
        alpha=float(raw["interaction_range"]),
        receptor_spacing=float(raw["receptor_spacing"]),
        spring_k=float(raw["spring_constant"]),
        kbt=float(raw["thermal_energy"]),
        cleavage_exposure_factor=float(raw["mean_field_cleavage_exposure_factor"]),
        gamma_parallel=float(raw["gamma_parallel"]),
        gamma_perp=float(raw["gamma_perpendicular"]),
        gamma_rot=float(raw["gamma_rotational"]),
    )


def load_geometries(inputs: SurfaceInputs) -> dict[tuple[str, int], Geometry]:
    raw = json.loads((CONFIG / "ligand_geometries.json").read_text())
    geometries: dict[tuple[str, int], Geometry] = {}
    for pattern in ("polarized", "mixed"):
        for n_binders in (4, 10, 16):
            mask = np.asarray(raw[f"{pattern}_{n_binders}"], dtype=bool)
            x = -inputs.length / 2.0 + np.arange(mask.size, dtype=float) + 0.5
            geometries[(pattern, n_binders)] = Geometry(
                pattern=pattern,
                n_binders=n_binders,
                x_binders=x[mask],
                x_cleavers=x[~mask],
            )
    return geometries


def molecular_state(
    kd: float, geometry: Geometry, inputs: SurfaceInputs
) -> dict[str, float]:
    """Mean attachments, friction, force fluctuations and planar persistence."""
    kd = max(float(kd), 1.0e-300)
    p = inputs.nu_b / (kd + inputs.nu_b)
    tau_bond = CHI_B / kd
    bond_friction = inputs.spring_k * geometry.n_binders * p * tau_bond
    lever_arm_sum = float(np.sum(geometry.x_binders**2))
    rotational_bond_friction = inputs.spring_k * p * tau_bond * lever_arm_sum
    zeta_parallel = inputs.gamma_parallel + bond_friction
    zeta_perp = inputs.gamma_perp + bond_friction
    zeta_rot = inputs.gamma_rot + rotational_bond_friction

    force_covariance = geometry.n_binders * p * inputs.spring_force_variance
    d_bond_2d = 0.5 * force_covariance * tau_bond * (zeta_parallel**-2 + zeta_perp**-2)
    torque_covariance = p * inputs.spring_force_variance * lever_arm_sum
    d_rot = torque_covariance * tau_bond / zeta_rot**2

    # Planar angular diffusion has first harmonic exp(-D_R t).
    tau_direction = 1.0 / max(d_rot, 1.0e-300)
    return {
        "attachment_probability": p,
        "mean_attachments": geometry.n_binders * p,
        "tau_bond": tau_bond,
        "B_t": bond_friction,
        "B_R": rotational_bond_friction,
        "zeta_parallel": zeta_parallel,
        "zeta_perp": zeta_perp,
        "zeta_rot": zeta_rot,
        "D_bond_2d": d_bond_2d,
        "D_rot": d_rot,
        "tau_direction": tau_direction,
        "lever_arm_sum": lever_arm_sum,
    }


def cleavage_frequency(kc: float, geometry: Geometry, inputs: SurfaceInputs) -> float:
    if geometry.pattern == "polarized":
        swept_length = 2.0 * inputs.alpha * geometry.x_cleavers.size
    else:
        cleaver_fraction = geometry.x_cleavers.size / (
            geometry.n_binders + geometry.x_cleavers.size
        )
        swept_length = inputs.alpha * cleaver_fraction
    return inputs.cleavage_exposure_factor * float(kc) * inputs.nu_c * swept_length


def drive(
    kd: float, exposure: float, geometry: Geometry, inputs: SurfaceInputs
) -> dict[str, float]:
    exposure = max(float(exposure), 0.0)
    if geometry.pattern == "polarized":
        rho_high = 1.0
        rho_low = math.exp(-min(exposure, 700.0))
        delta_g = math.log(
            (float(kd) + inputs.nu_b * rho_high) / (float(kd) + inputs.nu_b * rho_low)
        )
        force = inputs.kbt * delta_g / (4.0 * inputs.alpha)
    else:
        rho_high = math.exp(-min((1.0 - MU) * exposure, 700.0))
        rho_low = math.exp(-min((1.0 + MU) * exposure, 700.0))
        delta_g = math.log(
            (float(kd) + inputs.nu_b * rho_high) / (float(kd) + inputs.nu_b * rho_low)
        )
        force = geometry.n_binders * inputs.kbt * delta_g / (6.0 * inputs.alpha)
    return {
        "force": force,
        "delta_g": delta_g,
        "contrast": rho_high - rho_low,
        "rho_high": rho_high,
        "rho_low": rho_low,
    }


def solve_speed(
    kd: float,
    kc: float,
    geometry: Geometry,
    inputs: SurfaceInputs,
    previous_speed: float | None = None,
) -> dict[str, float]:
    """Solve the SI force-exposure balance and follow its stable branch."""
    omega = cleavage_frequency(kc, geometry, inputs)
    if kc <= 0.0 or omega <= 0.0:
        return {
            "speed": 0.0,
            "exposure": 0.0,
            "omega": omega,
            **drive(kd, 0.0, geometry, inputs),
        }

    state = molecular_state(kd, geometry, inputs)
    zeta = (
        state["zeta_parallel"]
        if geometry.pattern == "polarized"
        else state["zeta_perp"]
    )

    def residual(speed: float) -> float:
        return (
            zeta * speed
            - drive(kd, omega / max(speed, 1.0e-300), geometry, inputs)["force"]
        )

    grid = np.logspace(-14.0, 2.0, 1200)
    values = np.asarray([residual(float(value)) for value in grid])
    roots: list[float] = []
    for low, high, f_low, f_high in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high < 0.0:
            roots.append(
                float(
                    brentq(
                        residual, float(low), float(high), xtol=1.0e-13, rtol=1.0e-11
                    )
                )
            )
    stable: list[float] = []
    for root in roots:
        step = max(root * 1.0e-5, 1.0e-14)
        low = max(root - step, 1.0e-300)
        high = root + step
        if (residual(high) - residual(low)) / (high - low) > 0.0:
            stable.append(root)
    candidates = stable or roots
    if not candidates:
        return {
            "speed": 0.0,
            "exposure": 0.0,
            "omega": omega,
            "n_roots": 0.0,
            **drive(kd, 0.0, geometry, inputs),
        }
    if previous_speed is not None and previous_speed > 0.0:
        speed = min(candidates, key=lambda value: abs(math.log(value / previous_speed)))
    else:
        speed = min(candidates)
    exposure = omega / speed
    return {
        "speed": speed,
        "exposure": exposure,
        "omega": omega,
        "n_roots": float(len(roots)),
        **drive(kd, exposure, geometry, inputs),
    }


def surface_transport(
    pattern: str,
    kd: float,
    kc: float,
    n_binders: int,
    inputs: SurfaceInputs,
    geometries: dict[tuple[str, int], Geometry],
    previous_speed: float | None = None,
) -> dict[str, float | str]:
    """Return every transport ingredient used by the recurrence calculation."""
    geometry = geometries[(pattern, int(n_binders))]
    state = molecular_state(kd, geometry, inputs)
    branch = solve_speed(kd, kc, geometry, inputs, previous_speed)
    phi_b = geometry.n_binders / (geometry.n_binders + geometry.x_cleavers.size)
    phi_c = 1.0 - phi_b
    coherent = branch["speed"] ** 2

    tau_direction = state["tau_direction"]
    sign_renewal_rate = 0.0
    trail_probability = 0.0
    if pattern == "polarized":
        # Two independent interaction-scale samples replace the three samples
        # in the three-dimensional uniform theory.
        interface = (2.0 * min(phi_b, phi_c)) ** 2
        speed_second_moment = coherent * (
            1.0
            + 4.0
            * state["attachment_probability"]
            * (1.0 - state["attachment_probability"])
            * interface
        )
        attachment_support_probability = 1.0
    else:
        attachment_support_probability = (
            1.0 - (1.0 - state["attachment_probability"]) ** geometry.n_binders
        )
        # Mixed motion has two transverse signs.  Translating through a
        # Gaussian receptor field renews the selected sign even when the body
        # axis remains fixed.  The kernel autocorrelation is
        # exp[-Delta^2/(2 alpha^2)], whose integral length is
        # sqrt(pi/2) alpha.  Solve its rate together with the finite-ligand
        # speed variance because each depends on the same directional clock.
        for _ in range(100):
            trail_probability = -math.expm1(
                -min(branch["omega"] * tau_direction, 700.0)
            )
            finite_variance = (
                4.0
                * state["attachment_probability"]
                * (1.0 - state["attachment_probability"])
                * trail_probability
                * (4.0 * phi_b * phi_c)
                / geometry.n_binders
            )
            speed_second_moment = (
                attachment_support_probability * coherent * (1.0 + finite_variance)
            )
            sign_renewal_rate = (
                math.sqrt(2.0 / math.pi)
                * math.sqrt(max(speed_second_moment, 0.0))
                / inputs.alpha
            )
            updated_tau = 1.0 / max(state["D_rot"] + sign_renewal_rate, 1.0e-300)
            if abs(updated_tau - tau_direction) <= 1.0e-10 * max(updated_tau, 1.0):
                tau_direction = updated_tau
                break
            tau_direction = 0.5 * (tau_direction + updated_tau)
        interface = 4.0 * phi_b * phi_c

    d_active_2d = speed_second_moment * tau_direction / 2.0
    d_mobile_2d = state["D_bond_2d"] + d_active_2d
    step_length = math.sqrt(max(2.0 * d_mobile_2d * tau_direction, 0.0))
    write_speed = step_length / tau_direction
    return {
        "pattern": pattern,
        "K_D": float(kd),
        "K_C": float(kc),
        "N_b": float(n_binders),
        "N_c": float(geometry.x_cleavers.size),
        "nu_2d": inputs.nu_kernel,
        "nu_b_2d": inputs.nu_kernel,
        "nu_c_2d": inputs.nu_kernel,
        "v_coherent": branch["speed"],
        "v_rms": math.sqrt(max(speed_second_moment, 0.0)),
        "v_write_one_clock": write_speed,
        "D_active_2d": d_active_2d,
        "D_bond_2d": state["D_bond_2d"],
        "D_mobile_2d": d_mobile_2d,
        "tau_direction_2d": tau_direction,
        "tau_rotational_2d": state["tau_direction"],
        "sign_renewal_rate": sign_renewal_rate,
        "ell_one_clock": step_length,
        "interface_sampling": interface,
        "attachment_support_probability": attachment_support_probability,
        **state,
        **branch,
    }
