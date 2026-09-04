from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
MICRO_INPUTS = HERE / "config" / "microscopic_inputs.json"
GEOMETRIES = HERE / "config" / "ligand_geometries.json"

MU = 4.0 / (3.0 * math.pi)
GAUSSIAN_BOUND_WEIGHT = 1.0 + 2.0 ** (-1.5)


@dataclass(frozen=True)
class Inputs:
    L: float
    alpha: float
    d_rec: float
    spring_k: float
    kbt: float
    cleavage_exposure_factor: float
    gamma_parallel: float
    gamma_perp: float
    gamma_rot: float
    D_parallel: float
    D_perp: float
    D_rot0: float

    @property
    def nu_b(self) -> float:
        return math.pi**1.5 * (self.alpha / self.d_rec) ** 3

    @property
    def nu_c(self) -> float:
        return self.nu_b

    @property
    def spring_force_variance(self) -> float:
        return self.spring_k**2 * self.alpha**2 / 2.0


@dataclass(frozen=True)
class Geometry:
    pattern: str
    n_binders: int
    x_binders: np.ndarray
    x_cleavers: np.ndarray


def load_inputs() -> Inputs:
    raw = json.loads(MICRO_INPUTS.read_text(encoding="utf-8"))
    return Inputs(
        L=float(raw["rod_length"]),
        alpha=float(raw["interaction_range"]),
        d_rec=float(raw["receptor_spacing"]),
        spring_k=float(raw["spring_constant"]),
        kbt=float(raw["thermal_energy"]),
        cleavage_exposure_factor=float(raw["mean_field_cleavage_exposure_factor"]),
        gamma_parallel=float(raw["gamma_parallel"]),
        gamma_perp=float(raw["gamma_perpendicular"]),
        gamma_rot=float(raw["gamma_rotational"]),
        D_parallel=float(raw["bath_diffusion_parallel"]),
        D_perp=float(raw["bath_diffusion_perpendicular"]),
        D_rot0=float(raw["bath_diffusion_rotational"]),
    )


def load_geometries(inputs: Inputs) -> dict[tuple[str, int], Geometry]:
    raw = json.loads(GEOMETRIES.read_text(encoding="utf-8"))
    result = {}
    for pattern in ("polarized", "mixed"):
        for nb in (4, 10, 16):
            mask = np.asarray(raw[f"{pattern}_{nb}"], dtype=bool)
            x = -inputs.L / 2.0 + np.arange(mask.size, dtype=float) + 0.5
            result[(pattern, nb)] = Geometry(pattern, nb, x[mask], x[~mask])
    return result


def molecular_state(kd: float, geometry: Geometry, inputs: Inputs) -> dict[str, float]:
    kd = max(float(kd), 1e-300)
    p = inputs.nu_b / (kd + inputs.nu_b)
    tau_off = GAUSSIAN_BOUND_WEIGHT / kd
    tau_switch = GAUSSIAN_BOUND_WEIGHT / (kd + inputs.nu_b)
    per_binder_friction = inputs.spring_k * p * tau_off

    s1 = float(np.sum(geometry.x_binders))
    s2 = float(np.sum(geometry.x_binders**2))
    bt = per_binder_friction * geometry.n_binders
    br = per_binder_friction * s2
    zpar = inputs.gamma_parallel + bt

    moment = np.asarray([[geometry.n_binders, s1], [s1, s2]], dtype=float)
    bond_friction = per_binder_friction * moment
    friction = np.diag([inputs.gamma_perp, inputs.gamma_rot]) + bond_friction
    mobility = np.linalg.inv(friction)
    bond_noise = inputs.spring_force_variance / inputs.spring_k * bond_friction
    bath_noise = np.diag(
        [
            inputs.D_perp * inputs.gamma_perp**2,
            inputs.D_rot0 * inputs.gamma_rot**2,
        ]
    )
    bond_diffusion = mobility @ bond_noise @ mobility.T
    bath_diffusion = mobility @ bath_noise @ mobility.T

    force_covariance = geometry.n_binders * p * inputs.spring_force_variance
    d_parallel_bond = force_covariance * tau_off / zpar**2
    d_parallel_bath = inputs.D_parallel * inputs.gamma_parallel**2 / zpar**2
    d_perp_bond = float(bond_diffusion[0, 0])
    d_perp_bath = float(bath_diffusion[0, 0])
    d_rot_bond = float(bond_diffusion[1, 1])
    d_rot_bath = float(bath_diffusion[1, 1])
    d_bond = (d_parallel_bond + 2.0 * d_perp_bond) / 3.0
    d_bath = (d_parallel_bath + 2.0 * d_perp_bath) / 3.0
    d_rot = d_rot_bond + d_rot_bath
    tau_run = 1.0 / max(2.0 * d_rot, 1e-300)

    return {
        "attachment_probability": p,
        "mean_attachments": geometry.n_binders * p,
        "tau_off": tau_off,
        "tau_switch": tau_switch,
        "B_t": bt,
        "B_R": br,
        "B_TR": per_binder_friction * s1,
        "zeta_parallel": zpar,
        "zeta_perp": float(friction[0, 0]),
        "zeta_rot": float(friction[1, 1]),
        "binding_unbinding_diffusivity": d_bond,
        "background_diffusivity": d_bath,
        "D_parallel_bond": d_parallel_bond,
        "D_parallel_bath": d_parallel_bath,
        "D_perp_bond": d_perp_bond,
        "D_perp_bath": d_perp_bath,
        "D_R_bond": d_rot_bond,
        "D_R_bath": d_rot_bath,
        "tau": tau_run,
        "tau_run": tau_run,
    }


def cleavage_frequency(
    kd: float, kc: float, geometry: Geometry, inputs: Inputs
) -> tuple[float, float]:
    del kd
    line_integral = math.sqrt(math.pi) * inputs.alpha
    if geometry.pattern == "polarized":
        active_tracks = geometry.x_cleavers.size
    else:
        active_tracks = geometry.x_cleavers.size / (
            geometry.n_binders + geometry.x_cleavers.size
        )
    return (
        inputs.cleavage_exposure_factor * float(kc) * line_integral * active_tracks,
        1.0,
    )


def drive(
    pattern: str, kd: float, exposure: float, geometry: Geometry, inputs: Inputs
) -> dict[str, float]:
    x = max(float(exposure), 0.0)
    if pattern == "polarized":
        rho_high = 1.0
        rho_low = math.exp(-min(x, 700.0))
        gradient_length = 2.0 * inputs.alpha
    else:
        rho_high = math.exp(-min((1.0 - MU) * x, 700.0))
        rho_low = math.exp(-min((1.0 + MU) * x, 700.0))
        gradient_length = 2.0 * inputs.alpha
    delta_g = math.log(
        (float(kd) + inputs.nu_b * rho_high)
        / (float(kd) + inputs.nu_b * rho_low)
    )
    result = {
        "force": 0.0,
        "delta_g": delta_g,
        "contrast": rho_high - rho_low,
        "rho_high": rho_high,
        "rho_low": rho_low,
        "gradient_length": gradient_length,
    }
    if pattern == "polarized":
        total = geometry.n_binders + geometry.x_cleavers.size
        binder_fraction = geometry.n_binders / total
        cleaver_fraction = geometry.x_cleavers.size / total
        pair_availability = 4.0 * binder_fraction * cleaver_fraction
        result["force"] = inputs.kbt * delta_g / gradient_length * pair_availability
        result["pair_availability"] = pair_availability
    else:
        result["force"] = (
            geometry.n_binders * inputs.kbt * delta_g / (MU * gradient_length)
        )
    return result


def solve_branch(
    kd: float,
    kc: float,
    geometry: Geometry,
    inputs: Inputs,
    previous_speed: float | None = None,
) -> dict[str, float]:
    omega, free_receptor_fraction = cleavage_frequency(kd, kc, geometry, inputs)
    if kc <= 0.0 or omega <= 0.0:
        return {
            "speed": 0.0,
            "exposure": 0.0,
            "omega": omega,
            "cleaver_free_fraction": free_receptor_fraction,
            **drive(geometry.pattern, kd, 0.0, geometry, inputs),
        }
    state = molecular_state(kd, geometry, inputs)
    zeta = (
        state["zeta_parallel"]
        if geometry.pattern == "polarized"
        else state["zeta_perp"]
    )

    def residual(speed: float) -> float:
        return (
            zeta * float(speed)
            - drive(
                geometry.pattern,
                kd,
                omega / max(float(speed), 1e-300),
                geometry,
                inputs,
            )["force"]
        )

    grid = np.logspace(-14.0, 3.0, 1000)
    values = np.asarray([residual(float(speed)) for speed in grid])
    roots = []
    for low, high, f_low, f_high in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high < 0.0:
            roots.append(
                float(brentq(residual, float(low), float(high), xtol=1e-12, rtol=1e-10))
            )
    if not roots:
        return {
            "speed": 0.0,
            "exposure": 0.0,
            "omega": omega,
            "cleaver_free_fraction": free_receptor_fraction,
            **drive(geometry.pattern, kd, 0.0, geometry, inputs),
        }
    stable_roots = []
    for root in roots:
        step = max(root * 1e-5, 1e-14)
        low = max(root - step, 1e-300)
        high = root + step
        slope = (residual(high) - residual(low)) / (high - low)
        if slope > 0.0:
            stable_roots.append(root)
    candidates = stable_roots or roots
    if previous_speed is not None and previous_speed > 0.0:
        speed = min(
            candidates,
            key=lambda value: abs(math.log(max(value, 1e-300) / previous_speed)),
        )
    else:
        speed = min(candidates)
    exposure = omega / speed
    return {
        "speed": speed,
        "exposure": exposure,
        "omega": omega,
        "cleaver_free_fraction": free_receptor_fraction,
        **drive(geometry.pattern, kd, exposure, geometry, inputs),
    }


def local_depletion(
    kd: float,
    kc: float,
    geometry: Geometry,
    molecular: dict[str, float],
    inputs: Inputs,
) -> dict[str, float]:
    q_fresh = float(kd) / (float(kd) + inputs.nu_b)
    p0_fresh = q_fresh**geometry.n_binders

    if kc <= 0.0 or geometry.x_cleavers.size == 0:
        contact_coverage = 0.0
        residence_time = 0.0
        cleavage_exposure = 0.0
    else:
        cleaver_fraction = geometry.x_cleavers.size / (
            geometry.n_binders + geometry.x_cleavers.size
        )
        contact_coverage = cleaver_fraction**3
        written_width = 3.0 * (2.0 * inputs.alpha) * contact_coverage
        depleted_region_length = inputs.L + 2.0 * inputs.alpha
        passive_diffusion = (
            molecular["background_diffusivity"]
            + molecular["binding_unbinding_diffusivity"]
        )
        residence_time = (
            written_width
            * depleted_region_length
            / max(8.0 * passive_diffusion, 1e-300)
        )
        cleavage_exposure = (
            inputs.cleavage_exposure_factor * float(kc) * inputs.nu_c * residence_time
        )
    rho_at_binders = math.exp(-min(cleavage_exposure, 700.0))
    q_depleted = float(kd) / (float(kd) + inputs.nu_b * rho_at_binders)
    p0_depleted = q_depleted**geometry.n_binders
    psi = (p0_depleted - p0_fresh) / max(1.0 - p0_fresh, 1e-300)
    local_loss = float(np.clip(psi, 0.0, 1.0))
    return {
        "zero_attachment_probability_intact": p0_fresh,
        "zero_attachment_probability_depleted": p0_depleted,
        "local_support_loss_probability": local_loss,
        "cleaved_region_overlap": contact_coverage,
        "depleted_region_residence_time": residence_time,
        "mean_cleavage_exposure": cleavage_exposure,
        "remaining_receptor_fraction": rho_at_binders,
    }


def _base_state(
    pattern: str,
    kd: float,
    kc: float,
    nb: int,
    inputs: Inputs,
    geometries: dict[tuple[str, int], Geometry],
    previous_speed: float | None = None,
) -> dict[str, float]:
    geometry = geometries[(pattern, int(nb))]
    molecular = molecular_state(kd, geometry, inputs)
    branch = solve_branch(kd, kc, geometry, inputs, previous_speed)
    depletion = local_depletion(kd, kc, geometry, molecular, inputs)

    return {
        "pattern": pattern,
        "K_D": float(kd),
        "K_C": float(kc),
        "N_b": float(nb),
        "N_c": float(geometry.x_cleavers.size),
        "L": inputs.L,
        "alpha": inputs.alpha,
        "d_rec": inputs.d_rec,
        "nu_b": inputs.nu_b,
        "nu_c": inputs.nu_c,
        "background_diffusivity": molecular["background_diffusivity"],
        "binding_unbinding_diffusivity": molecular[
            "binding_unbinding_diffusivity"
        ],
        "speed": branch["speed"],
        "persistence_time": molecular["tau_run"],
        "delta_g": branch["delta_g"],
        "contrast": branch["contrast"],
        "rho_shallow": branch["rho_high"],
        "rho_deep": branch["rho_low"],
        "geometric_mean_receptor_fraction": math.sqrt(
            branch["rho_high"] * branch["rho_low"]
        ),
        "cleaver_free_fraction": branch["cleaver_free_fraction"],
        **molecular,
        **depletion,
    }


def predict(
    pattern: str,
    kd: float,
    kc: float,
    nb: int,
    inputs: Inputs,
    geometries: dict[tuple[str, int], Geometry],
    previous_speed: float | None = None,
) -> dict[str, float]:
    """Map molecular inputs to speed, persistence and long-time diffusivity."""
    result = _base_state(
        pattern, kd, kc, nb, inputs, geometries, previous_speed
    )
    binder_fraction = result["N_b"] / (result["N_b"] + result["N_c"])
    cleaver_fraction = 1.0 - binder_fraction
    interface_coverage = (2.0 * min(binder_fraction, cleaver_fraction)) ** 3
    if result["pattern"] == "mixed":
        speed_gate = 1.0 - (1.0 - result["attachment_probability"]) ** result["N_b"]
        fluctuation_factor = 1.0 + (
            4.0
            * result["attachment_probability"]
            * (1.0 - result["attachment_probability"])
            * interface_coverage
        )
    else:
        speed_gate = 1.0
        response_time = inputs.gamma_parallel / inputs.spring_k
        renewal_probability = -math.expm1(
            -response_time / max(result["tau_off"], 1e-300)
        )
        fluctuation_factor = 1.0 + (
            4.0
            * result["attachment_probability"]
            * (1.0 - result["attachment_probability"])
            * interface_coverage
            * renewal_probability
        )
    coherent_speed_second_moment = result["speed"] ** 2 * speed_gate
    fluctuating_speed_second_moment = coherent_speed_second_moment * (
        fluctuation_factor - 1.0
    )
    speed_second_moment = coherent_speed_second_moment + fluctuating_speed_second_moment

    if result["pattern"] == "mixed" and result["K_C"] > 0.0:
        span = result["L"] + 2.0 * result["alpha"]
        coherent_rms_speed = math.sqrt(max(coherent_speed_second_moment, 0.0))
        directed_footprint_rate = coherent_rms_speed / span
        diffusive_footprint_rate = 8.0 * inputs.D_perp / span**2
        directional_lifetime = 1.0 / (
            1.0 / result["persistence_time"]
            + directed_footprint_rate
            + diffusive_footprint_rate
        )
        persistence_length = coherent_rms_speed * directional_lifetime
        crossing = math.exp(-min(span / max(persistence_length, 1e-300), 700.0))
        k_in = (
            crossing
            * result["local_support_loss_probability"]
            / directional_lifetime
        )
        k_out = diffusive_footprint_rate
        mobile = k_out / max(k_in + k_out, 1e-300)
        peclet = persistence_length / span
    else:
        span = 0.0
        coherent_rms_speed = math.sqrt(max(coherent_speed_second_moment, 0.0))
        directed_footprint_rate = 0.0
        diffusive_footprint_rate = 0.0
        directional_lifetime = result["persistence_time"]
        peclet = 0.0
        crossing = 0.0
        k_in = 0.0
        k_out = 0.0
        mobile = 1.0

    coherent_active_diffusion = (
        coherent_speed_second_moment * directional_lifetime / 3.0
    )
    fluctuating_active_diffusion = (
        fluctuating_speed_second_moment * result["persistence_time"] / 3.0
    )
    d_active = coherent_active_diffusion + fluctuating_active_diffusion
    active_excess = (
        mobile * d_active
        - (1.0 - mobile) * result["binding_unbinding_diffusivity"]
        if result["K_C"] > 0.0
        else 0.0
    )

    result.update(
        {
            "coherent_speed_second_moment": coherent_speed_second_moment,
            "fluctuating_speed_second_moment": fluctuating_speed_second_moment,
            "active_speed_second_moment": speed_second_moment,
            "finite_contact_factor": fluctuation_factor,
            "active_contact_gate": speed_gate,
            "active_diffusivity": d_active,
            "coherent_active_diffusivity": coherent_active_diffusion,
            "finite_ligand_active_diffusivity": fluctuating_active_diffusion,
            "directional_lifetime": directional_lifetime,
            "directed_footprint_rate": directed_footprint_rate,
            "diffusive_footprint_rate": diffusive_footprint_rate,
            "mobile_fraction": mobile,
            "locally_depleted_fraction": 1.0 - mobile,
            "local_depletion_entry_rate": k_in,
            "local_depletion_exit_rate": k_out,
            "rms_run_speed": coherent_rms_speed,
            "cleaver_span": span,
            "encounter_peclet": peclet,
            "crossing_probability": crossing,
            "persistence_length": coherent_rms_speed * directional_lifetime,
            "diffusivity_shift": active_excess,
            "effective_diffusivity": result["background_diffusivity"]
            + result["binding_unbinding_diffusivity"]
            + active_excess,
            "directional_persistence_time": result["persistence_time"] / 3.0
            if result["pattern"] == "mixed"
            else result["persistence_time"],
        }
    )
    return result
