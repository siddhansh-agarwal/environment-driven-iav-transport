import numpy as np
from numba import njit
from typing import Tuple

FREE = 0
BOUND = 1
CLEAVED = 2

EVENT_NONE = np.int8(0)
EVENT_BIND = np.int8(1)
EVENT_UNBIND = np.int8(2)
EVENT_CLEAVE = np.int8(3)


@njit(cache=True)
def seed_rng(seed: int) -> None:
    """Seed the random stream used inside the compiled reaction kernels."""
    np.random.seed(seed)


@njit(cache=True)
def _propensity_terms_3d(
    dx: float,
    dy: float,
    dz: float,
    alpha_sq: float,
    K_D: float,
    K_C: float,
) -> Tuple[float, float, float]:
    dr2 = dx * dx + dy * dy + dz * dz
    exp_term = np.exp(-dr2 / alpha_sq)
    denom = 1.0 + exp_term
    q_on = exp_term / denom
    q_off = K_D / denom
    q_c = K_C * exp_term
    return q_on, q_off, q_c


@njit(cache=True, inline="always")
def _q_on_from_dr2_inline(
    dr2: float,
    alpha_sq: float,
) -> float:
    exp_term = np.exp(-dr2 / alpha_sq)
    return exp_term / (1.0 + exp_term)


@njit(cache=True, inline="always")
def _q_c_from_dr2_inline(
    dr2: float,
    alpha_sq: float,
    K_C: float,
) -> float:
    return K_C * np.exp(-dr2 / alpha_sq)


@njit(cache=True)
def build_reaction_rate_tables(
    n_free: int,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    n_bound: int,
    bound_ligand_idx: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
    r_i: np.ndarray,
    alpha: float,
    K_D: float,
    K_C: float,
    ligand_types: np.ndarray,
    ligand_bound: np.ndarray,
    bind_sum_by_receptor: np.ndarray,
    cleave_sum_by_receptor: np.ndarray,
    unbind_rate_by_bound_slot: np.ndarray,
):
    """
    Build the reaction-rate table for the current particle geometry.

    Free channels are aggregated per receptor. Bound channels are aggregated per
    active bound ligand slot.
    """
    alpha_sq = alpha * alpha
    total_propensity = 0.0
    bind_top_channels = 0
    unbind_top_channels = 0
    cleave_top_channels = 0
    window_len_total = 0

    for slot in range(n_free):
        bind_sum_by_receptor[slot] = 0.0
        cleave_sum_by_receptor[slot] = 0.0
    for slot in range(n_bound):
        unbind_rate_by_bound_slot[slot] = 0.0

    for slot in range(n_free):
        s_val = free_s[slot]
        rho_sq = free_rho_sq[slot]
        i_min = int(free_i_min[slot])
        i_max = int(free_i_max[slot])
        if i_max < i_min:
            continue

        window_len_total += i_max - i_min + 1
        bind_sum = 0.0
        cleave_sum = 0.0
        for lig_idx in range(i_min, i_max + 1):
            dr2 = rho_sq + (s_val - r_i[lig_idx]) * (s_val - r_i[lig_idx])
            if bool(ligand_types[lig_idx]):
                if bool(ligand_bound[lig_idx]):
                    continue
                bind_sum += _q_on_from_dr2_inline(dr2, alpha_sq)
            else:
                cleave_sum += _q_c_from_dr2_inline(dr2, alpha_sq, K_C)
        bind_sum_by_receptor[slot] = bind_sum
        cleave_sum_by_receptor[slot] = cleave_sum
        total_propensity += bind_sum + cleave_sum
        if bind_sum > 0.0:
            bind_top_channels += 1
        if cleave_sum > 0.0:
            cleave_top_channels += 1

    for slot in range(n_bound):
        _, q_off, _ = _propensity_terms_3d(
            bound_dx[slot],
            bound_dy[slot],
            bound_dz[slot],
            alpha_sq,
            K_D,
            K_C,
        )
        unbind_rate_by_bound_slot[slot] = q_off
        total_propensity += q_off
        if q_off > 0.0:
            unbind_top_channels += 1

    return (
        total_propensity,
        bind_top_channels,
        unbind_top_channels,
        cleave_top_channels,
        window_len_total,
    )


@njit(cache=True)
def sample_reaction_event(
    n_free: int,
    bind_sum_by_receptor: np.ndarray,
    cleave_sum_by_receptor: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    n_bound: int,
    unbind_rate_by_bound_slot: np.ndarray,
    bound_ligand_idx: np.ndarray,
    r_i: np.ndarray,
    alpha: float,
    K_D: float,
    K_C: float,
    ligand_types: np.ndarray,
    ligand_bound: np.ndarray,
    total_propensity: float,
):
    """
    Exact hierarchical sampler for receptor-centric geometry aggregates.

    Stage 1 samples a free-receptor bind channel, a bound-slot unbind channel,
    or a free-receptor cleave channel. Stage 2 samples the ligand only when a
    bind channel was selected.
    """
    if total_propensity <= 0.0:
        return False, 0.1, EVENT_NONE, -1, -1

    alpha_sq = alpha * alpha
    u1 = np.random.random()
    if u1 <= 0.0:
        u1 = 1e-300
    tau = -np.log(u1) / total_propensity
    target = np.random.random() * total_propensity
    cumsum = 0.0
    for slot in range(n_free):
        bind_sum = bind_sum_by_receptor[slot]
        if bind_sum <= 0.0:
            continue
        next_cumsum = cumsum + bind_sum
        if target < next_cumsum:
            row_target = target - cumsum
            row_cumsum = 0.0
            s_val = free_s[slot]
            rho_sq = free_rho_sq[slot]
            i_min = int(free_i_min[slot])
            i_max = int(free_i_max[slot])
            for lig_idx in range(i_min, i_max + 1):
                if not bool(ligand_types[lig_idx]) or bool(ligand_bound[lig_idx]):
                    continue
                dr2 = rho_sq + (s_val - r_i[lig_idx]) * (s_val - r_i[lig_idx])
                q_on = _q_on_from_dr2_inline(dr2, alpha_sq)
                row_cumsum += q_on
                if row_target < row_cumsum:
                    return True, tau, EVENT_BIND, slot, lig_idx
            return False, tau, EVENT_NONE, -1, -1
        cumsum = next_cumsum

    for slot in range(n_bound):
        rate = unbind_rate_by_bound_slot[slot]
        if rate <= 0.0:
            continue
        cumsum += rate
        if target < cumsum:
            return (
                True,
                tau,
                EVENT_UNBIND,
                slot,
                int(bound_ligand_idx[slot]),
            )

    for slot in range(n_free):
        rate = cleave_sum_by_receptor[slot]
        if rate <= 0.0:
            continue
        cumsum += rate
        if target < cumsum:
            return True, tau, EVENT_CLEAVE, slot, -1

    return False, tau, EVENT_NONE, -1, -1


@njit(cache=True)
def sample_reaction_wait_time_rng(total_propensity: float):
    """Draw only the exponential reaction clock for a known total propensity."""
    if total_propensity <= 0.0:
        return False, 1.0e300
    u1 = np.random.random()
    if u1 <= 0.0:
        u1 = 1e-300
    tau = -np.log(u1) / total_propensity
    return True, tau


@njit(cache=True)
def select_reaction_channel(
    n_free: int,
    bind_sum_by_receptor: np.ndarray,
    cleave_sum_by_receptor: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    n_bound: int,
    unbind_rate_by_bound_slot: np.ndarray,
    bound_ligand_idx: np.ndarray,
    r_i: np.ndarray,
    alpha: float,
    K_D: float,
    K_C: float,
    ligand_types: np.ndarray,
    ligand_bound: np.ndarray,
    total_propensity: float,
    target: float,
):
    """
    Select a reaction channel from a supplied cumulative-rate target.

    The caller supplies `target` in `[0, total_propensity)`, corresponding to
    a single uniform draw scaled by the instantaneous total propensity.
    """
    if total_propensity <= 0.0:
        return False, EVENT_NONE, -1, -1

    if target < 0.0:
        target = 0.0
    elif target >= total_propensity:
        # Guard floating point edge cases from u == 1.
        target = np.nextafter(total_propensity, 0.0)

    alpha_sq = alpha * alpha
    cumsum = 0.0
    for slot in range(n_free):
        bind_sum = bind_sum_by_receptor[slot]
        if bind_sum <= 0.0:
            continue
        next_cumsum = cumsum + bind_sum
        if target < next_cumsum:
            row_target = target - cumsum
            row_cumsum = 0.0
            s_val = free_s[slot]
            rho_sq = free_rho_sq[slot]
            i_min = int(free_i_min[slot])
            i_max = int(free_i_max[slot])
            for lig_idx in range(i_min, i_max + 1):
                if not bool(ligand_types[lig_idx]) or bool(ligand_bound[lig_idx]):
                    continue
                dr2 = rho_sq + (s_val - r_i[lig_idx]) * (s_val - r_i[lig_idx])
                q_on = _q_on_from_dr2_inline(dr2, alpha_sq)
                row_cumsum += q_on
                if row_target < row_cumsum:
                    return True, EVENT_BIND, slot, lig_idx
            return False, EVENT_NONE, -1, -1
        cumsum = next_cumsum

    for slot in range(n_bound):
        rate = unbind_rate_by_bound_slot[slot]
        if rate <= 0.0:
            continue
        cumsum += rate
        if target < cumsum:
            return True, EVENT_UNBIND, slot, int(bound_ligand_idx[slot])

    for slot in range(n_free):
        rate = cleave_sum_by_receptor[slot]
        if rate <= 0.0:
            continue
        cumsum += rate
        if target < cumsum:
            return True, EVENT_CLEAVE, slot, -1

    return False, EVENT_NONE, -1, -1


@njit(cache=True)
def sample_reaction_channel(
    n_free: int,
    bind_sum_by_receptor: np.ndarray,
    cleave_sum_by_receptor: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    n_bound: int,
    unbind_rate_by_bound_slot: np.ndarray,
    bound_ligand_idx: np.ndarray,
    r_i: np.ndarray,
    alpha: float,
    K_D: float,
    K_C: float,
    ligand_types: np.ndarray,
    ligand_bound: np.ndarray,
    total_propensity: float,
):
    """
    Draw one reaction channel from the instantaneous rate table.
    """
    if total_propensity <= 0.0:
        return False, EVENT_NONE, -1, -1

    u = np.random.random()
    if u >= 1.0:
        u = np.nextafter(1.0, 0.0)
    target = u * total_propensity
    ok, event_kind, event_slot_idx, event_ligand_idx = (
        select_reaction_channel(
            n_free,
            bind_sum_by_receptor,
            cleave_sum_by_receptor,
            free_s,
            free_rho_sq,
            free_i_min,
            free_i_max,
            n_bound,
            unbind_rate_by_bound_slot,
            bound_ligand_idx,
            r_i,
            alpha,
            K_D,
            K_C,
            ligand_types,
            ligand_bound,
            total_propensity,
            target,
        )
    )
    return ok, event_kind, event_slot_idx, event_ligand_idx
