"""Stochastic simulation kernels for particle motion and receptor editing."""

import numpy as np
from typing import Dict, Any, Tuple
from numba import njit

from .core.reactions import (
    seed_rng,
    build_reaction_rate_tables,
    sample_reaction_event,
    sample_reaction_wait_time_rng,
    sample_reaction_channel,
    EVENT_NONE,
    EVENT_BIND,
    EVENT_UNBIND,
    EVENT_CLEAVE,
)
from .core.dynamics3d import (
    update_position_3d_bound_vectors,
    update_position_3d_pending_frozen_with_flag,
)
from .core.thermal import (
    brownian_kick_free_rod_3d,
    brownian_dynamics_step_bound_rod_relaxation_ou_3d,
    reversible_thermal_energy_from_alpha,
    thermal_diffusion_constants,
    validate_thermal_inputs,
)
from .core.grid_backend3d import (
    normalize_grid_backend,
    GRID_BACKEND_GRADIENT_SPARSE_COORDS,
    GRID_BACKEND_UNIFORM_SPARSE_COORDS,
)
from .core.gradient_axis import (
    DEFAULT_GRADIENT_MAX_SPACING,
    DEFAULT_GRADIENT_MIN_SPACING,
    GRADIENT_AXIS_LAW_MULTIPLICATIVE,
    build_gradient_axis_spec,
    dense_coordinate_from_axis_position,
    dense_interval_from_axis_interval,
    axis_position_from_index_implicit,
    axis_first_dense_coordinate_with_spacing_at_most,
    axis_first_dense_coordinate_with_spacing_at_least_on_sparse_side,
    axis_lower_bound_implicit,
    axis_upper_bound_implicit,
    axis_min_spacing_in_interval_implicit,
)

FREE = 0
BOUND = 1
CLEAVED = 2
DIMENSION_2D = "2d"
DIMENSION_3D = "3d"
INITIAL_NEARBY_CAPACITY = 2_000
NEARBY_PADDING = 500
CLEAVED_BLOCK_EDGE = 4

TERM_CHUNK_LIMIT = 0
TERM_T_FINAL = 1
TERM_GRADIENT_THRESHOLD = 2
TERM_NO_NEARBY = 3
TERM_RESOURCE_LIMIT = 5
TERM_GRADIENT_ESCAPE = 6

NO_NEARBY_POLICY_VALIDATED_TERMINAL = "validated_terminal"
NO_NEARBY_POLICY_VALIDATED_TERMINAL_CODE = 0

DEFAULT_NEARBY_CUTOFF_ALPHA_MULT = 3.0
DEFAULT_NEARBY_CUTOFF_VALIDATE_ALPHA_MULT = 5.0
DEFAULT_TAIL_PROPENSITY_EPS = 1e-3
DEFAULT_THERMAL_BROWNIAN_DT = 1.0e-2
DEFAULT_THERMAL_RATE_ENERGY_RTOL = 1.0e-10
DEFAULT_GRADIENT_MIN_SPACING_STOP = 0.0
GRADIENT_ESCAPE_STOP_MODE_OFF = "off"
GRADIENT_ESCAPE_STOP_MODE_AXIS_PLANE_DWELL = "axis_plane_dwell"
SUPPORTED_GRADIENT_ESCAPE_STOP_MODES = (
    GRADIENT_ESCAPE_STOP_MODE_OFF,
    GRADIENT_ESCAPE_STOP_MODE_AXIS_PLANE_DWELL,
)
DEFAULT_GRADIENT_ESCAPE_DWELL_TIME = 1000.0
DEFAULT_TRAJECTORY_RECORD_TARGET_POINTS = 512
DEFAULT_TRAJECTORY_RECORD_MIN_INTERVAL = 1.0e-4

MOTION_RULE_ATHERMAL = "athermal_event_driven"
MOTION_RULE_BROWNIAN = "adaptive_brownian_reaction"
RECEPTOR_MOBILITY_FIXED = "fixed"


def normalize_dimension(value: Any) -> str:
    mode = str(value if value is not None else DIMENSION_3D).strip().lower()
    if mode in ("", "default", "none"):
        mode = DIMENSION_3D
    if mode not in (DIMENSION_2D, DIMENSION_3D):
        raise ValueError("Invalid DIMENSION. Use one of: 2d, 3d")
    return mode


def validate_dimension_config(config: Dict[str, Any]) -> str:
    dimension = normalize_dimension(config.get("DIMENSION", DIMENSION_3D))
    gradient_type = str(config.get("GRADIENT_TYPE", "uniform")).strip().lower()
    if dimension == DIMENSION_2D and gradient_type == "z":
        raise ValueError("DIMENSION=2d only supports GRADIENT_TYPE in {uniform, x, y}")
    return dimension


def clamp_planar_pose(
    x: float, y: float, z: float, n_hat: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    vec = np.asarray(n_hat, dtype=np.float64).copy()
    vec[2] = 0.0
    xy_norm = float(np.hypot(vec[0], vec[1]))
    if xy_norm <= 1.0e-12:
        vec[0] = 1.0
        vec[1] = 0.0
    else:
        vec[0] /= xy_norm
        vec[1] /= xy_norm
    return float(x), float(y), 0.0, vec


def clamp_planar_sparse_coord_state(
    bound_iz: np.ndarray,
    cleaved_iz: np.ndarray,
    hash_iz: np.ndarray,
    pending_force_z: np.ndarray,
    pending_bound_rz: np.ndarray,
) -> None:
    if bound_iz.size > 0:
        bound_iz[:] = 0
    if cleaved_iz.size > 0:
        cleaved_iz[:] = 0
    if hash_iz.size > 0:
        hash_iz[:] = 0
    if pending_force_z.size > 0:
        pending_force_z[:] = 0.0
    if pending_bound_rz.size > 0:
        pending_bound_rz[:] = 0.0


def build_position_row_for_dimension(
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    dimension: str,
) -> np.ndarray:
    if normalize_dimension(dimension) == DIMENSION_2D:
        theta = float(np.arctan2(float(n_hat[1]), float(n_hat[0])))
        return np.array([x, y, theta], dtype=np.float64)
    return np.array([x, y, z, n_hat[0], n_hat[1], n_hat[2]], dtype=np.float64)


def normalize_no_nearby_policy(value: str) -> str:
    mode = str(value).strip().lower()
    if mode != NO_NEARBY_POLICY_VALIDATED_TERMINAL:
        raise ValueError("no_nearby_policy must be 'validated_terminal'")
    return mode


def no_nearby_policy_to_code(value: str) -> int:
    normalize_no_nearby_policy(value)
    return NO_NEARBY_POLICY_VALIDATED_TERMINAL_CODE


def normalize_gradient_escape_stop_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in SUPPORTED_GRADIENT_ESCAPE_STOP_MODES:
        raise ValueError(
            "Invalid gradient escape stop mode. Use one of: "
            + ", ".join(SUPPORTED_GRADIENT_ESCAPE_STOP_MODES)
        )
    return mode


@njit(cache=True)
def _coord_hash_slot(ix: int, iy: int, iz: int, mask: int) -> int:
    # 3D integer hash; exactness comes from tuple equality checks, not hash uniqueness.
    h = (
        np.uint64(np.int64(ix) * np.int64(73856093))
        ^ np.uint64(np.int64(iy) * np.int64(19349663))
        ^ np.uint64(np.int64(iz) * np.int64(83492791))
    )
    return int(np.int64(h & np.uint64(mask)))


@njit(cache=True)
def _coord_hash_contains(
    ix: int,
    iy: int,
    iz: int,
    h_ix: np.ndarray,
    h_iy: np.ndarray,
    h_iz: np.ndarray,
    h_used: np.ndarray,
) -> bool:
    cap = len(h_used)
    if cap <= 0:
        return False
    mask = cap - 1
    slot = _coord_hash_slot(ix, iy, iz, mask)
    for _ in range(cap):
        if h_used[slot] == 0:
            return False
        if (
            int(h_ix[slot]) == int(ix)
            and int(h_iy[slot]) == int(iy)
            and int(h_iz[slot]) == int(iz)
        ):
            return True
        slot = (slot + 1) & mask
    return False


@njit(cache=True)
def _coord_hash_insert_no_grow(
    ix: int,
    iy: int,
    iz: int,
    h_ix: np.ndarray,
    h_iy: np.ndarray,
    h_iz: np.ndarray,
    h_used: np.ndarray,
) -> bool:
    cap = len(h_used)
    if cap <= 0:
        return False
    mask = cap - 1
    slot = _coord_hash_slot(ix, iy, iz, mask)
    for _ in range(cap):
        if h_used[slot] == 0:
            h_used[slot] = np.uint8(1)
            h_ix[slot] = np.int64(ix)
            h_iy[slot] = np.int64(iy)
            h_iz[slot] = np.int64(iz)
            return True
        if (
            int(h_ix[slot]) == int(ix)
            and int(h_iy[slot]) == int(iy)
            and int(h_iz[slot]) == int(iz)
        ):
            return False
        slot = (slot + 1) & mask
    return False


@njit(cache=True)
def _cleaved_block_key_and_bit(
    ix: int, iy: int, iz: int
) -> Tuple[int, int, int, np.uint64]:
    edge = CLEAVED_BLOCK_EDGE
    bx = int(ix) // edge
    by = int(iy) // edge
    bz = int(iz) // edge
    lx = int(ix) - bx * edge
    ly = int(iy) - by * edge
    lz = int(iz) - bz * edge
    bit_idx = lx + edge * ly + edge * edge * lz
    return bx, by, bz, np.uint64(1) << np.uint64(bit_idx)


@njit(cache=True)
def _cleaved_block_hash_contains_coord(
    ix: int,
    iy: int,
    iz: int,
    block_ix: np.ndarray,
    block_iy: np.ndarray,
    block_iz: np.ndarray,
    block_bits: np.ndarray,
    block_used: np.ndarray,
) -> bool:
    cap = len(block_used)
    if cap <= 0:
        return False
    bx, by, bz, bit = _cleaved_block_key_and_bit(ix, iy, iz)
    mask = cap - 1
    slot = _coord_hash_slot(bx, by, bz, mask)
    for _ in range(cap):
        if block_used[slot] == 0:
            return False
        if (
            int(block_ix[slot]) == int(bx)
            and int(block_iy[slot]) == int(by)
            and int(block_iz[slot]) == int(bz)
        ):
            return (block_bits[slot] & bit) != np.uint64(0)
        slot = (slot + 1) & mask
    return False


@njit(cache=True)
def _cleaved_contains_coord(
    ix: int,
    iy: int,
    iz: int,
    h_ix: np.ndarray,
    h_iy: np.ndarray,
    h_iz: np.ndarray,
    h_used: np.ndarray,
    block_ix: np.ndarray,
    block_iy: np.ndarray,
    block_iz: np.ndarray,
    block_bits: np.ndarray,
    block_used: np.ndarray,
) -> bool:
    if len(block_used) > 0:
        return _cleaved_block_hash_contains_coord(
            ix, iy, iz, block_ix, block_iy, block_iz, block_bits, block_used
        )
    return _coord_hash_contains(ix, iy, iz, h_ix, h_iy, h_iz, h_used)


@njit(cache=True)
def _cleaved_block_hash_insert_no_grow(
    ix: int,
    iy: int,
    iz: int,
    block_ix: np.ndarray,
    block_iy: np.ndarray,
    block_iz: np.ndarray,
    block_bits: np.ndarray,
    block_used: np.ndarray,
) -> Tuple[bool, bool]:
    cap = len(block_used)
    if cap <= 0:
        return False, False
    bx, by, bz, bit = _cleaved_block_key_and_bit(ix, iy, iz)
    mask = cap - 1
    slot = _coord_hash_slot(bx, by, bz, mask)
    for _ in range(cap):
        if block_used[slot] == 0:
            block_used[slot] = np.uint8(1)
            block_ix[slot] = np.int64(bx)
            block_iy[slot] = np.int64(by)
            block_iz[slot] = np.int64(bz)
            block_bits[slot] = bit
            return True, True
        if (
            int(block_ix[slot]) == int(bx)
            and int(block_iy[slot]) == int(by)
            and int(block_iz[slot]) == int(bz)
        ):
            if (block_bits[slot] & bit) != np.uint64(0):
                return False, False
            block_bits[slot] = block_bits[slot] | bit
            return True, False
        slot = (slot + 1) & mask
    return False, False


@njit(cache=True)
def _rehash_cleaved_block_table(
    old_ix: np.ndarray,
    old_iy: np.ndarray,
    old_iz: np.ndarray,
    old_bits: np.ndarray,
    old_used: np.ndarray,
    target_cap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cap = 16
    while cap < target_cap:
        cap *= 2
    new_ix = np.zeros(cap, dtype=np.int64)
    new_iy = np.zeros(cap, dtype=np.int64)
    new_iz = np.zeros(cap, dtype=np.int64)
    new_bits = np.zeros(cap, dtype=np.uint64)
    new_used = np.zeros(cap, dtype=np.uint8)
    mask = cap - 1
    for i in range(len(old_used)):
        if old_used[i] == 0:
            continue
        bx = int(old_ix[i])
        by = int(old_iy[i])
        bz = int(old_iz[i])
        slot = _coord_hash_slot(bx, by, bz, mask)
        for _ in range(cap):
            if new_used[slot] == 0:
                new_used[slot] = np.uint8(1)
                new_ix[slot] = np.int64(bx)
                new_iy[slot] = np.int64(by)
                new_iz[slot] = np.int64(bz)
                new_bits[slot] = np.uint64(old_bits[i])
                break
            if (
                int(new_ix[slot]) == bx
                and int(new_iy[slot]) == by
                and int(new_iz[slot]) == bz
            ):
                new_bits[slot] = new_bits[slot] | np.uint64(old_bits[i])
                break
            slot = (slot + 1) & mask
    return new_ix, new_iy, new_iz, new_bits, new_used


@njit(cache=True)
def _build_cleaved_block_hash_from_coords(
    cleaved_ix: np.ndarray,
    cleaved_iy: np.ndarray,
    cleaved_iz: np.ndarray,
    n_cleaved: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    target = max(16, max(1, int(n_cleaved) // 8))
    cap = 16
    while cap < target:
        cap *= 2
    block_ix = np.zeros(cap, dtype=np.int64)
    block_iy = np.zeros(cap, dtype=np.int64)
    block_iz = np.zeros(cap, dtype=np.int64)
    block_bits = np.zeros(cap, dtype=np.uint64)
    block_used = np.zeros(cap, dtype=np.uint8)
    block_count = 0
    n = max(0, min(int(n_cleaved), len(cleaved_ix), len(cleaved_iy), len(cleaved_iz)))
    for i in range(n):
        if block_count * 10 >= 7 * len(block_used):
            block_ix, block_iy, block_iz, block_bits, block_used = (
                _rehash_cleaved_block_table(
                    block_ix,
                    block_iy,
                    block_iz,
                    block_bits,
                    block_used,
                    max(16, len(block_used) * 2),
                )
            )
        inserted, new_block = _cleaved_block_hash_insert_no_grow(
            int(cleaved_ix[i]),
            int(cleaved_iy[i]),
            int(cleaved_iz[i]),
            block_ix,
            block_iy,
            block_iz,
            block_bits,
            block_used,
        )
        if inserted and new_block:
            block_count += 1
    return block_ix, block_iy, block_iz, block_bits, block_used, block_count


@njit(cache=True)
def _rehash_coord_table(
    old_ix: np.ndarray,
    old_iy: np.ndarray,
    old_iz: np.ndarray,
    old_used: np.ndarray,
    new_cap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if new_cap < 16:
        new_cap = 16
    cap = 1
    while cap < new_cap:
        cap *= 2
    h_ix = np.zeros(cap, dtype=np.int64)
    h_iy = np.zeros(cap, dtype=np.int64)
    h_iz = np.zeros(cap, dtype=np.int64)
    h_used = np.zeros(cap, dtype=np.uint8)
    for i in range(len(old_used)):
        if old_used[i] == 0:
            continue
        _coord_hash_insert_no_grow(
            int(old_ix[i]),
            int(old_iy[i]),
            int(old_iz[i]),
            h_ix,
            h_iy,
            h_iz,
            h_used,
        )
    return h_ix, h_iy, h_iz, h_used


@njit(cache=True)
def _count_used_slots(h_used: np.ndarray) -> int:
    n = 0
    for i in range(len(h_used)):
        if h_used[i] != 0:
            n += 1
    return n


@njit(cache=True)
def _find_bound_ligand_for_coord(
    ix: int,
    iy: int,
    iz: int,
    bound_active: np.ndarray,
    bound_ix: np.ndarray,
    bound_iy: np.ndarray,
    bound_iz: np.ndarray,
) -> int:
    for lig_idx in range(len(bound_active)):
        if not bool(bound_active[lig_idx]):
            continue
        if (
            int(bound_ix[lig_idx]) == int(ix)
            and int(bound_iy[lig_idx]) == int(iy)
            and int(bound_iz[lig_idx]) == int(iz)
        ):
            return lig_idx
    return -1


@njit(cache=True)
def _count_bound_active(bound_active: np.ndarray) -> int:
    n = 0
    for i in range(len(bound_active)):
        if bool(bound_active[i]):
            n += 1
    return n


@njit(cache=True)
def _estimate_candidate_capacity(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    L: float,
    spacing: float,
    cutoff: float,
) -> int:
    p0x = x - 0.5 * L * n_x
    p0y = y - 0.5 * L * n_y
    p0z = z - 0.5 * L * n_z
    p1x = x + 0.5 * L * n_x
    p1y = y + 0.5 * L * n_y
    p1z = z + 0.5 * L * n_z
    min_x = p0x if p0x < p1x else p1x
    max_x = p1x if p1x > p0x else p0x
    min_y = p0y if p0y < p1y else p1y
    max_y = p1y if p1y > p0y else p0y
    min_z = p0z if p0z < p1z else p1z
    max_z = p1z if p1z > p0z else p0z
    ix_min = int(np.floor((min_x - cutoff) / spacing))
    ix_max = int(np.ceil((max_x + cutoff) / spacing))
    iy_min = int(np.floor((min_y - cutoff) / spacing))
    iy_max = int(np.ceil((max_y + cutoff) / spacing))
    iz_min = int(np.floor((min_z - cutoff) / spacing))
    iz_max = int(np.ceil((max_z + cutoff) / spacing))
    nx_box = ix_max - ix_min + 1
    ny_box = iy_max - iy_min + 1
    nz_box = iz_max - iz_min + 1
    if nx_box < 1:
        nx_box = 1
    if ny_box < 1:
        ny_box = 1
    if nz_box < 1:
        nz_box = 1
    total = nx_box * ny_box * nz_box
    if total < 64:
        total = 64
    return int(total)


def _gradient_axis_code_from_type(gradient_type: str) -> int:
    gt = str(gradient_type).strip().lower()
    if gt == "x":
        return 0
    if gt == "y":
        return 1
    if gt == "z":
        return 2
    return -1


def _build_gradient_axis_runtime_spec(config: Dict[str, Any]) -> Dict[str, Any]:
    return build_gradient_axis_spec(
        spacing=float(config["RECEPTOR_SPACING"]),
        gradient_scale=float(config["GRADIENT_SCALE"]),
        min_spacing=float(
            config.get("GRADIENT_MIN_SPACING", DEFAULT_GRADIENT_MIN_SPACING)
        ),
        max_spacing=float(
            config.get("GRADIENT_MAX_SPACING", DEFAULT_GRADIENT_MAX_SPACING)
        ),
    )


def _gradient_dense_direction_sign(config: Dict[str, Any]) -> int:
    gradient_scale = float(config.get("GRADIENT_SCALE", 1.0))
    if gradient_scale < 1.0:
        return 1
    if gradient_scale > 1.0:
        return -1
    return 0


def _gradient_axis_value_for_state(
    x: float, y: float, z: float, gradient_axis_code: int
) -> float:
    if gradient_axis_code == 0:
        return float(x)
    if gradient_axis_code == 1:
        return float(y)
    return float(z)


def _build_gradient_escape_geometry(
    config: Dict[str, Any],
    gradient_axis_code: int,
    dense_sign: int,
    axis_origin: float,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> Tuple[bool, float, float, float]:
    stop_mode = normalize_gradient_escape_stop_mode(
        str(config.get("GRADIENT_ESCAPE_STOP_MODE", GRADIENT_ESCAPE_STOP_MODE_OFF))
    )
    if stop_mode != GRADIENT_ESCAPE_STOP_MODE_AXIS_PLANE_DWELL:
        return False, 0.0, 0.0, 0.0
    if gradient_axis_code < 0 or dense_sign == 0:
        return False, 0.0, 0.0, 0.0

    stop_threshold = float(
        config.get("GRADIENT_MIN_SPACING_STOP", DEFAULT_GRADIENT_MIN_SPACING_STOP)
    )
    if stop_threshold <= 0.0:
        return False, 0.0, 0.0, 0.0

    u0 = float(dense_coordinate_from_axis_position(float(axis_origin), int(dense_sign)))
    u_dense_stop = float(
        axis_first_dense_coordinate_with_spacing_at_most(
            float(stop_threshold),
            int(dense_sign),
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
    )
    u_sparse_tail = float(
        axis_first_dense_coordinate_with_spacing_at_least_on_sparse_side(
            float(config.get("GRADIENT_MAX_SPACING", DEFAULT_GRADIENT_MAX_SPACING)),
            int(dense_sign),
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
    )
    if not np.isfinite(u_dense_stop) or not np.isfinite(u_sparse_tail):
        return False, 0.0, 0.0, 0.0

    u_sparse_sym = float(u0 - (u_dense_stop - u0))
    u_escape = float(u_sparse_sym if u_sparse_sym > u_sparse_tail else u_sparse_tail)
    return True, u_escape, u_sparse_tail, u0


def _advance_gradient_escape_state(
    axis_edge: float,
    dt_since_last_eval: float,
    plane: float,
    return_margin: float,
    dwell_time: float,
    armed: bool,
    dwell_elapsed: float,
) -> Tuple[bool, float, bool]:
    elapsed = float(dwell_elapsed)
    if bool(armed):
        elapsed += max(0.0, float(dt_since_last_eval))

    armed_now = bool(armed)
    if float(axis_edge) <= float(plane):
        armed_now = True
    elif armed_now and float(axis_edge) > (float(plane) + float(return_margin)):
        armed_now = False
        elapsed = 0.0

    triggered = bool(armed_now) and (
        float(dwell_time) <= 0.0 or float(elapsed) >= float(dwell_time)
    )
    return armed_now, elapsed, triggered


def _resolve_trajectory_record_interval(config: Dict[str, Any]) -> float:
    explicit = float(config.get("TRAJECTORY_RECORD_INTERVAL", 0.0))
    if explicit > 0.0:
        return explicit
    t_final = max(0.0, float(config.get("T_FINAL", 0.0)))
    if t_final <= 0.0:
        return DEFAULT_TRAJECTORY_RECORD_MIN_INTERVAL
    return max(
        DEFAULT_TRAJECTORY_RECORD_MIN_INTERVAL,
        t_final / float(DEFAULT_TRAJECTORY_RECORD_TARGET_POINTS),
    )


def _append_trajectory_samples_with_optional_backfill(
    times_list: list[float],
    positions_list: list[np.ndarray],
    *,
    dimension: str,
    last_t: float,
    last_x: float,
    last_y: float,
    last_z: float,
    last_n: np.ndarray,
    t: float,
    x: float,
    y: float,
    z: float,
    n_hat: np.ndarray,
    time_threshold: float,
    dense_backfill: bool,
) -> Tuple[float, float, float, float, np.ndarray]:
    def _append_one(
        t_store: float,
        x_store: float,
        y_store: float,
        z_store: float,
        n_store: np.ndarray,
    ) -> None:
        if times_list and abs(float(times_list[-1]) - float(t_store)) <= 1.0e-10:
            return
        times_list.append(float(t_store))
        positions_list.append(
            build_position_row_for_dimension(
                float(x_store),
                float(y_store),
                float(z_store),
                np.asarray(n_store, dtype=np.float64),
                dimension,
            )
        )

    last_n_arr = np.asarray(last_n, dtype=np.float64)
    n_hat_arr = np.asarray(n_hat, dtype=np.float64)
    total_dt = float(t) - float(last_t)
    if dense_backfill and time_threshold > 0.0 and total_dt > time_threshold + 1.0e-12:
        sample_t = float(last_t) + float(time_threshold)
        while sample_t < float(t) - 1.0e-10:
            frac = (sample_t - float(last_t)) / total_dt
            interp_n = last_n_arr + frac * (n_hat_arr - last_n_arr)
            interp_norm = float(np.linalg.norm(interp_n))
            if interp_norm > 1.0e-12:
                interp_n = interp_n / interp_norm
            else:
                interp_n = n_hat_arr.copy()
            _append_one(
                sample_t,
                float(last_x) + frac * (float(x) - float(last_x)),
                float(last_y) + frac * (float(y) - float(last_y)),
                float(last_z) + frac * (float(z) - float(last_z)),
                interp_n,
            )
            sample_t += float(time_threshold)

    _append_one(float(t), float(x), float(y), float(z), n_hat_arr)
    return float(x), float(y), float(z), float(t), n_hat_arr.copy()


@njit(cache=True)
def _coord_to_position_gradient_sparse(
    ix: int,
    iy: int,
    iz: int,
    spacing: float,
    gradient_axis_code: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> Tuple[float, float, float]:
    if gradient_axis_code == 0:
        return (
            float(
                axis_position_from_index_implicit(
                    ix, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
                )
            ),
            float(iy) * spacing,
            float(iz) * spacing,
        )
    if gradient_axis_code == 1:
        return (
            float(ix) * spacing,
            float(
                axis_position_from_index_implicit(
                    iy, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
                )
            ),
            float(iz) * spacing,
        )
    return (
        float(ix) * spacing,
        float(iy) * spacing,
        float(
            axis_position_from_index_implicit(
                iz, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
            )
        ),
    )


@njit(cache=True)
def _fill_axis_position_window_implicit(
    idx_min: int,
    idx_max: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
    out: np.ndarray,
) -> None:
    write_k = 0
    for idx in range(idx_min, idx_max + 1):
        out[write_k] = axis_position_from_index_implicit(
            idx,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        write_k += 1


def _compute_gradient_stop_interval_metrics(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    L: float,
    cutoff: float,
    gradient_axis_code: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> Tuple[float, float, float]:
    half_L = 0.5 * float(L)
    if gradient_axis_code == 0:
        axis_center = float(x)
        axis_dir = float(n_x)
    elif gradient_axis_code == 1:
        axis_center = float(y)
        axis_dir = float(n_y)
    else:
        axis_center = float(z)
        axis_dir = float(n_z)
    axis_min = axis_center - abs(half_L * axis_dir) - float(cutoff)
    axis_max = axis_center + abs(half_L * axis_dir) + float(cutoff)
    min_spacing = float(
        axis_min_spacing_in_interval_implicit(
            axis_min,
            axis_max,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
    )
    return axis_min, axis_max, min_spacing


@njit(cache=True)
def _build_candidate_cache_gradient_sparse_geometry(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    L: float,
    spacing: float,
    r_i: np.ndarray,
    cutoff: float,
    gradient_axis_code: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
    cleaved_hash_ix: np.ndarray,
    cleaved_hash_iy: np.ndarray,
    cleaved_hash_iz: np.ndarray,
    cleaved_hash_used: np.ndarray,
    cand_ix: np.ndarray,
    cand_iy: np.ndarray,
    cand_iz: np.ndarray,
    cand_rx: np.ndarray,
    cand_ry: np.ndarray,
    cand_rz: np.ndarray,
) -> int:
    half_L = 0.5 * L
    cutoff_sq = cutoff * cutoff
    min_x = x - abs(half_L * n_x) - cutoff
    max_x = x + abs(half_L * n_x) + cutoff
    min_y = y - abs(half_L * n_y) - cutoff
    max_y = y + abs(half_L * n_y) + cutoff
    min_z = z - abs(half_L * n_z) - cutoff
    max_z = z + abs(half_L * n_z) + cutoff

    if gradient_axis_code == 0:
        ix_min = axis_lower_bound_implicit(
            min_x, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        ix_max = axis_upper_bound_implicit(
            max_x, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        iy_min = int(np.floor(min_y / spacing))
        iy_max = int(np.ceil(max_y / spacing))
        iz_min = int(np.floor(min_z / spacing))
        iz_max = int(np.ceil(max_z / spacing))
    elif gradient_axis_code == 1:
        ix_min = int(np.floor(min_x / spacing))
        ix_max = int(np.ceil(max_x / spacing))
        iy_min = axis_lower_bound_implicit(
            min_y, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        iy_max = axis_upper_bound_implicit(
            max_y, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        iz_min = int(np.floor(min_z / spacing))
        iz_max = int(np.ceil(max_z / spacing))
    else:
        ix_min = int(np.floor(min_x / spacing))
        ix_max = int(np.ceil(max_x / spacing))
        iy_min = int(np.floor(min_y / spacing))
        iy_max = int(np.ceil(max_y / spacing))
        iz_min = axis_lower_bound_implicit(
            min_z, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        iz_max = axis_upper_bound_implicit(
            max_z, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )

    n_ligands = len(r_i)
    if n_ligands <= 0:
        return 0
    r0 = r_i[0]
    inv_dr = 1.0
    if n_ligands > 1:
        dr = r_i[1] - r_i[0]
        if dr != 0.0:
            inv_dr = 1.0 / dr

    if gradient_axis_code == 0:
        axis_positions = np.empty(max(1, ix_max - ix_min + 1), dtype=np.float64)
        _fill_axis_position_window_implicit(
            ix_min,
            ix_max,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
            axis_positions,
        )
    elif gradient_axis_code == 1:
        axis_positions = np.empty(max(1, iy_max - iy_min + 1), dtype=np.float64)
        _fill_axis_position_window_implicit(
            iy_min,
            iy_max,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
            axis_positions,
        )
    else:
        axis_positions = np.empty(max(1, iz_max - iz_min + 1), dtype=np.float64)
        _fill_axis_position_window_implicit(
            iz_min,
            iz_max,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
            axis_positions,
        )

    count = 0
    cap = len(cand_ix)
    for ix in range(ix_min, ix_max + 1):
        if gradient_axis_code == 0:
            rx = axis_positions[ix - ix_min]
        else:
            rx = float(ix) * spacing
        for iy in range(iy_min, iy_max + 1):
            if gradient_axis_code == 1:
                ry = axis_positions[iy - iy_min]
            else:
                ry = float(iy) * spacing
            for iz in range(iz_min, iz_max + 1):
                if _coord_hash_contains(
                    ix,
                    iy,
                    iz,
                    cleaved_hash_ix,
                    cleaved_hash_iy,
                    cleaved_hash_iz,
                    cleaved_hash_used,
                ):
                    continue
                if gradient_axis_code == 2:
                    rz = axis_positions[iz - iz_min]
                else:
                    rz = float(iz) * spacing
                ddx = rx - x
                ddy = ry - y
                ddz = rz - z
                s = ddx * n_x + ddy * n_y + ddz * n_z
                dist_center_sq = ddx * ddx + ddy * ddy + ddz * ddz
                rho_sq = dist_center_sq - s * s
                if rho_sq < 0.0:
                    rho_sq = 0.0
                if rho_sq > cutoff_sq:
                    continue
                delta_sq = cutoff_sq - rho_sq
                if delta_sq < 0.0:
                    continue
                delta = np.sqrt(delta_sq)

                if n_ligands == 1:
                    if (s < (r0 - delta)) or (s > (r0 + delta)):
                        continue
                else:
                    i_min = int(np.ceil((s - delta - r0) * inv_dr))
                    i_max = int(np.floor((s + delta - r0) * inv_dr))
                    if i_min < 0:
                        i_min = 0
                    if i_max >= n_ligands:
                        i_max = n_ligands - 1
                    if i_max < i_min:
                        continue

                if count >= cap:
                    return -1
                cand_ix[count] = np.int64(ix)
                cand_iy[count] = np.int64(iy)
                cand_iz[count] = np.int64(iz)
                cand_rx[count] = rx
                cand_ry[count] = ry
                cand_rz[count] = rz
                count += 1
    return count


@njit(cache=True)
def _build_uniform_reaction_geometry(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    L: float,
    spacing: float,
    cutoff: float,
    max_bond_sq: float,
    r_i: np.ndarray,
    cleaved_hash_ix: np.ndarray,
    cleaved_hash_iy: np.ndarray,
    cleaved_hash_iz: np.ndarray,
    cleaved_hash_used: np.ndarray,
    cleaved_block_ix: np.ndarray,
    cleaved_block_iy: np.ndarray,
    cleaved_block_iz: np.ndarray,
    cleaved_block_bits: np.ndarray,
    cleaved_block_used: np.ndarray,
    bound_active: np.ndarray,
    bound_ix: np.ndarray,
    bound_iy: np.ndarray,
    bound_iz: np.ndarray,
    ligand_bound: np.ndarray,
    free_ix: np.ndarray,
    free_iy: np.ndarray,
    free_iz: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    bound_slot_ix: np.ndarray,
    bound_slot_iy: np.ndarray,
    bound_slot_iz: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Build the local reaction geometry in one sparse-lattice scan."""
    n_ligands = len(r_i)
    cutoff_sq = cutoff * cutoff
    n_free = 0
    n_bound = 0
    n_candidates = 0
    bound_outside_cutoff = 0

    for lig_idx in range(n_ligands):
        ligand_bound[lig_idx] = False

    for lig_idx in range(n_ligands):
        if not bool(bound_active[lig_idx]):
            continue
        rx = float(bound_ix[lig_idx]) * spacing
        ry = float(bound_iy[lig_idx]) * spacing
        rz = float(bound_iz[lig_idx]) * spacing
        x_lig = x + r_i[lig_idx] * n_x
        y_lig = y + r_i[lig_idx] * n_y
        z_lig = z + r_i[lig_idx] * n_z
        ddx = rx - x_lig
        ddy = ry - y_lig
        ddz = rz - z_lig
        dr2 = ddx * ddx + ddy * ddy + ddz * ddz
        if dr2 > max_bond_sq:
            bound_active[lig_idx] = False
            continue
        ligand_bound[lig_idx] = True
        bound_slot_ix[n_bound] = np.int64(bound_ix[lig_idx])
        bound_slot_iy[n_bound] = np.int64(bound_iy[lig_idx])
        bound_slot_iz[n_bound] = np.int64(bound_iz[lig_idx])
        bound_ligand_idx[n_bound] = np.int32(lig_idx)
        bound_rx[n_bound] = rx
        bound_ry[n_bound] = ry
        bound_rz[n_bound] = rz
        bound_dx[n_bound] = ddx
        bound_dy[n_bound] = ddy
        bound_dz[n_bound] = ddz
        if dr2 > cutoff_sq:
            bound_outside_cutoff += 1
        n_bound += 1

    if n_ligands <= 0:
        return 0, 0, n_bound, bound_outside_cutoff

    half_L = 0.5 * L
    min_x = x - abs(half_L * n_x) - cutoff
    max_x = x + abs(half_L * n_x) + cutoff
    min_y = y - abs(half_L * n_y) - cutoff
    max_y = y + abs(half_L * n_y) + cutoff
    min_z = z - abs(half_L * n_z) - cutoff
    max_z = z + abs(half_L * n_z) + cutoff

    ix_min = int(np.floor(min_x / spacing))
    ix_max = int(np.ceil(max_x / spacing))
    iy_min = int(np.floor(min_y / spacing))
    iy_max = int(np.ceil(max_y / spacing))
    iz_min = int(np.floor(min_z / spacing))
    iz_max = int(np.ceil(max_z / spacing))

    r0 = r_i[0]
    inv_dr = 1.0
    if n_ligands > 1:
        dr = r_i[1] - r_i[0]
        if dr != 0.0:
            inv_dr = 1.0 / dr

    cap = len(free_ix)
    for ix in range(ix_min, ix_max + 1):
        rx = float(ix) * spacing
        for iy in range(iy_min, iy_max + 1):
            ry = float(iy) * spacing
            for iz in range(iz_min, iz_max + 1):
                rz = float(iz) * spacing
                ddx = rx - x
                ddy = ry - y
                ddz = rz - z
                s = ddx * n_x + ddy * n_y + ddz * n_z
                dist_center_sq = ddx * ddx + ddy * ddy + ddz * ddz
                rho_sq = dist_center_sq - s * s
                if rho_sq < 0.0:
                    rho_sq = 0.0
                if rho_sq > cutoff_sq:
                    continue
                delta_sq = cutoff_sq - rho_sq
                if delta_sq < 0.0:
                    continue
                delta = np.sqrt(delta_sq)

                if n_ligands == 1:
                    i_min = 0
                    i_max = 0
                    if (s < (r0 - delta)) or (s > (r0 + delta)):
                        continue
                else:
                    i_min = int(np.ceil((s - delta - r0) * inv_dr))
                    i_max = int(np.floor((s + delta - r0) * inv_dr))
                    if i_min < 0:
                        i_min = 0
                    if i_max >= n_ligands:
                        i_max = n_ligands - 1
                    if i_max < i_min:
                        continue

                if _cleaved_contains_coord(
                    ix,
                    iy,
                    iz,
                    cleaved_hash_ix,
                    cleaved_hash_iy,
                    cleaved_hash_iz,
                    cleaved_hash_used,
                    cleaved_block_ix,
                    cleaved_block_iy,
                    cleaved_block_iz,
                    cleaved_block_bits,
                    cleaved_block_used,
                ):
                    continue

                n_candidates += 1
                is_bound_coord = False
                for slot in range(n_bound):
                    if (
                        int(bound_slot_ix[slot]) == ix
                        and int(bound_slot_iy[slot]) == iy
                        and int(bound_slot_iz[slot]) == iz
                    ):
                        is_bound_coord = True
                        break
                if is_bound_coord:
                    continue

                if n_free >= cap:
                    return n_candidates, -1, n_bound, bound_outside_cutoff
                free_ix[n_free] = np.int64(ix)
                free_iy[n_free] = np.int64(iy)
                free_iz[n_free] = np.int64(iz)
                free_s[n_free] = s
                free_rho_sq[n_free] = rho_sq
                free_i_min[n_free] = np.int32(i_min)
                free_i_max[n_free] = np.int32(i_max)
                n_free += 1

    return n_candidates, n_free, n_bound, bound_outside_cutoff


@njit(cache=True)
def _refresh_uniform_reaction_geometry(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    spacing: float,
    cutoff: float,
    max_bond_sq: float,
    r_i: np.ndarray,
    bound_active: np.ndarray,
    bound_ix: np.ndarray,
    bound_iy: np.ndarray,
    bound_iz: np.ndarray,
    ligand_bound: np.ndarray,
    n_free: int,
    free_ix: np.ndarray,
    free_iy: np.ndarray,
    free_iz: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    bound_slot_ix: np.ndarray,
    bound_slot_iy: np.ndarray,
    bound_slot_iz: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
) -> Tuple[int, int, int]:
    """Refresh event-time geometry on a previously discovered free-site set."""
    n_ligands = len(r_i)
    cutoff_sq = cutoff * cutoff
    n_bound = 0
    valid_free = 0
    bound_outside_cutoff = 0

    for lig_idx in range(n_ligands):
        ligand_bound[lig_idx] = False

    for lig_idx in range(n_ligands):
        if not bool(bound_active[lig_idx]):
            continue
        rx = float(bound_ix[lig_idx]) * spacing
        ry = float(bound_iy[lig_idx]) * spacing
        rz = float(bound_iz[lig_idx]) * spacing
        x_lig = x + r_i[lig_idx] * n_x
        y_lig = y + r_i[lig_idx] * n_y
        z_lig = z + r_i[lig_idx] * n_z
        ddx = rx - x_lig
        ddy = ry - y_lig
        ddz = rz - z_lig
        dr2 = ddx * ddx + ddy * ddy + ddz * ddz
        if dr2 > max_bond_sq:
            bound_active[lig_idx] = False
            continue
        ligand_bound[lig_idx] = True
        bound_slot_ix[n_bound] = np.int64(bound_ix[lig_idx])
        bound_slot_iy[n_bound] = np.int64(bound_iy[lig_idx])
        bound_slot_iz[n_bound] = np.int64(bound_iz[lig_idx])
        bound_ligand_idx[n_bound] = np.int32(lig_idx)
        bound_rx[n_bound] = rx
        bound_ry[n_bound] = ry
        bound_rz[n_bound] = rz
        bound_dx[n_bound] = ddx
        bound_dy[n_bound] = ddy
        bound_dz[n_bound] = ddz
        if dr2 > cutoff_sq:
            bound_outside_cutoff += 1
        n_bound += 1

    if n_ligands <= 0:
        return 0, n_bound, bound_outside_cutoff

    r0 = r_i[0]
    inv_dr = 1.0
    if n_ligands > 1:
        dr = r_i[1] - r_i[0]
        if dr != 0.0:
            inv_dr = 1.0 / dr

    for slot in range(n_free):
        rx = float(free_ix[slot]) * spacing
        ry = float(free_iy[slot]) * spacing
        rz = float(free_iz[slot]) * spacing
        ddx = rx - x
        ddy = ry - y
        ddz = rz - z
        s = ddx * n_x + ddy * n_y + ddz * n_z
        dist_center_sq = ddx * ddx + ddy * ddy + ddz * ddz
        rho_sq = dist_center_sq - s * s
        if rho_sq < 0.0:
            rho_sq = 0.0
        free_s[slot] = s
        free_rho_sq[slot] = rho_sq
        if rho_sq > cutoff_sq:
            free_i_min[slot] = np.int32(1)
            free_i_max[slot] = np.int32(0)
            continue
        delta_sq = cutoff_sq - rho_sq
        if delta_sq < 0.0:
            free_i_min[slot] = np.int32(1)
            free_i_max[slot] = np.int32(0)
            continue
        delta = np.sqrt(delta_sq)
        if n_ligands == 1:
            if (s < (r0 - delta)) or (s > (r0 + delta)):
                free_i_min[slot] = np.int32(1)
                free_i_max[slot] = np.int32(0)
                continue
            i_min = 0
            i_max = 0
        else:
            i_min = int(np.ceil((s - delta - r0) * inv_dr))
            i_max = int(np.floor((s + delta - r0) * inv_dr))
            if i_min < 0:
                i_min = 0
            if i_max >= n_ligands:
                i_max = n_ligands - 1
            if i_max < i_min:
                free_i_min[slot] = np.int32(1)
                free_i_max[slot] = np.int32(0)
                continue
        free_i_min[slot] = np.int32(i_min)
        free_i_max[slot] = np.int32(i_max)
        valid_free += 1

    return valid_free, n_bound, bound_outside_cutoff


@njit(cache=True)
def _refresh_gradient_reaction_geometry(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    spacing: float,
    cutoff: float,
    max_bond_sq: float,
    r_i: np.ndarray,
    gradient_axis_code: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
    bound_active: np.ndarray,
    bound_ix: np.ndarray,
    bound_iy: np.ndarray,
    bound_iz: np.ndarray,
    ligand_bound: np.ndarray,
    n_free: int,
    free_ix: np.ndarray,
    free_iy: np.ndarray,
    free_iz: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    bound_slot_ix: np.ndarray,
    bound_slot_iy: np.ndarray,
    bound_slot_iz: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
) -> Tuple[int, int, int]:
    """Refresh event-time geometry on existing free sites for gradient sparse grids."""
    n_ligands = len(r_i)
    cutoff_sq = cutoff * cutoff
    n_bound = 0
    valid_free = 0
    bound_outside_cutoff = 0

    for lig_idx in range(n_ligands):
        ligand_bound[lig_idx] = False

    for lig_idx in range(n_ligands):
        if not bool(bound_active[lig_idx]):
            continue
        rx, ry, rz = _coord_to_position_gradient_sparse(
            int(bound_ix[lig_idx]),
            int(bound_iy[lig_idx]),
            int(bound_iz[lig_idx]),
            spacing,
            gradient_axis_code,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        x_lig = x + r_i[lig_idx] * n_x
        y_lig = y + r_i[lig_idx] * n_y
        z_lig = z + r_i[lig_idx] * n_z
        ddx = rx - x_lig
        ddy = ry - y_lig
        ddz = rz - z_lig
        dr2 = ddx * ddx + ddy * ddy + ddz * ddz
        if dr2 > max_bond_sq:
            bound_active[lig_idx] = False
            continue
        ligand_bound[lig_idx] = True
        bound_slot_ix[n_bound] = np.int64(bound_ix[lig_idx])
        bound_slot_iy[n_bound] = np.int64(bound_iy[lig_idx])
        bound_slot_iz[n_bound] = np.int64(bound_iz[lig_idx])
        bound_ligand_idx[n_bound] = np.int32(lig_idx)
        bound_rx[n_bound] = rx
        bound_ry[n_bound] = ry
        bound_rz[n_bound] = rz
        bound_dx[n_bound] = ddx
        bound_dy[n_bound] = ddy
        bound_dz[n_bound] = ddz
        if dr2 > cutoff_sq:
            bound_outside_cutoff += 1
        n_bound += 1

    if n_ligands <= 0:
        return 0, n_bound, bound_outside_cutoff

    r0 = r_i[0]
    inv_dr = 1.0
    if n_ligands > 1:
        dr = r_i[1] - r_i[0]
        if dr != 0.0:
            inv_dr = 1.0 / dr

    for slot in range(n_free):
        rx, ry, rz = _coord_to_position_gradient_sparse(
            int(free_ix[slot]),
            int(free_iy[slot]),
            int(free_iz[slot]),
            spacing,
            gradient_axis_code,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        ddx = rx - x
        ddy = ry - y
        ddz = rz - z
        s = ddx * n_x + ddy * n_y + ddz * n_z
        dist_center_sq = ddx * ddx + ddy * ddy + ddz * ddz
        rho_sq = dist_center_sq - s * s
        if rho_sq < 0.0:
            rho_sq = 0.0
        free_s[slot] = s
        free_rho_sq[slot] = rho_sq
        if rho_sq > cutoff_sq:
            free_i_min[slot] = np.int32(1)
            free_i_max[slot] = np.int32(0)
            continue
        delta_sq = cutoff_sq - rho_sq
        if delta_sq < 0.0:
            free_i_min[slot] = np.int32(1)
            free_i_max[slot] = np.int32(0)
            continue
        delta = np.sqrt(delta_sq)
        if n_ligands == 1:
            if (s < (r0 - delta)) or (s > (r0 + delta)):
                free_i_min[slot] = np.int32(1)
                free_i_max[slot] = np.int32(0)
                continue
            i_min = 0
            i_max = 0
        else:
            i_min = int(np.ceil((s - delta - r0) * inv_dr))
            i_max = int(np.floor((s + delta - r0) * inv_dr))
            if i_min < 0:
                i_min = 0
            if i_max >= n_ligands:
                i_max = n_ligands - 1
            if i_max < i_min:
                free_i_min[slot] = np.int32(1)
                free_i_max[slot] = np.int32(0)
                continue
        free_i_min[slot] = np.int32(i_min)
        free_i_max[slot] = np.int32(i_max)
        valid_free += 1

    return valid_free, n_bound, bound_outside_cutoff


@njit(cache=True)
def _build_candidate_cache_uniform_sparse_geometry_2d(
    x: float,
    y: float,
    n_x: float,
    n_y: float,
    L: float,
    spacing: float,
    r_i: np.ndarray,
    cutoff: float,
    cleaved_hash_ix: np.ndarray,
    cleaved_hash_iy: np.ndarray,
    cleaved_hash_iz: np.ndarray,
    cleaved_hash_used: np.ndarray,
    cleaved_block_ix: np.ndarray,
    cleaved_block_iy: np.ndarray,
    cleaved_block_iz: np.ndarray,
    cleaved_block_bits: np.ndarray,
    cleaved_block_used: np.ndarray,
    cand_ix: np.ndarray,
    cand_iy: np.ndarray,
    cand_iz: np.ndarray,
    cand_rx: np.ndarray,
    cand_ry: np.ndarray,
    cand_rz: np.ndarray,
) -> int:
    half_L = 0.5 * L
    cutoff_sq = cutoff * cutoff
    min_x = x - abs(half_L * n_x) - cutoff
    max_x = x + abs(half_L * n_x) + cutoff
    min_y = y - abs(half_L * n_y) - cutoff
    max_y = y + abs(half_L * n_y) + cutoff

    ix_min = int(np.floor(min_x / spacing))
    ix_max = int(np.ceil(max_x / spacing))
    iy_min = int(np.floor(min_y / spacing))
    iy_max = int(np.ceil(max_y / spacing))

    n_ligands = len(r_i)
    if n_ligands <= 0:
        return 0
    r0 = r_i[0]
    inv_dr = 1.0
    if n_ligands > 1:
        dr = r_i[1] - r_i[0]
        if dr != 0.0:
            inv_dr = 1.0 / dr

    count = 0
    cap = len(cand_ix)
    for ix in range(ix_min, ix_max + 1):
        rx = float(ix) * spacing
        for iy in range(iy_min, iy_max + 1):
            if _cleaved_contains_coord(
                ix,
                iy,
                0,
                cleaved_hash_ix,
                cleaved_hash_iy,
                cleaved_hash_iz,
                cleaved_hash_used,
                cleaved_block_ix,
                cleaved_block_iy,
                cleaved_block_iz,
                cleaved_block_bits,
                cleaved_block_used,
            ):
                continue
            ry = float(iy) * spacing
            ddx = rx - x
            ddy = ry - y
            s = ddx * n_x + ddy * n_y
            dist_center_sq = ddx * ddx + ddy * ddy
            rho_sq = dist_center_sq - s * s
            if rho_sq < 0.0:
                rho_sq = 0.0
            if rho_sq > cutoff_sq:
                continue
            delta_sq = cutoff_sq - rho_sq
            if delta_sq < 0.0:
                continue
            delta = np.sqrt(delta_sq)
            if n_ligands == 1:
                if (s < (r0 - delta)) or (s > (r0 + delta)):
                    continue
            else:
                i_min = int(np.ceil((s - delta - r0) * inv_dr))
                i_max = int(np.floor((s + delta - r0) * inv_dr))
                if i_min < 0:
                    i_min = 0
                if i_max >= n_ligands:
                    i_max = n_ligands - 1
                if i_max < i_min:
                    continue
            if count >= cap:
                return -1
            cand_ix[count] = np.int64(ix)
            cand_iy[count] = np.int64(iy)
            cand_iz[count] = np.int64(0)
            cand_rx[count] = rx
            cand_ry[count] = ry
            cand_rz[count] = 0.0
            count += 1
    return count


@njit(cache=True)
def _rod_candidate_cache_contains_new_pose(
    last_x: float,
    last_y: float,
    last_z: float,
    last_nx: float,
    last_ny: float,
    last_nz: float,
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    half_length: float,
    guard: float,
) -> bool:
    """
    Return whether an expanded capsule built at the old pose still contains the
    current capsule.

    Exact reuse is safe when the maximum displacement of any point on the rod is
    bounded by the extra candidate-cache guard radius. For a centered rod of
    half-length ``half_length``, that displacement is conservatively bounded by
    COM translation plus endpoint displacement from the axis rotation.
    """
    move_dx = x - last_x
    move_dy = y - last_y
    move_dz = z - last_z
    move_norm = np.sqrt(move_dx * move_dx + move_dy * move_dy + move_dz * move_dz)
    dot_n = n_x * last_nx + n_y * last_ny + n_z * last_nz
    if dot_n > 1.0:
        dot_n = 1.0
    elif dot_n < -1.0:
        dot_n = -1.0
    axis_delta_sq = 2.0 - 2.0 * dot_n
    if axis_delta_sq < 0.0:
        axis_delta_sq = 0.0
    max_point_shift = move_norm + half_length * np.sqrt(axis_delta_sq)
    return bool(max_point_shift <= guard)


@njit(cache=True)
def _build_surface_reaction_geometry(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    spacing: float,
    cutoff: float,
    max_bond_sq: float,
    r_i: np.ndarray,
    cand_ix: np.ndarray,
    cand_iy: np.ndarray,
    cand_iz: np.ndarray,
    cand_rx: np.ndarray,
    cand_ry: np.ndarray,
    cand_rz: np.ndarray,
    cand_count: int,
    cleaved_hash_ix: np.ndarray,
    cleaved_hash_iy: np.ndarray,
    cleaved_hash_iz: np.ndarray,
    cleaved_hash_used: np.ndarray,
    bound_active: np.ndarray,
    bound_ix: np.ndarray,
    bound_iy: np.ndarray,
    bound_iz: np.ndarray,
    ligand_bound: np.ndarray,
    free_ix: np.ndarray,
    free_iy: np.ndarray,
    free_iz: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    bound_slot_ix: np.ndarray,
    bound_slot_iy: np.ndarray,
    bound_slot_iz: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Build the receptor and attachment state for the current geometry.

    Free receptors are represented once with axial geometry and interacting ligand
    window bounds. Bound receptors are represented as compact force vectors.
    """
    n_ligands = len(r_i)
    cutoff_sq = cutoff * cutoff
    n_free = 0
    n_bound = 0
    bound_outside_cutoff = 0

    for lig_idx in range(n_ligands):
        ligand_bound[lig_idx] = False

    for lig_idx in range(n_ligands):
        if not bool(bound_active[lig_idx]):
            continue
        rx = float(bound_ix[lig_idx]) * spacing
        ry = float(bound_iy[lig_idx]) * spacing
        rz = float(bound_iz[lig_idx]) * spacing
        x_lig = x + r_i[lig_idx] * n_x
        y_lig = y + r_i[lig_idx] * n_y
        z_lig = z + r_i[lig_idx] * n_z
        ddx = rx - x_lig
        ddy = ry - y_lig
        ddz = rz - z_lig
        dr2 = ddx * ddx + ddy * ddy + ddz * ddz
        if dr2 > max_bond_sq:
            bound_active[lig_idx] = False
            continue
        ligand_bound[lig_idx] = True
        bound_slot_ix[n_bound] = np.int64(bound_ix[lig_idx])
        bound_slot_iy[n_bound] = np.int64(bound_iy[lig_idx])
        bound_slot_iz[n_bound] = np.int64(bound_iz[lig_idx])
        bound_ligand_idx[n_bound] = np.int32(lig_idx)
        bound_rx[n_bound] = rx
        bound_ry[n_bound] = ry
        bound_rz[n_bound] = rz
        bound_dx[n_bound] = ddx
        bound_dy[n_bound] = ddy
        bound_dz[n_bound] = ddz
        if dr2 > cutoff_sq:
            bound_outside_cutoff += 1
        n_bound += 1
    if n_ligands <= 0:
        return 0, n_bound, bound_outside_cutoff

    r0 = r_i[0]
    dr = 1.0
    inv_dr = 1.0
    if n_ligands > 1:
        dr = r_i[1] - r_i[0]
        if dr != 0.0:
            inv_dr = 1.0 / dr

    cap = len(free_ix)
    for c in range(cand_count):
        ix = int(cand_ix[c])
        iy = int(cand_iy[c])
        iz = int(cand_iz[c])
        rx = cand_rx[c]
        ry = cand_ry[c]
        rz = cand_rz[c]
        ddx = rx - x
        ddy = ry - y
        ddz = rz - z
        s = ddx * n_x + ddy * n_y + ddz * n_z
        dist_center_sq = ddx * ddx + ddy * ddy + ddz * ddz
        rho_sq = dist_center_sq - s * s
        if rho_sq < 0.0:
            rho_sq = 0.0
        if rho_sq > cutoff_sq:
            continue
        delta_sq = cutoff_sq - rho_sq
        if delta_sq < 0.0:
            continue
        delta = np.sqrt(delta_sq)

        if n_ligands == 1:
            i_min = 0
            i_max = 0
            if (s < (r0 - delta)) or (s > (r0 + delta)):
                continue
        else:
            i_min = int(np.ceil((s - delta - r0) * inv_dr))
            i_max = int(np.floor((s + delta - r0) * inv_dr))
            if i_min < 0:
                i_min = 0
            if i_max >= n_ligands:
                i_max = n_ligands - 1
            if i_max < i_min:
                continue

        if _coord_hash_contains(
            ix,
            iy,
            iz,
            cleaved_hash_ix,
            cleaved_hash_iy,
            cleaved_hash_iz,
            cleaved_hash_used,
        ):
            continue
        is_bound_coord = False
        for slot in range(n_bound):
            if (
                int(bound_slot_ix[slot]) == ix
                and int(bound_slot_iy[slot]) == iy
                and int(bound_slot_iz[slot]) == iz
            ):
                is_bound_coord = True
                break
        if is_bound_coord:
            continue

        if n_free >= cap:
            return -1, n_bound, bound_outside_cutoff
        free_ix[n_free] = np.int64(ix)
        free_iy[n_free] = np.int64(iy)
        free_iz[n_free] = np.int64(iz)
        free_s[n_free] = s
        free_rho_sq[n_free] = rho_sq
        free_i_min[n_free] = np.int32(i_min)
        free_i_max[n_free] = np.int32(i_max)
        n_free += 1

    return n_free, n_bound, bound_outside_cutoff


@njit(cache=True)
def _build_gradient_reaction_geometry(
    x: float,
    y: float,
    z: float,
    n_x: float,
    n_y: float,
    n_z: float,
    spacing: float,
    cutoff: float,
    max_bond_sq: float,
    r_i: np.ndarray,
    gradient_axis_code: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
    cand_ix: np.ndarray,
    cand_iy: np.ndarray,
    cand_iz: np.ndarray,
    cand_rx: np.ndarray,
    cand_ry: np.ndarray,
    cand_rz: np.ndarray,
    cand_count: int,
    cleaved_hash_ix: np.ndarray,
    cleaved_hash_iy: np.ndarray,
    cleaved_hash_iz: np.ndarray,
    cleaved_hash_used: np.ndarray,
    bound_active: np.ndarray,
    bound_ix: np.ndarray,
    bound_iy: np.ndarray,
    bound_iz: np.ndarray,
    ligand_bound: np.ndarray,
    free_ix: np.ndarray,
    free_iy: np.ndarray,
    free_iz: np.ndarray,
    free_s: np.ndarray,
    free_rho_sq: np.ndarray,
    free_i_min: np.ndarray,
    free_i_max: np.ndarray,
    bound_slot_ix: np.ndarray,
    bound_slot_iy: np.ndarray,
    bound_slot_iz: np.ndarray,
    bound_ligand_idx: np.ndarray,
    bound_rx: np.ndarray,
    bound_ry: np.ndarray,
    bound_rz: np.ndarray,
    bound_dx: np.ndarray,
    bound_dy: np.ndarray,
    bound_dz: np.ndarray,
) -> Tuple[int, int, int]:
    n_ligands = len(r_i)
    cutoff_sq = cutoff * cutoff
    n_free = 0
    n_bound = 0
    bound_outside_cutoff = 0

    for lig_idx in range(n_ligands):
        ligand_bound[lig_idx] = False

    for lig_idx in range(n_ligands):
        if not bool(bound_active[lig_idx]):
            continue
        rx, ry, rz = _coord_to_position_gradient_sparse(
            int(bound_ix[lig_idx]),
            int(bound_iy[lig_idx]),
            int(bound_iz[lig_idx]),
            spacing,
            gradient_axis_code,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        x_lig = x + r_i[lig_idx] * n_x
        y_lig = y + r_i[lig_idx] * n_y
        z_lig = z + r_i[lig_idx] * n_z
        ddx = rx - x_lig
        ddy = ry - y_lig
        ddz = rz - z_lig
        dr2 = ddx * ddx + ddy * ddy + ddz * ddz
        if dr2 > max_bond_sq:
            bound_active[lig_idx] = False
            continue
        ligand_bound[lig_idx] = True
        bound_slot_ix[n_bound] = np.int64(bound_ix[lig_idx])
        bound_slot_iy[n_bound] = np.int64(bound_iy[lig_idx])
        bound_slot_iz[n_bound] = np.int64(bound_iz[lig_idx])
        bound_ligand_idx[n_bound] = np.int32(lig_idx)
        bound_rx[n_bound] = rx
        bound_ry[n_bound] = ry
        bound_rz[n_bound] = rz
        bound_dx[n_bound] = ddx
        bound_dy[n_bound] = ddy
        bound_dz[n_bound] = ddz
        if dr2 > cutoff_sq:
            bound_outside_cutoff += 1
        n_bound += 1

    if n_ligands <= 0:
        return 0, n_bound, bound_outside_cutoff

    r0 = r_i[0]
    inv_dr = 1.0
    if n_ligands > 1:
        dr = r_i[1] - r_i[0]
        if dr != 0.0:
            inv_dr = 1.0 / dr

    cap = len(free_ix)
    for c in range(cand_count):
        ix = int(cand_ix[c])
        iy = int(cand_iy[c])
        iz = int(cand_iz[c])
        rx = cand_rx[c]
        ry = cand_ry[c]
        rz = cand_rz[c]
        ddx = rx - x
        ddy = ry - y
        ddz = rz - z
        s = ddx * n_x + ddy * n_y + ddz * n_z
        dist_center_sq = ddx * ddx + ddy * ddy + ddz * ddz
        rho_sq = dist_center_sq - s * s
        if rho_sq < 0.0:
            rho_sq = 0.0
        if rho_sq > cutoff_sq:
            continue
        delta_sq = cutoff_sq - rho_sq
        if delta_sq < 0.0:
            continue
        delta = np.sqrt(delta_sq)

        if n_ligands == 1:
            i_min = 0
            i_max = 0
            if (s < (r0 - delta)) or (s > (r0 + delta)):
                continue
        else:
            i_min = int(np.ceil((s - delta - r0) * inv_dr))
            i_max = int(np.floor((s + delta - r0) * inv_dr))
            if i_min < 0:
                i_min = 0
            if i_max >= n_ligands:
                i_max = n_ligands - 1
            if i_max < i_min:
                continue

        if _coord_hash_contains(
            ix,
            iy,
            iz,
            cleaved_hash_ix,
            cleaved_hash_iy,
            cleaved_hash_iz,
            cleaved_hash_used,
        ):
            continue
        is_bound_coord = False
        for slot in range(n_bound):
            if (
                int(bound_slot_ix[slot]) == ix
                and int(bound_slot_iy[slot]) == iy
                and int(bound_slot_iz[slot]) == iz
            ):
                is_bound_coord = True
                break
        if is_bound_coord:
            continue

        if n_free >= cap:
            return -1, n_bound, bound_outside_cutoff
        free_ix[n_free] = np.int64(ix)
        free_iy[n_free] = np.int64(iy)
        free_iz[n_free] = np.int64(iz)
        free_s[n_free] = s
        free_rho_sq[n_free] = rho_sq
        free_i_min[n_free] = np.int32(i_min)
        free_i_max[n_free] = np.int32(i_max)
        n_free += 1

    return n_free, n_bound, bound_outside_cutoff


def initialize_simulation_state_3d(
    config: Dict[str, Any],
    ligand_types: np.ndarray,
    grid_backend: str,
) -> Dict[str, Any]:
    """Create the initial state for one manuscript trajectory."""
    dimension = validate_dimension_config(config)
    backend = normalize_grid_backend(grid_backend)
    gradient_type = str(config.get("GRADIENT_TYPE", "uniform")).lower()
    is_uniform = gradient_type == "uniform" or np.isclose(
        float(config.get("GRADIENT_SCALE", 1.0)), 1.0
    )
    if backend == GRID_BACKEND_UNIFORM_SPARSE_COORDS and not is_uniform:
        raise ValueError("Uniform receptor coordinates require a uniform landscape")
    if backend == GRID_BACKEND_GRADIENT_SPARSE_COORDS and is_uniform:
        raise ValueError("Gradient receptor coordinates require a nonuniform landscape")

    n_ligands = max(1, int(ligand_types.shape[0]))
    working_cutoff = max(
        0.0,
        float(config.get("nearby_cutoff_alpha_mult", DEFAULT_NEARBY_CUTOFF_ALPHA_MULT)),
    )
    validation_cutoff = max(
        working_cutoff,
        float(
            config.get(
                "nearby_cutoff_validate_alpha_mult",
                DEFAULT_NEARBY_CUTOFF_VALIDATE_ALPHA_MULT,
            )
        ),
    )
    tail_tolerance = max(
        0.0, float(config.get("tail_propensity_eps", DEFAULT_TAIL_PROPENSITY_EPS))
    )
    no_nearby_policy = normalize_no_nearby_policy(
        str(config.get("no_nearby_policy", NO_NEARBY_POLICY_VALIDATED_TERMINAL))
    )

    hash_capacity = 2048
    state: Dict[str, Any] = {
        "dimension": dimension,
        "grid_backend": backend,
        "runtime_backend_effective": backend,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "n_hat": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "last_stored_x": 0.0,
        "last_stored_y": 0.0,
        "last_stored_z": 0.0,
        "last_stored_t": 0.0,
        "last_stored_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "t": 0.0,
        "attempts": 0,
        "step_counter": 0,
        "pending_event_active": False,
        "pending_tau_remaining": 0.0,
        "pending_event_kind": int(EVENT_NONE),
        "pending_ligand_idx": -1,
        "pending_receptor_ix": 0,
        "pending_receptor_iy": 0,
        "pending_receptor_iz": 0,
        "pending_n_bound": 0,
        "pending_tau_total": 0.0,
        "pending_tau_elapsed": 0.0,
        "pending_base_x": 0.0,
        "pending_base_y": 0.0,
        "pending_base_z": 0.0,
        "pending_base_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "pending_ligand_indices": np.full(n_ligands, np.int32(-1), dtype=np.int32),
        "pending_force_x": np.zeros(n_ligands, dtype=np.float64),
        "pending_force_y": np.zeros(n_ligands, dtype=np.float64),
        "pending_force_z": np.zeros(n_ligands, dtype=np.float64),
        "pending_bound_rx": np.zeros(n_ligands, dtype=np.float64),
        "pending_bound_ry": np.zeros(n_ligands, dtype=np.float64),
        "pending_bound_rz": np.zeros(n_ligands, dtype=np.float64),
        "bound_active": np.zeros(n_ligands, dtype=np.bool_),
        "bound_ix": np.zeros(n_ligands, dtype=np.int64),
        "bound_iy": np.zeros(n_ligands, dtype=np.int64),
        "bound_iz": np.zeros(n_ligands, dtype=np.int64),
        "n_bound_tracked": 0,
        "cleaved_ix": np.zeros(256, dtype=np.int64),
        "cleaved_iy": np.zeros(256, dtype=np.int64),
        "cleaved_iz": np.zeros(256, dtype=np.int64),
        "n_cleaved": 0,
        "cleaved_hash_ix": np.zeros(hash_capacity, dtype=np.int64),
        "cleaved_hash_iy": np.zeros(hash_capacity, dtype=np.int64),
        "cleaved_hash_iz": np.zeros(hash_capacity, dtype=np.int64),
        "cleaved_hash_used": np.zeros(hash_capacity, dtype=np.uint8),
        "cleaved_block_ix": np.zeros(16, dtype=np.int64),
        "cleaved_block_iy": np.zeros(16, dtype=np.int64),
        "cleaved_block_iz": np.zeros(16, dtype=np.int64),
        "cleaved_block_bits": np.zeros(16, dtype=np.uint64),
        "cleaved_block_used": np.zeros(16, dtype=np.uint8),
        "cleaved_block_count": 0,
        "cleaved_block_source_n": 0,
        "nearby_cutoff_alpha_mult": working_cutoff,
        "nearby_cutoff_validate_alpha_mult": validation_cutoff,
        "tail_propensity_eps": tail_tolerance,
        "no_nearby_policy": no_nearby_policy,
        "receptor_mobility_mode": RECEPTOR_MOBILITY_FIXED,
        "gradient_min_spacing_stop": float(
            config.get("GRADIENT_MIN_SPACING_STOP", DEFAULT_GRADIENT_MIN_SPACING_STOP)
        ),
        "gradient_escape_stop_mode": normalize_gradient_escape_stop_mode(
            str(config.get("GRADIENT_ESCAPE_STOP_MODE", GRADIENT_ESCAPE_STOP_MODE_OFF))
        ),
        "gradient_escape_dwell_time": max(
            0.0,
            float(
                config.get(
                    "GRADIENT_ESCAPE_DWELL_TIME", DEFAULT_GRADIENT_ESCAPE_DWELL_TIME
                )
            ),
        ),
        "gradient_escape_triggered": False,
        "gradient_escape_plane": 0.0,
        "gradient_escape_tail_plane": 0.0,
        "gradient_escape_axis_edge": 0.0,
        "gradient_escape_dwell_elapsed": 0.0,
        "gradient_escape_return_margin": 0.0,
        "gradient_escape_armed": False,
        "gradient_escape_origin_u": 0.0,
        "gradient_axis_law": GRADIENT_AXIS_LAW_MULTIPLICATIVE,
        "max_nearby_hint": INITIAL_NEARBY_CAPACITY,
    }
    return state


def _termination_reason_from_code(code: int) -> str:
    if int(code) == TERM_T_FINAL:
        return "t_final"
    if int(code) == TERM_GRADIENT_THRESHOLD:
        return "gradient_threshold"
    if int(code) == TERM_GRADIENT_ESCAPE:
        return "gradient_escape"
    if int(code) == TERM_NO_NEARBY:
        return "no_nearby"
    if int(code) == TERM_RESOURCE_LIMIT:
        return "resource_limit"
    return "chunk_limit"


def initialize_rng(seed: int) -> None:
    """Initialize the deterministic random stream for one trajectory."""
    seed_rng(int(seed))


def _ensure_uniform_sparse_coord_state(
    state: Dict[str, Any], ligand_types: np.ndarray
) -> None:
    n_ligands = max(1, int(ligand_types.shape[0]))
    state.setdefault("bound_active", np.zeros(n_ligands, dtype=np.bool_))
    state.setdefault("bound_ix", np.zeros(n_ligands, dtype=np.int64))
    state.setdefault("bound_iy", np.zeros(n_ligands, dtype=np.int64))
    state.setdefault("bound_iz", np.zeros(n_ligands, dtype=np.int64))
    state["bound_active"] = np.asarray(state["bound_active"], dtype=np.bool_)
    state["bound_ix"] = np.asarray(state["bound_ix"], dtype=np.int64)
    state["bound_iy"] = np.asarray(state["bound_iy"], dtype=np.int64)
    state["bound_iz"] = np.asarray(state["bound_iz"], dtype=np.int64)
    if (
        len(state["bound_active"]) != n_ligands
        or len(state["bound_ix"]) != n_ligands
        or len(state["bound_iy"]) != n_ligands
        or (len(state["bound_iz"]) != n_ligands)
    ):
        state["bound_active"] = np.zeros(n_ligands, dtype=np.bool_)
        state["bound_ix"] = np.zeros(n_ligands, dtype=np.int64)
        state["bound_iy"] = np.zeros(n_ligands, dtype=np.int64)
        state["bound_iz"] = np.zeros(n_ligands, dtype=np.int64)
    state.setdefault("cleaved_ix", np.zeros(256, dtype=np.int64))
    state.setdefault("cleaved_iy", np.zeros(256, dtype=np.int64))
    state.setdefault("cleaved_iz", np.zeros(256, dtype=np.int64))
    state["cleaved_ix"] = np.asarray(state["cleaved_ix"], dtype=np.int64)
    state["cleaved_iy"] = np.asarray(state["cleaved_iy"], dtype=np.int64)
    state["cleaved_iz"] = np.asarray(state["cleaved_iz"], dtype=np.int64)
    n_cleaved = int(state.get("n_cleaved", 0))
    if n_cleaved < 0:
        n_cleaved = 0
    n_cleaved = min(
        n_cleaved,
        len(state["cleaved_ix"]),
        len(state["cleaved_iy"]),
        len(state["cleaved_iz"]),
    )
    state["n_cleaved"] = int(n_cleaved)
    if len(state["cleaved_ix"]) < max(16, n_cleaved):
        cap = max(16, len(state["cleaved_ix"]))
        while cap < n_cleaved:
            cap = cap * 2
        new_ix = np.zeros(cap, dtype=np.int64)
        new_iy = np.zeros(cap, dtype=np.int64)
        new_iz = np.zeros(cap, dtype=np.int64)
        if n_cleaved > 0:
            new_ix[:n_cleaved] = state["cleaved_ix"][:n_cleaved]
            new_iy[:n_cleaved] = state["cleaved_iy"][:n_cleaved]
            new_iz[:n_cleaved] = state["cleaved_iz"][:n_cleaved]
        state["cleaved_ix"] = new_ix
        state["cleaved_iy"] = new_iy
        state["cleaved_iz"] = new_iz
    hash_ix = np.asarray(
        state.get("cleaved_hash_ix", np.zeros(0, dtype=np.int64)), dtype=np.int64
    )
    hash_iy = np.asarray(
        state.get("cleaved_hash_iy", np.zeros(0, dtype=np.int64)), dtype=np.int64
    )
    hash_iz = np.asarray(
        state.get("cleaved_hash_iz", np.zeros(0, dtype=np.int64)), dtype=np.int64
    )
    hash_used = np.asarray(
        state.get("cleaved_hash_used", np.zeros(0, dtype=np.uint8)), dtype=np.uint8
    )
    need_rebuild = (
        len(hash_used) < 16
        or len(hash_ix) != len(hash_used)
        or len(hash_iy) != len(hash_used)
        or (len(hash_iz) != len(hash_used))
    )
    if not need_rebuild:
        load = _count_used_slots(hash_used)
        if load < n_cleaved:
            need_rebuild = True
    if need_rebuild:
        target_cap = 16
        while target_cap < max(16, 2 * max(1, n_cleaved)):
            target_cap *= 2
        hash_ix, hash_iy, hash_iz, hash_used = _rehash_coord_table(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.uint8),
            target_cap,
        )
        for i in range(n_cleaved):
            _coord_hash_insert_no_grow(
                int(state["cleaved_ix"][i]),
                int(state["cleaved_iy"][i]),
                int(state["cleaved_iz"][i]),
                hash_ix,
                hash_iy,
                hash_iz,
                hash_used,
            )
    state["cleaved_hash_ix"] = hash_ix
    state["cleaved_hash_iy"] = hash_iy
    state["cleaved_hash_iz"] = hash_iz
    state["cleaved_hash_used"] = hash_used
    block_ix = np.asarray(
        state.get("cleaved_block_ix", np.zeros(0, dtype=np.int64)), dtype=np.int64
    )
    block_iy = np.asarray(
        state.get("cleaved_block_iy", np.zeros(0, dtype=np.int64)), dtype=np.int64
    )
    block_iz = np.asarray(
        state.get("cleaved_block_iz", np.zeros(0, dtype=np.int64)), dtype=np.int64
    )
    block_bits = np.asarray(
        state.get("cleaved_block_bits", np.zeros(0, dtype=np.uint64)), dtype=np.uint64
    )
    block_used = np.asarray(
        state.get("cleaved_block_used", np.zeros(0, dtype=np.uint8)), dtype=np.uint8
    )
    block_source_n = int(state.get("cleaved_block_source_n", -1))
    need_block_rebuild = (
        len(block_used) < 16
        or len(block_ix) != len(block_used)
        or len(block_iy) != len(block_used)
        or (len(block_iz) != len(block_used))
        or (len(block_bits) != len(block_used))
        or (block_source_n != n_cleaved)
    )
    if need_block_rebuild:
        block_ix, block_iy, block_iz, block_bits, block_used, block_count = (
            _build_cleaved_block_hash_from_coords(
                state["cleaved_ix"], state["cleaved_iy"], state["cleaved_iz"], n_cleaved
            )
        )
    else:
        block_count = int(
            state.get("cleaved_block_count", _count_used_slots(block_used))
        )
    state["cleaved_block_ix"] = block_ix
    state["cleaved_block_iy"] = block_iy
    state["cleaved_block_iz"] = block_iz
    state["cleaved_block_bits"] = block_bits
    state["cleaved_block_used"] = block_used
    state["cleaved_block_count"] = int(block_count)
    state["cleaved_block_source_n"] = int(n_cleaved)
    state.setdefault("pending_receptor_ix", 0)
    state.setdefault("pending_receptor_iy", 0)
    state.setdefault("pending_receptor_iz", 0)
    state.setdefault("uniform_domain_mode", "infinite")
    state.setdefault("runtime_backend_effective", GRID_BACKEND_UNIFORM_SPARSE_COORDS)
    state.setdefault("gradient_escape_stop_mode", GRADIENT_ESCAPE_STOP_MODE_OFF)
    state.setdefault(
        "gradient_escape_dwell_time", float(DEFAULT_GRADIENT_ESCAPE_DWELL_TIME)
    )
    state.setdefault("gradient_escape_triggered", False)
    state.setdefault("gradient_escape_plane", 0.0)
    state.setdefault("gradient_escape_tail_plane", 0.0)
    state.setdefault("gradient_escape_axis_edge", 0.0)
    state.setdefault("gradient_escape_dwell_elapsed", 0.0)
    state.setdefault("gradient_escape_return_margin", 0.0)
    state.setdefault("gradient_escape_armed", False)
    state.setdefault("gradient_escape_origin_u", 0.0)


@njit(cache=True)
def _insert_cleaved_coord_blocked(
    ix: int,
    iy: int,
    iz: int,
    cleaved_ix: np.ndarray,
    cleaved_iy: np.ndarray,
    cleaved_iz: np.ndarray,
    n_cleaved: int,
    hash_ix: np.ndarray,
    hash_iy: np.ndarray,
    hash_iz: np.ndarray,
    hash_used: np.ndarray,
    block_ix: np.ndarray,
    block_iy: np.ndarray,
    block_iz: np.ndarray,
    block_bits: np.ndarray,
    block_used: np.ndarray,
    block_count: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    bool,
]:
    if len(block_used) < 16:
        block_ix, block_iy, block_iz, block_bits, block_used, block_count = (
            _build_cleaved_block_hash_from_coords(
                cleaved_ix,
                cleaved_iy,
                cleaved_iz,
                n_cleaved,
            )
        )
    if block_count * 10 >= 7 * len(block_used):
        block_ix, block_iy, block_iz, block_bits, block_used = (
            _rehash_cleaved_block_table(
                block_ix,
                block_iy,
                block_iz,
                block_bits,
                block_used,
                max(16, len(block_used) * 2),
            )
        )

    block_inserted, new_block = _cleaved_block_hash_insert_no_grow(
        ix, iy, iz, block_ix, block_iy, block_iz, block_bits, block_used
    )
    if not block_inserted:
        return (
            cleaved_ix,
            cleaved_iy,
            cleaved_iz,
            n_cleaved,
            hash_ix,
            hash_iy,
            hash_iz,
            hash_used,
            block_ix,
            block_iy,
            block_iz,
            block_bits,
            block_used,
            block_count,
            False,
        )
    if new_block:
        block_count += 1

    used_slots = max(0, int(n_cleaved))
    if len(hash_used) < 16 or (used_slots * 10) >= (7 * len(hash_used)):
        target_cap = max(16, len(hash_used) * 2)
        hash_ix, hash_iy, hash_iz, hash_used = _rehash_coord_table(
            hash_ix, hash_iy, hash_iz, hash_used, target_cap
        )

    if n_cleaved >= len(cleaved_ix):
        new_cap = max(16, len(cleaved_ix) * 2)
        if new_cap <= n_cleaved:
            new_cap = n_cleaved + 1
        nx = np.zeros(new_cap, dtype=np.int64)
        ny = np.zeros(new_cap, dtype=np.int64)
        nz = np.zeros(new_cap, dtype=np.int64)
        if n_cleaved > 0:
            nx[:n_cleaved] = cleaved_ix[:n_cleaved]
            ny[:n_cleaved] = cleaved_iy[:n_cleaved]
            nz[:n_cleaved] = cleaved_iz[:n_cleaved]
        cleaved_ix = nx
        cleaved_iy = ny
        cleaved_iz = nz

    cleaved_ix[n_cleaved] = np.int64(ix)
    cleaved_iy[n_cleaved] = np.int64(iy)
    cleaved_iz[n_cleaved] = np.int64(iz)
    n_cleaved += 1
    _coord_hash_insert_no_grow(ix, iy, iz, hash_ix, hash_iy, hash_iz, hash_used)
    return (
        cleaved_ix,
        cleaved_iy,
        cleaved_iz,
        n_cleaved,
        hash_ix,
        hash_iy,
        hash_iz,
        hash_used,
        block_ix,
        block_iy,
        block_iz,
        block_bits,
        block_used,
        block_count,
        True,
    )


def _resolve_thermal_brownian_parameters(
    config: Dict[str, Any], alpha: float
) -> Tuple[float, float, float]:
    implied_thermal_energy = float(reversible_thermal_energy_from_alpha(float(alpha)))
    if "THERMAL_ENERGY" in config:
        thermal_energy = float(config["THERMAL_ENERGY"])
    else:
        thermal_energy = implied_thermal_energy
    if thermal_energy <= 0.0:
        raise ValueError("THERMAL_ENERGY must be positive for Brownian calculations")

    scale = max(abs(implied_thermal_energy), 1.0)
    if (
        abs(thermal_energy - implied_thermal_energy)
        > DEFAULT_THERMAL_RATE_ENERGY_RTOL * scale
    ):
        raise ValueError(
            "THERMAL_ENERGY must equal ALPHA^2 / 2 to match the reversible "
            "attachment-rate law"
        )

    dt_bd = float(
        config.get(
            "THERMAL_BROWNIAN_DT", config.get("DT_MAX", DEFAULT_THERMAL_BROWNIAN_DT)
        )
    )
    if dt_bd <= 0.0:
        dt_bd = DEFAULT_THERMAL_BROWNIAN_DT
    return thermal_energy, implied_thermal_energy, dt_bd


def _validate_thermal_brownian_friction_parameters(
    thermal_energy: float,
    gamma_parallel: float,
    gamma_perp: float,
    gamma_R: float,
    min_gamma: float,
) -> None:
    validate_thermal_inputs(thermal_energy, gamma_parallel, gamma_perp, gamma_R)
    if min_gamma < 0.0:
        raise ValueError("MIN_GAMMA must be nonnegative")
    if min_gamma > 0.0 and (
        gamma_parallel < min_gamma or gamma_perp < min_gamma or gamma_R < min_gamma
    ):
        raise ValueError(
            "MIN_GAMMA must not exceed any particle friction coefficient"
        )


def _resolve_thermal_passive_diffusion(
    config: Dict[str, Any],
    key: str,
    default_value: float,
) -> Tuple[float, bool]:
    """Return optional passive Brownian diffusivity override and whether it was set."""
    if key not in config:
        return float(default_value), False
    value = float(config[key])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and nonnegative when provided")
    return value, True


def _run_simulation_chunk_sparse_thermal(
    config: Dict[str, Any],
    ligand_types: np.ndarray,
    state: Dict[str, Any],
    max_steps: int,
    nearby_cutoff_alpha_mult: float,
) -> Dict[str, Any]:
    """Advance a 3D trajectory with Brownian motion and stochastic reactions."""
    grid_backend_mode = str(
        state.get("grid_backend", GRID_BACKEND_UNIFORM_SPARSE_COORDS)
    )
    is_gradient_sparse_runtime = (
        grid_backend_mode == GRID_BACKEND_GRADIENT_SPARSE_COORDS
    )
    if grid_backend_mode not in (
        GRID_BACKEND_UNIFORM_SPARSE_COORDS,
        GRID_BACKEND_GRADIENT_SPARSE_COORDS,
    ):
        raise ValueError(
            "Brownian calculations require uniform or gradient sparse receptor coordinates"
        )
    _ensure_uniform_sparse_coord_state(state, ligand_types)
    dimension = normalize_dimension(
        state.get("dimension", config.get("DIMENSION", DIMENSION_3D))
    )
    gradient_axis_code = -1
    pos_prefix = np.zeros(0, dtype=np.float64)
    neg_prefix = np.zeros(0, dtype=np.float64)
    pos_tail_spacing = float(config["RECEPTOR_SPACING"])
    neg_tail_spacing = float(config["RECEPTOR_SPACING"])
    gradient_dense_sign = 0
    gradient_min_spacing_stop = DEFAULT_GRADIENT_MIN_SPACING_STOP
    effective_min_spacing = float(config["RECEPTOR_SPACING"])
    if is_gradient_sparse_runtime:
        axis_spec = _build_gradient_axis_runtime_spec(config)
        gradient_axis_code = _gradient_axis_code_from_type(str(config["GRADIENT_TYPE"]))
        if gradient_axis_code < 0:
            raise ValueError(
                "gradient_sparse_coords requires GRADIENT_TYPE in {x, y, z}"
            )
        pos_prefix = np.asarray(axis_spec["pos_prefix"], dtype=np.float64)
        neg_prefix = np.asarray(axis_spec["neg_prefix"], dtype=np.float64)
        pos_tail_spacing = float(axis_spec["pos_tail_spacing"])
        neg_tail_spacing = float(axis_spec["neg_tail_spacing"])
        gradient_dense_sign = _gradient_dense_direction_sign(config)
        gradient_min_spacing_stop = float(
            state.get("gradient_min_spacing_stop", DEFAULT_GRADIENT_MIN_SPACING_STOP)
        )
        effective_min_spacing = max(
            float(DEFAULT_GRADIENT_MIN_SPACING),
            float(gradient_min_spacing_stop)
            if float(gradient_min_spacing_stop) > 0.0
            else float(DEFAULT_GRADIENT_MIN_SPACING),
        )
    n_ligands = int(ligand_types.shape[0])
    r_i = np.linspace(-float(config["L"]) / 2.0, float(config["L"]) / 2.0, n_ligands)
    alpha = float(config["ALPHA"])
    thermal_energy, _, dt_bd = _resolve_thermal_brownian_parameters(config, alpha)
    receptor_spacing = float(config["RECEPTOR_SPACING"])
    t_final = float(config["T_FINAL"])
    gamma_parallel = float(config["GAMMA_T_PARALLEL"])
    gamma_perp = float(config["GAMMA_T_PERPENDICULAR"])
    gamma_R = float(config["GAMMA_R"])
    min_gamma = float(config.get("MIN_GAMMA", 1e-06))
    _validate_thermal_brownian_friction_parameters(
        thermal_energy,
        gamma_parallel,
        gamma_perp,
        gamma_R,
        min_gamma,
    )
    gamma_parallel_eff = max(gamma_parallel, min_gamma)
    gamma_perp_eff = max(gamma_perp, min_gamma)
    gamma_R_eff = max(gamma_R, min_gamma)
    (
        reference_diffusion_parallel,
        reference_diffusion_perp,
        reference_diffusion_rot,
    ) = thermal_diffusion_constants(
        thermal_energy, gamma_parallel_eff, gamma_perp_eff, gamma_R_eff
    )
    diffusion_parallel, _ = _resolve_thermal_passive_diffusion(
        config, "THERMAL_PASSIVE_D_PARALLEL", reference_diffusion_parallel
    )
    diffusion_perp, _ = _resolve_thermal_passive_diffusion(
        config, "THERMAL_PASSIVE_D_PERP", reference_diffusion_perp
    )
    diffusion_rot, _ = _resolve_thermal_passive_diffusion(
        config, "THERMAL_PASSIVE_D_ROT", reference_diffusion_rot
    )
    diffusion_iso = (float(diffusion_parallel) + 2.0 * float(diffusion_perp)) / 3.0
    free_translation_msd_factor = 6.0
    adaptive_dt_min = float(config.get("THERMAL_BROWNIAN_DT_MIN", dt_bd))
    adaptive_dt_max = float(config.get("THERMAL_BROWNIAN_DT_MAX", dt_bd))
    if adaptive_dt_min <= 0.0:
        adaptive_dt_min = dt_bd
    if adaptive_dt_max <= 0.0:
        adaptive_dt_max = dt_bd
    if adaptive_dt_max < adaptive_dt_min:
        adaptive_dt_min, adaptive_dt_max = (adaptive_dt_max, adaptive_dt_min)
    adaptive_dt_target_hazard = max(
        0.0, float(config.get("THERMAL_ADAPTIVE_DT_TARGET_HAZARD", 0.08))
    )
    adaptive_dt_free_rms_over_alpha = max(
        0.0, float(config.get("THERMAL_ADAPTIVE_DT_FREE_RMS_OVER_ALPHA", 0.35))
    )
    adaptive_dt_free_rot_rms_rad = max(
        0.0, float(config.get("THERMAL_ADAPTIVE_DT_FREE_ROT_RMS_RAD", 0.25))
    )
    adaptive_dt_bound_multiplier = max(
        0.0, float(config.get("THERMAL_ADAPTIVE_DT_BOUND_MULTIPLIER", 1.0))
    )
    adaptive_dt_free_motion_cap = 1e300
    if adaptive_dt_free_rms_over_alpha > 0.0 and diffusion_iso > 0.0:
        adaptive_dt_free_motion_cap = min(
            adaptive_dt_free_motion_cap,
            (adaptive_dt_free_rms_over_alpha * alpha) ** 2
            / (free_translation_msd_factor * diffusion_iso),
        )
    if adaptive_dt_free_rot_rms_rad > 0.0 and diffusion_rot > 0.0:
        adaptive_dt_free_motion_cap = min(
            adaptive_dt_free_motion_cap,
            adaptive_dt_free_rot_rms_rad**2 / (2.0 * diffusion_rot),
        )
    k_d = float(config["K_D"])
    k_c = float(config["K_C"])
    cutoff = max(0.0, float(nearby_cutoff_alpha_mult)) * alpha
    thermal_geometry_cache_guard = (
        max(0.0, float(config.get("THERMAL_GEOMETRY_CACHE_GUARD_ALPHA", 0.0))) * alpha
    )
    thermal_geometry_cache_enabled = bool(thermal_geometry_cache_guard > 0.0)
    max_bond_sq = np.inf
    x = float(state.get("x", 0.0))
    y = float(state.get("y", 0.0))
    z = float(state.get("z", 0.0))
    n_hat = np.asarray(
        state.get("n_hat", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        dtype=np.float64,
    )
    n_norm = float(np.linalg.norm(n_hat))
    if n_norm <= 1e-12:
        n_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n_hat = n_hat / n_norm
    t = float(state.get("t", 0.0))
    attempts = int(state.get("attempts", 0))
    step_counter = int(state.get("step_counter", 0))
    gradient_escape_stop_mode = normalize_gradient_escape_stop_mode(
        str(
            state.get(
                "gradient_escape_stop_mode",
                config.get("GRADIENT_ESCAPE_STOP_MODE", GRADIENT_ESCAPE_STOP_MODE_OFF),
            )
        )
    )
    gradient_escape_dwell_time = max(
        0.0,
        float(
            state.get(
                "gradient_escape_dwell_time",
                config.get(
                    "GRADIENT_ESCAPE_DWELL_TIME", DEFAULT_GRADIENT_ESCAPE_DWELL_TIME
                ),
            )
        ),
    )
    gradient_escape_triggered = bool(state.get("gradient_escape_triggered", False))
    gradient_escape_dwell_elapsed = float(
        state.get("gradient_escape_dwell_elapsed", 0.0)
    )
    gradient_escape_armed = bool(state.get("gradient_escape_armed", False))
    gradient_escape_origin_u = 0.0
    gradient_escape_enabled = False
    gradient_escape_plane = 0.0
    gradient_escape_tail_plane = 0.0
    gradient_escape_return_margin = 0.0
    gradient_escape_axis_edge = float(state.get("gradient_escape_axis_edge", 0.0))
    gradient_escape_eval_t = float(t)
    gradient_stop_spacing_min_seen = float(
        state.get("gradient_stop_spacing_observed", 0.0)
    )
    if gradient_stop_spacing_min_seen <= 0.0:
        gradient_stop_spacing_min_seen = 1e18
    if is_gradient_sparse_runtime:
        gradient_escape_origin_u = float(
            state.get(
                "gradient_escape_origin_u",
                dense_coordinate_from_axis_position(
                    _gradient_axis_value_for_state(x, y, z, gradient_axis_code),
                    gradient_dense_sign,
                ),
            )
        )
        (
            gradient_escape_enabled,
            gradient_escape_plane,
            gradient_escape_tail_plane,
            _,
        ) = _build_gradient_escape_geometry(
            config,
            gradient_axis_code,
            gradient_dense_sign,
            float(gradient_escape_origin_u) / float(gradient_dense_sign)
            if gradient_dense_sign != 0
            else 0.0,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        gradient_escape_return_margin = 0.5 * float(config["L"]) + float(cutoff)
    last_stored_x = float(state.get("last_stored_x", x))
    last_stored_y = float(state.get("last_stored_y", y))
    last_stored_z = float(state.get("last_stored_z", z))
    last_stored_t = float(state.get("last_stored_t", t))
    last_stored_n = np.asarray(
        state.get("last_stored_n", n_hat.copy()), dtype=np.float64
    )
    time_threshold = _resolve_trajectory_record_interval(config)
    dense_backfill_records = (
        bool(config.get("DISABLE_TRAJECTORY_COMPRESSION", False))
        and time_threshold > 0.0
    )
    displacement_threshold = 0.5 * float(config["L"])
    angle_threshold = 0.25
    bound_active = np.asarray(state["bound_active"], dtype=np.bool_)
    bound_ix = np.asarray(state["bound_ix"], dtype=np.int64)
    bound_iy = np.asarray(state["bound_iy"], dtype=np.int64)
    bound_iz = np.asarray(state["bound_iz"], dtype=np.int64)
    cleaved_ix = np.asarray(state["cleaved_ix"], dtype=np.int64)
    cleaved_iy = np.asarray(state["cleaved_iy"], dtype=np.int64)
    cleaved_iz = np.asarray(state["cleaved_iz"], dtype=np.int64)
    n_cleaved = int(state.get("n_cleaved", 0))
    hash_ix = np.asarray(state["cleaved_hash_ix"], dtype=np.int64)
    hash_iy = np.asarray(state["cleaved_hash_iy"], dtype=np.int64)
    hash_iz = np.asarray(state["cleaved_hash_iz"], dtype=np.int64)
    hash_used = np.asarray(state["cleaved_hash_used"], dtype=np.uint8)
    block_ix = np.asarray(state["cleaved_block_ix"], dtype=np.int64)
    block_iy = np.asarray(state["cleaved_block_iy"], dtype=np.int64)
    block_iz = np.asarray(state["cleaved_block_iz"], dtype=np.int64)
    block_bits = np.asarray(state["cleaved_block_bits"], dtype=np.uint64)
    block_used = np.asarray(state["cleaved_block_used"], dtype=np.uint8)
    block_count = int(state.get("cleaved_block_count", _count_used_slots(block_used)))
    max_nearby = int(
        max(
            INITIAL_NEARBY_CAPACITY,
            state.get("max_nearby_hint", INITIAL_NEARBY_CAPACITY),
            _estimate_candidate_capacity(
                x,
                y,
                z,
                float(n_hat[0]),
                float(n_hat[1]),
                float(n_hat[2]),
                float(config["L"]),
                effective_min_spacing
                if is_gradient_sparse_runtime
                else receptor_spacing,
                cutoff,
            ),
        )
    )
    free_ix = np.empty(max_nearby, dtype=np.int64)
    free_iy = np.empty(max_nearby, dtype=np.int64)
    free_iz = np.empty(max_nearby, dtype=np.int64)
    free_s = np.empty(max_nearby, dtype=np.float64)
    free_rho_sq = np.empty(max_nearby, dtype=np.float64)
    free_i_min = np.empty(max_nearby, dtype=np.int32)
    free_i_max = np.empty(max_nearby, dtype=np.int32)
    bind_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
    cleave_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
    ligand_bound_buffer = np.zeros(max(1, n_ligands), dtype=np.bool_)
    bound_slot_ix = np.empty(max(1, n_ligands), dtype=np.int64)
    bound_slot_iy = np.empty(max(1, n_ligands), dtype=np.int64)
    bound_slot_iz = np.empty(max(1, n_ligands), dtype=np.int64)
    bound_slot_ligand_idx = np.empty(max(1, n_ligands), dtype=np.int32)
    bound_rx = np.empty(max(1, n_ligands), dtype=np.float64)
    bound_ry = np.empty(max(1, n_ligands), dtype=np.float64)
    bound_rz = np.empty(max(1, n_ligands), dtype=np.float64)
    bound_dx = np.empty(max(1, n_ligands), dtype=np.float64)
    bound_dy = np.empty(max(1, n_ligands), dtype=np.float64)
    bound_dz = np.empty(max(1, n_ligands), dtype=np.float64)
    unbind_rate_by_bound_slot = np.zeros(max(1, n_ligands), dtype=np.float64)
    cand_ix = np.empty(max_nearby, dtype=np.int64)
    cand_iy = np.empty(max_nearby, dtype=np.int64)
    cand_iz = np.empty(max_nearby, dtype=np.int64)
    cand_rx = np.empty(max_nearby, dtype=np.float64)
    cand_ry = np.empty(max_nearby, dtype=np.float64)
    cand_rz = np.empty(max_nearby, dtype=np.float64)
    thermal_geometry_cache_valid = False
    thermal_geometry_cache_count = 0
    thermal_geometry_cache_last_x = float(x)
    thermal_geometry_cache_last_y = float(y)
    thermal_geometry_cache_last_z = float(z)
    thermal_geometry_cache_last_nx = float(n_hat[0])
    thermal_geometry_cache_last_ny = float(n_hat[1])
    thermal_geometry_cache_last_nz = float(n_hat[2])
    thermal_geometry_cache_half_length = 0.5 * float(config["L"])
    times_list: list[float] = []
    positions_list: list[np.ndarray] = []
    if abs(t) <= 1e-12:
        times_list.append(0.0)
        positions_list.append(
            build_position_row_for_dimension(x, y, z, n_hat, dimension)
        )
        last_stored_t = 0.0
    reaction_steps = 0
    bind_events = 0
    unbind_events = 0
    cleavage_events = 0
    done = False
    termination_code = TERM_CHUNK_LIMIT

    def _grow_free_buffers() -> None:
        nonlocal max_nearby, free_ix, free_iy, free_iz, free_s, free_rho_sq
        nonlocal \
            free_i_min, \
            free_i_max, \
            bind_sum_by_free_receptor, \
            cleave_sum_by_free_receptor
        nonlocal cand_ix, cand_iy, cand_iz, cand_rx, cand_ry, cand_rz
        nonlocal thermal_geometry_cache_valid, thermal_geometry_cache_count
        max_nearby = max(64, int(max_nearby) * 2)
        free_ix = np.empty(max_nearby, dtype=np.int64)
        free_iy = np.empty(max_nearby, dtype=np.int64)
        free_iz = np.empty(max_nearby, dtype=np.int64)
        free_s = np.empty(max_nearby, dtype=np.float64)
        free_rho_sq = np.empty(max_nearby, dtype=np.float64)
        free_i_min = np.empty(max_nearby, dtype=np.int32)
        free_i_max = np.empty(max_nearby, dtype=np.int32)
        bind_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
        cleave_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
        cand_ix = np.empty(max_nearby, dtype=np.int64)
        cand_iy = np.empty(max_nearby, dtype=np.int64)
        cand_iz = np.empty(max_nearby, dtype=np.int64)
        cand_rx = np.empty(max_nearby, dtype=np.float64)
        cand_ry = np.empty(max_nearby, dtype=np.float64)
        cand_rz = np.empty(max_nearby, dtype=np.float64)
        thermal_geometry_cache_valid = False
        thermal_geometry_cache_count = 0

    def _build_current_geometry() -> Tuple[int, int, int, int]:
        nonlocal thermal_geometry_cache_valid, thermal_geometry_cache_count
        nonlocal \
            thermal_geometry_cache_last_x, \
            thermal_geometry_cache_last_y, \
            thermal_geometry_cache_last_z
        nonlocal \
            thermal_geometry_cache_last_nx, \
            thermal_geometry_cache_last_ny, \
            thermal_geometry_cache_last_nz
        while True:
            if is_gradient_sparse_runtime:
                can_use_candidate_cache = bool(thermal_geometry_cache_enabled)
                if can_use_candidate_cache and bool(thermal_geometry_cache_valid):
                    if not _rod_candidate_cache_contains_new_pose(
                        thermal_geometry_cache_last_x,
                        thermal_geometry_cache_last_y,
                        thermal_geometry_cache_last_z,
                        thermal_geometry_cache_last_nx,
                        thermal_geometry_cache_last_ny,
                        thermal_geometry_cache_last_nz,
                        x,
                        y,
                        z,
                        float(n_hat[0]),
                        float(n_hat[1]),
                        float(n_hat[2]),
                        thermal_geometry_cache_half_length,
                        thermal_geometry_cache_guard,
                    ):
                        thermal_geometry_cache_valid = False
                        thermal_geometry_cache_count = 0
                if can_use_candidate_cache and bool(thermal_geometry_cache_valid):
                    cand_count_local = int(thermal_geometry_cache_count)
                else:
                    candidate_cutoff = cutoff
                    if can_use_candidate_cache:
                        candidate_cutoff = cutoff + float(thermal_geometry_cache_guard)
                    cand_count_local = _build_candidate_cache_gradient_sparse_geometry(
                        x,
                        y,
                        z,
                        float(n_hat[0]),
                        float(n_hat[1]),
                        float(n_hat[2]),
                        float(config["L"]),
                        receptor_spacing,
                        r_i,
                        candidate_cutoff,
                        gradient_axis_code,
                        pos_prefix,
                        neg_prefix,
                        pos_tail_spacing,
                        neg_tail_spacing,
                        hash_ix,
                        hash_iy,
                        hash_iz,
                        hash_used,
                        cand_ix,
                        cand_iy,
                        cand_iz,
                        cand_rx,
                        cand_ry,
                        cand_rz,
                    )
                    if cand_count_local >= 0 and can_use_candidate_cache:
                        thermal_geometry_cache_count = int(cand_count_local)
                        thermal_geometry_cache_last_x = float(x)
                        thermal_geometry_cache_last_y = float(y)
                        thermal_geometry_cache_last_z = float(z)
                        thermal_geometry_cache_last_nx = float(n_hat[0])
                        thermal_geometry_cache_last_ny = float(n_hat[1])
                        thermal_geometry_cache_last_nz = float(n_hat[2])
                        thermal_geometry_cache_valid = True
                if cand_count_local >= 0:
                    n_free_local, n_bound_local, bound_outside_local = (
                        _build_gradient_reaction_geometry(
                            x,
                            y,
                            z,
                            float(n_hat[0]),
                            float(n_hat[1]),
                            float(n_hat[2]),
                            receptor_spacing,
                            cutoff,
                            max_bond_sq,
                            r_i,
                            gradient_axis_code,
                            pos_prefix,
                            neg_prefix,
                            pos_tail_spacing,
                            neg_tail_spacing,
                            cand_ix,
                            cand_iy,
                            cand_iz,
                            cand_rx,
                            cand_ry,
                            cand_rz,
                            int(cand_count_local),
                            hash_ix,
                            hash_iy,
                            hash_iz,
                            hash_used,
                            bound_active,
                            bound_ix,
                            bound_iy,
                            bound_iz,
                            ligand_bound_buffer,
                            free_ix,
                            free_iy,
                            free_iz,
                            free_s,
                            free_rho_sq,
                            free_i_min,
                            free_i_max,
                            bound_slot_ix,
                            bound_slot_iy,
                            bound_slot_iz,
                            bound_slot_ligand_idx,
                            bound_rx,
                            bound_ry,
                            bound_rz,
                            bound_dx,
                            bound_dy,
                            bound_dz,
                        )
                    )
                else:
                    n_free_local = -1
                    n_bound_local = 0
                    bound_outside_local = 0
            else:
                cand_count_local, n_free_local, n_bound_local, bound_outside_local = (
                    _build_uniform_reaction_geometry(
                        x,
                        y,
                        z,
                        float(n_hat[0]),
                        float(n_hat[1]),
                        float(n_hat[2]),
                        float(config["L"]),
                        receptor_spacing,
                        cutoff,
                        max_bond_sq,
                        r_i,
                        hash_ix,
                        hash_iy,
                        hash_iz,
                        hash_used,
                        block_ix,
                        block_iy,
                        block_iz,
                        block_bits,
                        block_used,
                        bound_active,
                        bound_ix,
                        bound_iy,
                        bound_iz,
                        ligand_bound_buffer,
                        free_ix,
                        free_iy,
                        free_iz,
                        free_s,
                        free_rho_sq,
                        free_i_min,
                        free_i_max,
                        bound_slot_ix,
                        bound_slot_iy,
                        bound_slot_iz,
                        bound_slot_ligand_idx,
                        bound_rx,
                        bound_ry,
                        bound_rz,
                        bound_dx,
                        bound_dy,
                        bound_dz,
                    )
                )
            if cand_count_local >= 0 and n_free_local >= 0:
                break
            _grow_free_buffers()
        return (
            int(cand_count_local),
            int(n_free_local),
            int(n_bound_local),
            int(bound_outside_local),
        )

    def _advance_brownian(dt: float, n_bound_local: int) -> None:
        nonlocal x, y, z, n_hat
        if dt <= 0.0:
            return
        if int(n_bound_local) <= 0:
            x_new, y_new, z_new, nx_new, ny_new, nz_new = brownian_kick_free_rod_3d(
                x,
                y,
                z,
                float(n_hat[0]),
                float(n_hat[1]),
                float(n_hat[2]),
                float(dt),
                diffusion_parallel,
                diffusion_perp,
                diffusion_rot,
            )
        else:
            x_new, y_new, z_new, nx_new, ny_new, nz_new = (
                brownian_dynamics_step_bound_rod_relaxation_ou_3d(
                    x,
                    y,
                    z,
                    float(n_hat[0]),
                    float(n_hat[1]),
                    float(n_hat[2]),
                    float(dt),
                    r_i,
                    bound_slot_ligand_idx[:n_bound_local],
                    bound_rx[:n_bound_local],
                    bound_ry[:n_bound_local],
                    bound_rz[:n_bound_local],
                    thermal_energy,
                    gamma_parallel,
                    gamma_perp,
                    gamma_R,
                    min_gamma,
                    diffusion_parallel,
                    diffusion_perp,
                    diffusion_rot,
                )
            )
        x = float(x_new)
        y = float(y_new)
        z = float(z_new)
        n_hat[0] = float(nx_new)
        n_hat[1] = float(ny_new)
        n_hat[2] = float(nz_new)

    def _maybe_record_pose() -> None:
        nonlocal \
            last_stored_x, \
            last_stored_y, \
            last_stored_z, \
            last_stored_t, \
            last_stored_n
        displacement = np.sqrt(
            (x - last_stored_x) ** 2
            + (y - last_stored_y) ** 2
            + (z - last_stored_z) ** 2
        )
        dot_product = float(
            n_hat[0] * last_stored_n[0]
            + n_hat[1] * last_stored_n[1]
            + n_hat[2] * last_stored_n[2]
        )
        dot_product = min(1.0, max(-1.0, dot_product))
        angle_change = np.arccos(dot_product)
        if (
            displacement > displacement_threshold
            or angle_change > angle_threshold
            or t - last_stored_t >= time_threshold
        ):
            (
                last_stored_x,
                last_stored_y,
                last_stored_z,
                last_stored_t,
                last_stored_n,
            ) = _append_trajectory_samples_with_optional_backfill(
                times_list,
                positions_list,
                dimension=dimension,
                last_t=last_stored_t,
                last_x=last_stored_x,
                last_y=last_stored_y,
                last_z=last_stored_z,
                last_n=last_stored_n,
                t=t,
                x=x,
                y=y,
                z=z,
                n_hat=n_hat,
                time_threshold=time_threshold,
                dense_backfill=dense_backfill_records,
            )

    def _evaluate_gradient_stop_and_escape() -> bool:
        nonlocal done, termination_code, gradient_stop_spacing_min_seen
        nonlocal gradient_escape_eval_t, gradient_escape_axis_edge
        nonlocal \
            gradient_escape_armed, \
            gradient_escape_dwell_elapsed, \
            gradient_escape_triggered
        if not is_gradient_sparse_runtime:
            return False
        axis_min, axis_max, spacing_observed = _compute_gradient_stop_interval_metrics(
            x,
            y,
            z,
            float(n_hat[0]),
            float(n_hat[1]),
            float(n_hat[2]),
            float(config["L"]),
            float(cutoff),
            gradient_axis_code,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        if spacing_observed < gradient_stop_spacing_min_seen:
            gradient_stop_spacing_min_seen = float(spacing_observed)
        state["gradient_stop_axis_min"] = float(axis_min)
        state["gradient_stop_axis_max"] = float(axis_max)
        state["gradient_stop_spacing_observed"] = float(spacing_observed)
        if (
            gradient_min_spacing_stop > 0.0
            and spacing_observed < gradient_min_spacing_stop
        ):
            state["gradient_stop_triggered"] = True
            termination_code = TERM_GRADIENT_THRESHOLD
            done = True
            return True
        if gradient_escape_enabled:
            dt_escape_eval = float(t - gradient_escape_eval_t)
            gradient_escape_eval_t = float(t)
            _u_sparse_edge, gradient_escape_axis_edge = (
                dense_interval_from_axis_interval(
                    float(axis_min), float(axis_max), int(gradient_dense_sign)
                )
            )
            state["gradient_escape_axis_edge"] = float(gradient_escape_axis_edge)
            (
                gradient_escape_armed,
                gradient_escape_dwell_elapsed,
                gradient_escape_triggered,
            ) = _advance_gradient_escape_state(
                float(gradient_escape_axis_edge),
                float(dt_escape_eval),
                float(gradient_escape_plane),
                float(gradient_escape_return_margin),
                float(gradient_escape_dwell_time),
                bool(gradient_escape_armed),
                float(gradient_escape_dwell_elapsed),
            )
            if gradient_escape_triggered:
                state["gradient_escape_triggered"] = True
                termination_code = TERM_GRADIENT_ESCAPE
                done = True
                return True
        return False

    def _adaptive_brownian_window(total_propensity: float, n_bound_local: int) -> float:
        dt_candidate = float(adaptive_dt_max)
        if adaptive_dt_target_hazard > 0.0 and total_propensity > 0.0:
            hazard_dt = float(adaptive_dt_target_hazard) / max(
                float(total_propensity), 1e-300
            )
            if hazard_dt < dt_candidate:
                dt_candidate = hazard_dt
        if int(n_bound_local) <= 0:
            if adaptive_dt_free_motion_cap < dt_candidate:
                dt_candidate = float(adaptive_dt_free_motion_cap)
        elif adaptive_dt_bound_multiplier > 0.0:
            bound_dt = float(dt_bd) * float(adaptive_dt_bound_multiplier)
            if bound_dt < dt_candidate:
                dt_candidate = bound_dt
        if dt_candidate < adaptive_dt_min:
            dt_candidate = float(adaptive_dt_min)
        if dt_candidate > adaptive_dt_max:
            dt_candidate = float(adaptive_dt_max)
        if dt_candidate <= 0.0 or not np.isfinite(dt_candidate):
            dt_candidate = float(dt_bd)
        return float(dt_candidate)

    max_reactions_per_brownian_step = max(
        1, int(config.get("THERMAL_MAX_REACTIONS_PER_BROWNIAN_STEP", 100000))
    )
    local_steps = 0
    while local_steps < max(1, int(max_steps)) and t < t_final - 1e-15:
        step_end_t = float(t_final)
        reactions_this_brownian_step = 0
        while t < step_end_t - 1e-15:
            if _evaluate_gradient_stop_and_escape():
                break
            _, n_free, n_bound, _ = _build_current_geometry()
            total_propensity, _, _, _, _ = build_reaction_rate_tables(
                n_free,
                free_s,
                free_rho_sq,
                free_i_min,
                free_i_max,
                n_bound,
                bound_slot_ligand_idx,
                bound_dx,
                bound_dy,
                bound_dz,
                r_i,
                alpha,
                k_d,
                k_c,
                ligand_types,
                ligand_bound_buffer,
                bind_sum_by_free_receptor,
                cleave_sum_by_free_receptor,
                unbind_rate_by_bound_slot,
            )
            remaining_dt = max(
                0.0,
                min(
                    float(step_end_t - t),
                    float(_adaptive_brownian_window(total_propensity, n_bound)),
                ),
            )
            reaction_occurred = False
            tau = 1e300
            event_kind = int(EVENT_NONE)
            event_slot_idx = -1
            event_ligand_idx = -1
            n_bound_event = n_bound
            wait_propensity = float(total_propensity)
            if wait_propensity > 0.0:
                clock_active, tau = sample_reaction_wait_time_rng(wait_propensity)
            apply_event = bool(
                wait_propensity > 0.0 and bool(clock_active) and (tau <= remaining_dt)
            )
            tau_before = float(tau if apply_event else remaining_dt)
            tau_before = min(max(tau_before, 0.0), remaining_dt)
            _advance_brownian(tau_before, n_bound)
            t += tau_before
            if _evaluate_gradient_stop_and_escape():
                break
            if apply_event:
                if is_gradient_sparse_runtime:
                    _valid_free_event, n_bound_event, _bound_outside_event = (
                        _refresh_gradient_reaction_geometry(
                            x,
                            y,
                            z,
                            float(n_hat[0]),
                            float(n_hat[1]),
                            float(n_hat[2]),
                            receptor_spacing,
                            cutoff,
                            max_bond_sq,
                            r_i,
                            gradient_axis_code,
                            pos_prefix,
                            neg_prefix,
                            pos_tail_spacing,
                            neg_tail_spacing,
                            bound_active,
                            bound_ix,
                            bound_iy,
                            bound_iz,
                            ligand_bound_buffer,
                            n_free,
                            free_ix,
                            free_iy,
                            free_iz,
                            free_s,
                            free_rho_sq,
                            free_i_min,
                            free_i_max,
                            bound_slot_ix,
                            bound_slot_iy,
                            bound_slot_iz,
                            bound_slot_ligand_idx,
                            bound_rx,
                            bound_ry,
                            bound_rz,
                            bound_dx,
                            bound_dy,
                            bound_dz,
                        )
                    )
                else:
                    _valid_free_event, n_bound_event, _bound_outside_event = (
                        _refresh_uniform_reaction_geometry(
                            x,
                            y,
                            z,
                            float(n_hat[0]),
                            float(n_hat[1]),
                            float(n_hat[2]),
                            receptor_spacing,
                            cutoff,
                            max_bond_sq,
                            r_i,
                            bound_active,
                            bound_ix,
                            bound_iy,
                            bound_iz,
                            ligand_bound_buffer,
                            n_free,
                            free_ix,
                            free_iy,
                            free_iz,
                            free_s,
                            free_rho_sq,
                            free_i_min,
                            free_i_max,
                            bound_slot_ix,
                            bound_slot_iy,
                            bound_slot_iz,
                            bound_slot_ligand_idx,
                            bound_rx,
                            bound_ry,
                            bound_rz,
                            bound_dx,
                            bound_dy,
                            bound_dz,
                        )
                    )
                n_free_event = n_free
                (
                    total_propensity_event,
                    _bind_top_event,
                    _unbind_top_event,
                    _cleave_top_event,
                    _window_len_total_event,
                ) = build_reaction_rate_tables(
                    n_free_event,
                    free_s,
                    free_rho_sq,
                    free_i_min,
                    free_i_max,
                    n_bound_event,
                    bound_slot_ligand_idx,
                    bound_dx,
                    bound_dy,
                    bound_dz,
                    r_i,
                    alpha,
                    k_d,
                    k_c,
                    ligand_types,
                    ligand_bound_buffer,
                    bind_sum_by_free_receptor,
                    cleave_sum_by_free_receptor,
                    unbind_rate_by_bound_slot,
                )
                if apply_event and total_propensity_event > 0.0:
                    (
                        reaction_occurred,
                        event_kind,
                        event_slot_idx,
                        event_ligand_idx,
                    ) = sample_reaction_channel(
                        n_free_event,
                        bind_sum_by_free_receptor,
                        cleave_sum_by_free_receptor,
                        free_s,
                        free_rho_sq,
                        free_i_min,
                        free_i_max,
                        n_bound_event,
                        unbind_rate_by_bound_slot,
                        bound_slot_ligand_idx,
                        r_i,
                        alpha,
                        k_d,
                        k_c,
                        ligand_types,
                        ligand_bound_buffer,
                        total_propensity_event,
                    )
                    apply_event = bool(reaction_occurred and event_slot_idx >= 0)
                else:
                    reaction_occurred = False
                    apply_event = False
            reaction_steps += 1
            attempts += 1
            step_counter += 1
            if not apply_event:
                break
            reactions_this_brownian_step += 1
            if int(event_kind) == int(EVENT_BIND):
                rec_ix = int(free_ix[event_slot_idx])
                rec_iy = int(free_iy[event_slot_idx])
                rec_iz = int(free_iz[event_slot_idx])
                if 0 <= event_ligand_idx < n_ligands and (
                    not bool(bound_active[event_ligand_idx])
                ):
                    bound_active[event_ligand_idx] = True
                    bound_ix[event_ligand_idx] = np.int64(rec_ix)
                    bound_iy[event_ligand_idx] = np.int64(rec_iy)
                    bound_iz[event_ligand_idx] = np.int64(rec_iz)
                    bind_events += 1
            elif int(event_kind) == int(EVENT_UNBIND):
                if 0 <= event_slot_idx < n_bound_event:
                    unbind_lig = int(bound_slot_ligand_idx[event_slot_idx])
                    if 0 <= unbind_lig < n_ligands and bool(bound_active[unbind_lig]):
                        bound_active[unbind_lig] = False
                    unbind_events += 1
            elif int(event_kind) == int(EVENT_CLEAVE):
                rec_ix = int(free_ix[event_slot_idx])
                rec_iy = int(free_iy[event_slot_idx])
                rec_iz = int(free_iz[event_slot_idx])
                (
                    cleaved_ix,
                    cleaved_iy,
                    cleaved_iz,
                    n_cleaved,
                    hash_ix,
                    hash_iy,
                    hash_iz,
                    hash_used,
                    block_ix,
                    block_iy,
                    block_iz,
                    block_bits,
                    block_used,
                    block_count,
                    inserted,
                ) = _insert_cleaved_coord_blocked(
                    rec_ix,
                    rec_iy,
                    rec_iz,
                    cleaved_ix,
                    cleaved_iy,
                    cleaved_iz,
                    n_cleaved,
                    hash_ix,
                    hash_iy,
                    hash_iz,
                    hash_used,
                    block_ix,
                    block_iy,
                    block_iz,
                    block_bits,
                    block_used,
                    block_count,
                )
                if bool(inserted):
                    cleavage_events += 1
            _maybe_record_pose()
            if reactions_this_brownian_step >= max_reactions_per_brownian_step:
                break
        if done:
            break
        local_steps += 1
        _maybe_record_pose()
    if t >= t_final - 1e-12:
        done = True
        termination_code = TERM_T_FINAL
    if len(times_list) == 0 or abs(float(times_list[-1]) - float(t)) > 1e-10:
        last_stored_x, last_stored_y, last_stored_z, last_stored_t, last_stored_n = (
            _append_trajectory_samples_with_optional_backfill(
                times_list,
                positions_list,
                dimension=dimension,
                last_t=last_stored_t,
                last_x=last_stored_x,
                last_y=last_stored_y,
                last_z=last_stored_z,
                last_n=last_stored_n,
                t=t,
                x=x,
                y=y,
                z=z,
                n_hat=n_hat,
                time_threshold=time_threshold,
                dense_backfill=dense_backfill_records,
            )
        )
    times_chunk = (
        np.asarray(times_list, dtype=np.float64)
        if len(times_list) > 0
        else np.zeros(0, dtype=np.float64)
    )
    positions_chunk = (
        np.asarray(positions_list, dtype=np.float64)
        if len(positions_list) > 0
        else np.zeros((0, 6), dtype=np.float64)
    )
    state["x"] = float(x)
    state["y"] = float(y)
    state["z"] = float(z)
    state["n_hat"] = np.asarray(n_hat, dtype=np.float64)
    state["last_stored_x"] = float(last_stored_x)
    state["last_stored_y"] = float(last_stored_y)
    state["last_stored_z"] = float(last_stored_z)
    state["last_stored_t"] = float(last_stored_t)
    state["last_stored_n"] = np.asarray(last_stored_n, dtype=np.float64)
    state["t"] = float(t)
    state["attempts"] = int(attempts)
    state["step_counter"] = int(step_counter)
    state["bound_active"] = bound_active
    state["bound_ix"] = bound_ix
    state["bound_iy"] = bound_iy
    state["bound_iz"] = bound_iz
    state["n_bound_tracked"] = int(_count_bound_active(bound_active))
    state["cleaved_ix"] = cleaved_ix
    state["cleaved_iy"] = cleaved_iy
    state["cleaved_iz"] = cleaved_iz
    state["n_cleaved"] = int(n_cleaved)
    state["cleaved_hash_ix"] = hash_ix
    state["cleaved_hash_iy"] = hash_iy
    state["cleaved_hash_iz"] = hash_iz
    state["cleaved_hash_used"] = hash_used
    state["cleaved_block_ix"] = block_ix
    state["cleaved_block_iy"] = block_iy
    state["cleaved_block_iz"] = block_iz
    state["cleaved_block_bits"] = block_bits
    state["cleaved_block_used"] = block_used
    state["cleaved_block_count"] = int(block_count)
    state["cleaved_block_source_n"] = int(n_cleaved)
    runtime_backend_effective = (
        GRID_BACKEND_GRADIENT_SPARSE_COORDS
        if is_gradient_sparse_runtime
        else GRID_BACKEND_UNIFORM_SPARSE_COORDS
    )
    state["runtime_backend_effective"] = runtime_backend_effective
    state["motion_rule"] = MOTION_RULE_BROWNIAN
    if is_gradient_sparse_runtime:
        state["gradient_min_spacing_stop"] = float(gradient_min_spacing_stop)
        state["gradient_stop_triggered"] = bool(
            state.get("gradient_stop_triggered", False)
        )
        if gradient_stop_spacing_min_seen < 1e17:
            state["gradient_stop_spacing_observed"] = float(
                gradient_stop_spacing_min_seen
            )
        state["gradient_escape_stop_mode"] = gradient_escape_stop_mode
        state["gradient_escape_dwell_time"] = float(gradient_escape_dwell_time)
        state["gradient_escape_triggered"] = bool(gradient_escape_triggered)
        state["gradient_escape_plane"] = float(gradient_escape_plane)
        state["gradient_escape_tail_plane"] = float(gradient_escape_tail_plane)
        state["gradient_escape_axis_edge"] = float(gradient_escape_axis_edge)
        state["gradient_escape_dwell_elapsed"] = float(gradient_escape_dwell_elapsed)
        state["gradient_escape_return_margin"] = float(gradient_escape_return_margin)
        state["gradient_escape_armed"] = bool(gradient_escape_armed)
        state["gradient_escape_origin_u"] = float(gradient_escape_origin_u)
        state["gradient_axis_law"] = GRADIENT_AXIS_LAW_MULTIPLICATIVE
    state["reaction_steps"] = int(state.get("reaction_steps", 0)) + int(reaction_steps)
    state["bind_events"] = int(state.get("bind_events", 0)) + int(
        bind_events
    )
    state["unbind_events"] = int(state.get("unbind_events", 0)) + int(
        unbind_events
    )
    state["cleavage_events"] = int(state.get("cleavage_events", 0)) + int(
        cleavage_events
    )
    state["thermal_brownian_enabled"] = True
    state["thermal_brownian_dt_mode"] = "adaptive"
    state["thermal_brownian_dt_min"] = float(adaptive_dt_min)
    state["thermal_brownian_dt_max"] = float(adaptive_dt_max)
    state["thermal_passive_d_parallel"] = float(diffusion_parallel)
    state["thermal_passive_d_perp"] = float(diffusion_perp)
    state["thermal_passive_d_rot"] = float(diffusion_rot)
    state["receptor_mobility_mode"] = RECEPTOR_MOBILITY_FIXED
    state["max_nearby_hint"] = int(max_nearby)

    return {
        "times": times_chunk,
        "positions": positions_chunk,
        "done": bool(done),
        "termination_code": int(termination_code),
        "termination_reason": _termination_reason_from_code(int(termination_code)),
        "n_cleaved": int(n_cleaved),
        "bind_events": int(bind_events),
        "unbind_events": int(unbind_events),
        "cleavage_events": int(cleavage_events),
    }


def _run_simulation_chunk_uniform_sparse_infinite(
    config: Dict[str, Any],
    ligand_types: np.ndarray,
    state: Dict[str, Any],
    max_steps: int,
    nearby_cutoff_alpha_mult: float,
    nearby_cutoff_validate_alpha_mult: float,
    tail_propensity_eps: float,
    no_nearby_policy_mode: str,
) -> Dict[str, Any]:
    _ensure_uniform_sparse_coord_state(state, ligand_types)
    normalize_dimension(state.get("dimension", config.get("DIMENSION", DIMENSION_3D)))
    n_ligands = int(ligand_types.shape[0])
    r_i = np.linspace(-float(config["L"]) / 2.0, float(config["L"]) / 2.0, n_ligands)
    alpha = float(config["ALPHA"])
    receptor_spacing = float(config["RECEPTOR_SPACING"])
    t_final = float(config["T_FINAL"])
    gamma_parallel = float(config["GAMMA_T_PARALLEL"])
    gamma_perp = float(config["GAMMA_T_PERPENDICULAR"])
    gamma_R = float(config["GAMMA_R"])
    min_gamma = float(config.get("MIN_GAMMA", 1e-06))
    k_d = float(config["K_D"])
    k_c = float(config["K_C"])
    base_cutoff = max(0.0, float(nearby_cutoff_alpha_mult)) * alpha
    validate_cutoff = max(base_cutoff, float(nearby_cutoff_validate_alpha_mult) * alpha)
    tail_eps = max(0.0, float(tail_propensity_eps))
    x = float(state.get("x", 0.0))
    y = float(state.get("y", 0.0))
    z = float(state.get("z", 0.0))
    n_hat = np.asarray(
        state.get("n_hat", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        dtype=np.float64,
    )
    last_stored_x = float(state.get("last_stored_x", x))
    last_stored_y = float(state.get("last_stored_y", y))
    last_stored_z = float(state.get("last_stored_z", z))
    last_stored_n = np.asarray(
        state.get("last_stored_n", n_hat.copy()), dtype=np.float64
    )
    t = float(state.get("t", 0.0))
    last_stored_t = float(state.get("last_stored_t", t))
    attempts = int(state.get("attempts", 0))
    step_counter = int(state.get("step_counter", 0))
    bound_active = np.asarray(state["bound_active"], dtype=np.bool_)
    bound_ix = np.asarray(state["bound_ix"], dtype=np.int64)
    bound_iy = np.asarray(state["bound_iy"], dtype=np.int64)
    bound_iz = np.asarray(state["bound_iz"], dtype=np.int64)
    cleaved_ix = np.asarray(state["cleaved_ix"], dtype=np.int64)
    cleaved_iy = np.asarray(state["cleaved_iy"], dtype=np.int64)
    cleaved_iz = np.asarray(state["cleaved_iz"], dtype=np.int64)
    n_cleaved = int(state.get("n_cleaved", 0))
    hash_ix = np.asarray(state["cleaved_hash_ix"], dtype=np.int64)
    hash_iy = np.asarray(state["cleaved_hash_iy"], dtype=np.int64)
    hash_iz = np.asarray(state["cleaved_hash_iz"], dtype=np.int64)
    hash_used = np.asarray(state["cleaved_hash_used"], dtype=np.uint8)
    block_ix = np.asarray(state["cleaved_block_ix"], dtype=np.int64)
    block_iy = np.asarray(state["cleaved_block_iy"], dtype=np.int64)
    block_iz = np.asarray(state["cleaved_block_iz"], dtype=np.int64)
    block_bits = np.asarray(state["cleaved_block_bits"], dtype=np.uint64)
    block_used = np.asarray(state["cleaved_block_used"], dtype=np.uint8)
    block_count = int(state.get("cleaved_block_count", _count_used_slots(block_used)))
    pending_event_active = bool(state.get("pending_event_active", False))
    pending_tau_remaining = float(state.get("pending_tau_remaining", 0.0))
    pending_event_kind = int(state.get("pending_event_kind", int(EVENT_NONE)))
    pending_ligand_idx = int(state.get("pending_ligand_idx", -1))
    pending_receptor_ix = int(state.get("pending_receptor_ix", 0))
    pending_receptor_iy = int(state.get("pending_receptor_iy", 0))
    pending_receptor_iz = int(state.get("pending_receptor_iz", 0))
    pending_n_bound = int(state.get("pending_n_bound", 0))
    pending_tau_total = float(state.get("pending_tau_total", pending_tau_remaining))
    pending_tau_elapsed = float(state.get("pending_tau_elapsed", 0.0))
    pending_base_x = float(state.get("pending_base_x", x))
    pending_base_y = float(state.get("pending_base_y", y))
    pending_base_z = float(state.get("pending_base_z", z))
    pending_base_n = np.asarray(
        state.get("pending_base_n", n_hat.copy()), dtype=np.float64
    )
    pending_ligand_indices = np.asarray(
        state.get(
            "pending_ligand_indices",
            np.full(max(1, n_ligands), np.int32(-1), dtype=np.int32),
        ),
        dtype=np.int32,
    )
    pending_force_x = np.asarray(
        state.get("pending_force_x", np.zeros(max(1, n_ligands), dtype=np.float64)),
        dtype=np.float64,
    )
    pending_force_y = np.asarray(
        state.get("pending_force_y", np.zeros(max(1, n_ligands), dtype=np.float64)),
        dtype=np.float64,
    )
    pending_force_z = np.asarray(
        state.get("pending_force_z", np.zeros(max(1, n_ligands), dtype=np.float64)),
        dtype=np.float64,
    )
    pending_bound_rx = np.asarray(
        state.get("pending_bound_rx", np.zeros(max(1, n_ligands), dtype=np.float64)),
        dtype=np.float64,
    )
    pending_bound_ry = np.asarray(
        state.get("pending_bound_ry", np.zeros(max(1, n_ligands), dtype=np.float64)),
        dtype=np.float64,
    )
    pending_bound_rz = np.asarray(
        state.get("pending_bound_rz", np.zeros(max(1, n_ligands), dtype=np.float64)),
        dtype=np.float64,
    )
    x, y, z, n_hat = clamp_planar_pose(x, y, z, n_hat)
    last_stored_x, last_stored_y, last_stored_z, last_stored_n = clamp_planar_pose(
        last_stored_x, last_stored_y, last_stored_z, last_stored_n
    )
    pending_base_x, pending_base_y, pending_base_z, pending_base_n = clamp_planar_pose(
        pending_base_x, pending_base_y, pending_base_z, pending_base_n
    )
    pending_receptor_iz = 0
    clamp_planar_sparse_coord_state(
        bound_iz, cleaved_iz, hash_iz, pending_force_z, pending_bound_rz
    )
    max_nearby = int(
        max(
            INITIAL_NEARBY_CAPACITY,
            state.get("max_nearby_hint", INITIAL_NEARBY_CAPACITY),
        )
    )
    bound_rx_cache = np.empty(n_ligands, dtype=np.float64)
    bound_ry_cache = np.empty(n_ligands, dtype=np.float64)
    bound_rz_cache = np.empty(n_ligands, dtype=np.float64)
    ligand_bound_buffer = np.zeros(n_ligands, dtype=np.bool_)
    free_ix = np.empty(max_nearby, dtype=np.int64)
    free_iy = np.empty(max_nearby, dtype=np.int64)
    free_iz = np.empty(max_nearby, dtype=np.int64)
    free_s = np.empty(max_nearby, dtype=np.float64)
    free_rho_sq = np.empty(max_nearby, dtype=np.float64)
    free_i_min = np.empty(max_nearby, dtype=np.int32)
    free_i_max = np.empty(max_nearby, dtype=np.int32)
    bind_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
    cleave_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
    bound_slot_ix = np.empty(n_ligands, dtype=np.int64)
    bound_slot_iy = np.empty(n_ligands, dtype=np.int64)
    bound_slot_iz = np.empty(n_ligands, dtype=np.int64)
    bound_slot_ligand_idx = np.empty(n_ligands, dtype=np.int32)
    bound_force_x = np.empty(n_ligands, dtype=np.float64)
    bound_force_y = np.empty(n_ligands, dtype=np.float64)
    bound_force_z = np.empty(n_ligands, dtype=np.float64)
    unbind_rate_by_bound_slot = np.zeros(n_ligands, dtype=np.float64)
    cache_guard = max(4.0 * receptor_spacing, alpha)
    cache_cutoff = base_cutoff
    cache_half_length = 0.5 * float(config["L"])
    cache_valid = False
    cache_last_x = x
    cache_last_y = y
    cache_last_z = z
    cache_last_nx = n_hat[0]
    cache_last_ny = n_hat[1]
    cache_last_nz = n_hat[2]
    cand_cap = max(
        64,
        _estimate_candidate_capacity(
            x,
            y,
            z,
            n_hat[0],
            n_hat[1],
            n_hat[2],
            float(config["L"]),
            receptor_spacing,
            cache_cutoff + cache_guard,
        ),
    )
    cand_ix = np.empty(cand_cap, dtype=np.int64)
    cand_iy = np.empty(cand_cap, dtype=np.int64)
    cand_iz = np.empty(cand_cap, dtype=np.int64)
    cand_rx = np.empty(cand_cap, dtype=np.float64)
    cand_ry = np.empty(cand_cap, dtype=np.float64)
    cand_rz = np.empty(cand_cap, dtype=np.float64)
    cand_count = 0
    times_list = []
    positions_list = []
    termination_code = TERM_CHUNK_LIMIT
    done = False
    local_steps = 0
    displacement_threshold = 0.5 * float(config["L"])
    angle_threshold = 0.25
    time_threshold = _resolve_trajectory_record_interval(config)
    dense_backfill_records = (
        bool(config.get("DISABLE_TRAJECTORY_COMPRESSION", False))
        and time_threshold > 0.0
    )
    policy_code = no_nearby_policy_to_code(no_nearby_policy_mode)
    reaction_steps = 0
    if abs(t) <= 1e-12:
        times_list.append(0.0)
        positions_list.append(build_position_row_for_dimension(x, y, z, n_hat, "2d"))
        last_stored_t = 0.0
    bind_events = 0
    unbind_events = 0
    cleavage_events = 0
    while t < t_final and local_steps < int(max_steps):
        if pending_event_active:
            remaining = t_final - t
            if remaining <= 0.0:
                t = t_final
                done = True
                termination_code = TERM_T_FINAL
                break
            tau = max(0.0, pending_tau_remaining)
            tau_step = tau
            clipped_pending = False
            if tau_step > remaining:
                tau_step = remaining
                clipped_pending = True
            target_elapsed = pending_tau_elapsed + tau_step
            if pending_n_bound > 0 and target_elapsed > 0.0:
                x, y, z, n_hat, broke_early = (
                    update_position_3d_pending_frozen_with_flag(
                        pending_base_x,
                        pending_base_y,
                        pending_base_z,
                        pending_base_n,
                        target_elapsed,
                        r_i,
                        pending_ligand_indices[:pending_n_bound],
                        pending_force_x[:pending_n_bound],
                        pending_force_y[:pending_n_bound],
                        pending_force_z[:pending_n_bound],
                        pending_bound_rx[:pending_n_bound],
                        pending_bound_ry[:pending_n_bound],
                        pending_bound_rz[:pending_n_bound],
                        gamma_parallel,
                        gamma_perp,
                        gamma_R,
                        min_gamma,
                        5.0 * alpha * (5.0 * alpha),
                    )
                )
                if clipped_pending and broke_early:
                    pending_n_bound = 0
                    pending_base_x = x
                    pending_base_y = y
                    pending_base_z = z
                    pending_base_n = n_hat.copy()
                    pending_base_x, pending_base_y, pending_base_z, pending_base_n = (
                        clamp_planar_pose(
                            pending_base_x,
                            pending_base_y,
                            pending_base_z,
                            pending_base_n,
                        )
                    )
                    pending_tau_total = max(0.0, tau - tau_step)
                    pending_tau_elapsed = 0.0
            t += tau_step
            if clipped_pending:
                pending_tau_elapsed = target_elapsed
                pending_tau_remaining = max(
                    0.0, pending_tau_total - pending_tau_elapsed
                )
                t = t_final
                termination_code = TERM_T_FINAL
                done = True
                break
            old_state = np.int8(FREE)
            reaction_applied = False
            bound_lig_here = _find_bound_ligand_for_coord(
                pending_receptor_ix,
                pending_receptor_iy,
                pending_receptor_iz,
                bound_active,
                bound_ix,
                bound_iy,
                bound_iz,
            )
            is_cleaved = _cleaved_contains_coord(
                pending_receptor_ix,
                pending_receptor_iy,
                pending_receptor_iz,
                hash_ix,
                hash_iy,
                hash_iz,
                hash_used,
                block_ix,
                block_iy,
                block_iz,
                block_bits,
                block_used,
            )
            if is_cleaved:
                old_state = np.int8(CLEAVED)
            elif bound_lig_here >= 0:
                old_state = np.int8(BOUND)
            if pending_event_kind == int(EVENT_BIND):
                if (
                    old_state == np.int8(FREE)
                    and pending_ligand_idx >= 0
                    and (pending_ligand_idx < n_ligands)
                    and (not bool(bound_active[pending_ligand_idx]))
                ):
                    bound_active[pending_ligand_idx] = True
                    bound_ix[pending_ligand_idx] = np.int64(pending_receptor_ix)
                    bound_iy[pending_ligand_idx] = np.int64(pending_receptor_iy)
                    bound_iz[pending_ligand_idx] = np.int64(pending_receptor_iz)
                    reaction_applied = True
            elif pending_event_kind == int(EVENT_UNBIND):
                unbind_lig = pending_ligand_idx
                if unbind_lig < 0 or unbind_lig >= n_ligands:
                    unbind_lig = bound_lig_here
                if (
                    unbind_lig >= 0
                    and unbind_lig < n_ligands
                    and bool(bound_active[unbind_lig])
                ):
                    if (
                        int(bound_ix[unbind_lig]) == pending_receptor_ix
                        and int(bound_iy[unbind_lig]) == pending_receptor_iy
                        and (int(bound_iz[unbind_lig]) == pending_receptor_iz)
                    ):
                        bound_active[unbind_lig] = False
                        reaction_applied = True
            elif pending_event_kind == int(EVENT_CLEAVE):
                if old_state != np.int8(CLEAVED):
                    if bound_lig_here >= 0:
                        bound_active[bound_lig_here] = False
                    (
                        cleaved_ix,
                        cleaved_iy,
                        cleaved_iz,
                        n_cleaved,
                        hash_ix,
                        hash_iy,
                        hash_iz,
                        hash_used,
                        block_ix,
                        block_iy,
                        block_iz,
                        block_bits,
                        block_used,
                        block_count,
                        _,
                    ) = _insert_cleaved_coord_blocked(
                        pending_receptor_ix,
                        pending_receptor_iy,
                        pending_receptor_iz,
                        cleaved_ix,
                        cleaved_iy,
                        cleaved_iz,
                        n_cleaved,
                        hash_ix,
                        hash_iy,
                        hash_iz,
                        hash_used,
                        block_ix,
                        block_iy,
                        block_iz,
                        block_bits,
                        block_used,
                        block_count,
                    )
                    reaction_applied = True
            if reaction_applied:
                if int(pending_event_kind) == int(EVENT_BIND):
                    bind_events += 1
                elif int(pending_event_kind) == int(EVENT_UNBIND):
                    unbind_events += 1
                elif int(pending_event_kind) == int(EVENT_CLEAVE):
                    cleavage_events += 1
            pending_event_active = False
            pending_tau_remaining = 0.0
            pending_event_kind = int(EVENT_NONE)
            pending_ligand_idx = -1
            pending_n_bound = 0
            pending_tau_total = 0.0
            pending_tau_elapsed = 0.0
            pending_base_x = x
            pending_base_y = y
            pending_base_z = z
            pending_base_n = n_hat.copy()
            x, y, z, n_hat = clamp_planar_pose(x, y, z, n_hat)
            pending_base_x, pending_base_y, pending_base_z, pending_base_n = (
                clamp_planar_pose(
                    pending_base_x, pending_base_y, pending_base_z, pending_base_n
                )
            )
            bound_iz[:] = 0
            cleaved_iz[:n_cleaved] = 0
            hash_iz[:] = 0
            displacement = np.sqrt(
                (x - last_stored_x) ** 2
                + (y - last_stored_y) ** 2
                + (z - last_stored_z) ** 2
            )
            dot_product = (
                n_hat[0] * last_stored_n[0]
                + n_hat[1] * last_stored_n[1]
                + n_hat[2] * last_stored_n[2]
            )
            dot_product = min(1.0, max(-1.0, dot_product))
            angle_change = np.arccos(dot_product)
            if (
                displacement > displacement_threshold
                or angle_change > angle_threshold
                or t - last_stored_t >= time_threshold
            ):
                (
                    last_stored_x,
                    last_stored_y,
                    last_stored_z,
                    last_stored_t,
                    last_stored_n,
                ) = _append_trajectory_samples_with_optional_backfill(
                    times_list,
                    positions_list,
                    dimension="2d",
                    last_t=last_stored_t,
                    last_x=last_stored_x,
                    last_y=last_stored_y,
                    last_z=last_stored_z,
                    last_n=last_stored_n,
                    t=t,
                    x=x,
                    y=y,
                    z=z,
                    n_hat=n_hat,
                    time_threshold=time_threshold,
                    dense_backfill=dense_backfill_records,
                )
            if t >= t_final:
                done = True
                termination_code = TERM_T_FINAL
                break
            continue
        n_x, n_y, n_z = (float(n_hat[0]), float(n_hat[1]), float(n_hat[2]))
        step_counter += 1
        local_steps += 1
        candidate_count_for_buffers = cand_count
        if cache_valid:
            if not _rod_candidate_cache_contains_new_pose(
                cache_last_x,
                cache_last_y,
                cache_last_z,
                cache_last_nx,
                cache_last_ny,
                cache_last_nz,
                x,
                y,
                z,
                n_x,
                n_y,
                n_z,
                cache_half_length,
                cache_guard,
            ):
                cache_valid = False
        if not cache_valid:
            needed_cap = _estimate_candidate_capacity(
                x,
                y,
                z,
                n_x,
                n_y,
                n_z,
                float(config["L"]),
                receptor_spacing,
                cache_cutoff + cache_guard,
            )
            if needed_cap > len(cand_ix):
                cand_ix = np.empty(needed_cap, dtype=np.int64)
                cand_iy = np.empty(needed_cap, dtype=np.int64)
                cand_iz = np.empty(needed_cap, dtype=np.int64)
                cand_rx = np.empty(needed_cap, dtype=np.float64)
                cand_ry = np.empty(needed_cap, dtype=np.float64)
                cand_rz = np.empty(needed_cap, dtype=np.float64)
            cand_count = _build_candidate_cache_uniform_sparse_geometry_2d(
                x,
                y,
                n_x,
                n_y,
                float(config["L"]),
                receptor_spacing,
                r_i,
                cache_cutoff + cache_guard,
                hash_ix,
                hash_iy,
                hash_iz,
                hash_used,
                block_ix,
                block_iy,
                block_iz,
                block_bits,
                block_used,
                cand_ix,
                cand_iy,
                cand_iz,
                cand_rx,
                cand_ry,
                cand_rz,
            )
            if cand_count < 0:
                needed_cap = max(len(cand_ix) * 2, 1024)
                cand_ix = np.empty(needed_cap, dtype=np.int64)
                cand_iy = np.empty(needed_cap, dtype=np.int64)
                cand_iz = np.empty(needed_cap, dtype=np.int64)
                cand_rx = np.empty(needed_cap, dtype=np.float64)
                cand_ry = np.empty(needed_cap, dtype=np.float64)
                cand_rz = np.empty(needed_cap, dtype=np.float64)
                cand_count = _build_candidate_cache_uniform_sparse_geometry_2d(
                    x,
                    y,
                    n_x,
                    n_y,
                    float(config["L"]),
                    receptor_spacing,
                    r_i,
                    cache_cutoff + cache_guard,
                    hash_ix,
                    hash_iy,
                    hash_iz,
                    hash_used,
                    block_ix,
                    block_iy,
                    block_iz,
                    block_bits,
                    block_used,
                    cand_ix,
                    cand_iy,
                    cand_iz,
                    cand_rx,
                    cand_ry,
                    cand_rz,
                )
            cache_last_x, cache_last_y, cache_last_z = (x, y, z)
            cache_last_nx, cache_last_ny, cache_last_nz = (n_x, n_y, n_z)
            cache_valid = True
        candidate_count_for_buffers = cand_count
        required_nearby = candidate_count_for_buffers + n_ligands + NEARBY_PADDING
        if required_nearby > max_nearby:
            grown = max_nearby + max_nearby // 2
            required = required_nearby
            new_max = required if required > grown else grown
            max_nearby = int(new_max)
            free_ix = np.empty(max_nearby, dtype=np.int64)
            free_iy = np.empty(max_nearby, dtype=np.int64)
            free_iz = np.empty(max_nearby, dtype=np.int64)
            free_s = np.empty(max_nearby, dtype=np.float64)
            free_rho_sq = np.empty(max_nearby, dtype=np.float64)
            free_i_min = np.empty(max_nearby, dtype=np.int32)
            free_i_max = np.empty(max_nearby, dtype=np.int32)
            bind_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
            cleave_sum_by_free_receptor = np.zeros(max_nearby, dtype=np.float64)
        remaining = t_final - t
        if remaining <= 0.0:
            t = t_final
            done = True
            termination_code = TERM_T_FINAL
            break
        active_cand_ix = cand_ix
        active_cand_iy = cand_iy
        active_cand_iz = cand_iz
        active_cand_rx = cand_rx
        active_cand_ry = cand_ry
        active_cand_rz = cand_rz
        active_cand_count = cand_count
        n_free, n_bound, _ = _build_surface_reaction_geometry(
            x,
            y,
            z,
            n_x,
            n_y,
            n_z,
            receptor_spacing,
            base_cutoff,
            5.0 * alpha * (5.0 * alpha),
            r_i,
            active_cand_ix,
            active_cand_iy,
            active_cand_iz,
            active_cand_rx,
            active_cand_ry,
            active_cand_rz,
            active_cand_count,
            hash_ix,
            hash_iy,
            hash_iz,
            hash_used,
            bound_active,
            bound_ix,
            bound_iy,
            bound_iz,
            ligand_bound_buffer,
            free_ix,
            free_iy,
            free_iz,
            free_s,
            free_rho_sq,
            free_i_min,
            free_i_max,
            bound_slot_ix,
            bound_slot_iy,
            bound_slot_iz,
            bound_slot_ligand_idx,
            bound_rx_cache,
            bound_ry_cache,
            bound_rz_cache,
            bound_force_x,
            bound_force_y,
            bound_force_z,
        )
        if n_free < 0:
            termination_code = TERM_RESOURCE_LIMIT
            done = True
            break
        total_propensity, _, _, _, _ = build_reaction_rate_tables(
            n_free,
            free_s,
            free_rho_sq,
            free_i_min,
            free_i_max,
            n_bound,
            bound_slot_ligand_idx,
            bound_force_x,
            bound_force_y,
            bound_force_z,
            r_i,
            alpha,
            k_d,
            k_c,
            ligand_types,
            ligand_bound_buffer,
            bind_sum_by_free_receptor,
            cleave_sum_by_free_receptor,
            unbind_rate_by_bound_slot,
        )
        used_validation_query = False
        if (
            total_propensity <= 0.0
            and policy_code == NO_NEARBY_POLICY_VALIDATED_TERMINAL_CODE
        ):
            used_validation_query = True
            val_cand_ix = active_cand_ix
            val_cand_iy = active_cand_iy
            val_cand_iz = active_cand_iz
            val_cand_rx = active_cand_rx
            val_cand_ry = active_cand_ry
            val_cand_rz = active_cand_rz
            val_cand_count = active_cand_count
            n_free, n_bound, _ = _build_surface_reaction_geometry(
                x,
                y,
                z,
                n_x,
                n_y,
                n_z,
                receptor_spacing,
                validate_cutoff,
                5.0 * alpha * (5.0 * alpha),
                r_i,
                val_cand_ix,
                val_cand_iy,
                val_cand_iz,
                val_cand_rx,
                val_cand_ry,
                val_cand_rz,
                val_cand_count,
                hash_ix,
                hash_iy,
                hash_iz,
                hash_used,
                bound_active,
                bound_ix,
                bound_iy,
                bound_iz,
                ligand_bound_buffer,
                free_ix,
                free_iy,
                free_iz,
                free_s,
                free_rho_sq,
                free_i_min,
                free_i_max,
                bound_slot_ix,
                bound_slot_iy,
                bound_slot_iz,
                bound_slot_ligand_idx,
                bound_rx_cache,
                bound_ry_cache,
                bound_rz_cache,
                bound_force_x,
                bound_force_y,
                bound_force_z,
            )
            if n_free < 0:
                termination_code = TERM_RESOURCE_LIMIT
                done = True
                break
            total_propensity, _, _, _, _ = build_reaction_rate_tables(
                n_free,
                free_s,
                free_rho_sq,
                free_i_min,
                free_i_max,
                n_bound,
                bound_slot_ligand_idx,
                bound_force_x,
                bound_force_y,
                bound_force_z,
                r_i,
                alpha,
                k_d,
                k_c,
                ligand_types,
                ligand_bound_buffer,
                bind_sum_by_free_receptor,
                cleave_sum_by_free_receptor,
                unbind_rate_by_bound_slot,
            )
        if total_propensity <= (tail_eps if used_validation_query else 0.0):
            if not pending_event_active:
                attempts += 1
                if attempts >= 5:
                    termination_code = TERM_NO_NEARBY
                    done = True
                    break
                continue
        attempts = 0
        (
            reaction_occurred,
            tau,
            event_kind,
            event_slot_idx,
            event_ligand_idx,
        ) = sample_reaction_event(
            n_free,
            bind_sum_by_free_receptor,
            cleave_sum_by_free_receptor,
            free_s,
            free_rho_sq,
            free_i_min,
            free_i_max,
            n_bound,
            unbind_rate_by_bound_slot,
            bound_slot_ligand_idx,
            r_i,
            alpha,
            k_d,
            k_c,
            ligand_types,
            ligand_bound_buffer,
            total_propensity,
        )
        reaction_steps += 1
        tau_step = tau
        clipped = False
        if tau_step > remaining:
            tau_step = remaining
            clipped = True
        event_base_x = x
        event_base_y = y
        event_base_z = z
        event_base_n = n_hat.copy()
        if n_bound > 0 and tau_step > 0.0:
            x, y, z, n_hat = update_position_3d_bound_vectors(
                x,
                y,
                z,
                n_hat,
                tau_step,
                r_i,
                bound_slot_ligand_idx[:n_bound],
                bound_force_x[:n_bound],
                bound_force_y[:n_bound],
                bound_force_z[:n_bound],
                bound_rx_cache[:n_bound],
                bound_ry_cache[:n_bound],
                bound_rz_cache[:n_bound],
                gamma_parallel,
                gamma_perp,
                gamma_R,
                min_gamma,
                5.0 * alpha * (5.0 * alpha),
            )
        t += tau_step
        if clipped:
            pending_event_active = bool(
                reaction_occurred and event_slot_idx >= 0 and (event_kind != EVENT_NONE)
            )
            if pending_event_active:
                pending_tau_total = float(tau)
                pending_tau_elapsed = float(tau_step)
                pending_tau_remaining = max(
                    0.0, pending_tau_total - pending_tau_elapsed
                )
                pending_event_kind = int(event_kind)
                pending_ligand_idx = int(event_ligand_idx)
                if int(event_kind) == int(EVENT_UNBIND):
                    pending_receptor_ix = int(bound_slot_ix[event_slot_idx])
                    pending_receptor_iy = int(bound_slot_iy[event_slot_idx])
                    pending_receptor_iz = int(bound_slot_iz[event_slot_idx])
                else:
                    pending_receptor_ix = int(free_ix[event_slot_idx])
                    pending_receptor_iy = int(free_iy[event_slot_idx])
                    pending_receptor_iz = int(free_iz[event_slot_idx])
                pending_base_x = event_base_x
                pending_base_y = event_base_y
                pending_base_z = event_base_z
                pending_base_n = event_base_n
                pending_receptor_iz = 0
                pending_base_x, pending_base_y, pending_base_z, pending_base_n = (
                    clamp_planar_pose(
                        pending_base_x, pending_base_y, pending_base_z, pending_base_n
                    )
                )
                pending_n_bound = int(min(n_bound, n_ligands))
                for k in range(pending_n_bound):
                    pending_ligand_indices[k] = np.int32(bound_slot_ligand_idx[k])
                    pending_force_x[k] = bound_force_x[k]
                    pending_force_y[k] = bound_force_y[k]
                    pending_force_z[k] = bound_force_z[k]
                    pending_bound_rx[k] = bound_rx_cache[k]
                    pending_bound_ry[k] = bound_ry_cache[k]
                    pending_bound_rz[k] = bound_rz_cache[k]
                pending_force_z[:pending_n_bound] = 0.0
                pending_bound_rz[:pending_n_bound] = 0.0
            else:
                pending_tau_remaining = 0.0
                pending_event_kind = int(EVENT_NONE)
                pending_ligand_idx = -1
                pending_n_bound = 0
                pending_tau_total = 0.0
                pending_tau_elapsed = 0.0
                pending_base_x = x
                pending_base_y = y
                pending_base_z = z
                pending_base_n = n_hat.copy()
                pending_base_x, pending_base_y, pending_base_z, pending_base_n = (
                    clamp_planar_pose(
                        pending_base_x, pending_base_y, pending_base_z, pending_base_n
                    )
                )
            t = t_final
            done = True
            termination_code = TERM_T_FINAL
            break
        if reaction_occurred and event_slot_idx >= 0 and (event_kind != EVENT_NONE):
            if int(event_kind) == int(EVENT_UNBIND):
                rec_ix = int(bound_slot_ix[event_slot_idx])
                rec_iy = int(bound_slot_iy[event_slot_idx])
                rec_iz = int(bound_slot_iz[event_slot_idx])
            else:
                rec_ix = int(free_ix[event_slot_idx])
                rec_iy = int(free_iy[event_slot_idx])
                rec_iz = int(free_iz[event_slot_idx])
            bound_lig_here = _find_bound_ligand_for_coord(
                rec_ix, rec_iy, rec_iz, bound_active, bound_ix, bound_iy, bound_iz
            )
            is_cleaved = _cleaved_contains_coord(
                rec_ix,
                rec_iy,
                rec_iz,
                hash_ix,
                hash_iy,
                hash_iz,
                hash_used,
                block_ix,
                block_iy,
                block_iz,
                block_bits,
                block_used,
            )
            old_state = np.int8(
                CLEAVED if is_cleaved else BOUND if bound_lig_here >= 0 else FREE
            )
            reaction_applied = False
            if int(event_kind) == int(EVENT_BIND):
                if (
                    old_state == np.int8(FREE)
                    and event_ligand_idx >= 0
                    and (event_ligand_idx < n_ligands)
                    and (not bool(bound_active[event_ligand_idx]))
                ):
                    bound_active[event_ligand_idx] = True
                    bound_ix[event_ligand_idx] = np.int64(rec_ix)
                    bound_iy[event_ligand_idx] = np.int64(rec_iy)
                    bound_iz[event_ligand_idx] = np.int64(rec_iz)
                    reaction_applied = True
            elif int(event_kind) == int(EVENT_UNBIND):
                unbind_lig = event_ligand_idx
                if unbind_lig < 0 or unbind_lig >= n_ligands:
                    unbind_lig = bound_lig_here
                if (
                    unbind_lig >= 0
                    and unbind_lig < n_ligands
                    and bool(bound_active[unbind_lig])
                ):
                    if (
                        int(bound_ix[unbind_lig]) == rec_ix
                        and int(bound_iy[unbind_lig]) == rec_iy
                        and (int(bound_iz[unbind_lig]) == rec_iz)
                    ):
                        bound_active[unbind_lig] = False
                        reaction_applied = True
            elif int(event_kind) == int(EVENT_CLEAVE):
                if old_state != np.int8(CLEAVED):
                    if bound_lig_here >= 0:
                        bound_active[bound_lig_here] = False
                    (
                        cleaved_ix,
                        cleaved_iy,
                        cleaved_iz,
                        n_cleaved,
                        hash_ix,
                        hash_iy,
                        hash_iz,
                        hash_used,
                        block_ix,
                        block_iy,
                        block_iz,
                        block_bits,
                        block_used,
                        block_count,
                        _,
                    ) = _insert_cleaved_coord_blocked(
                        rec_ix,
                        rec_iy,
                        rec_iz,
                        cleaved_ix,
                        cleaved_iy,
                        cleaved_iz,
                        n_cleaved,
                        hash_ix,
                        hash_iy,
                        hash_iz,
                        hash_used,
                        block_ix,
                        block_iy,
                        block_iz,
                        block_bits,
                        block_used,
                        block_count,
                    )
                    reaction_applied = True
            if reaction_applied:
                if int(event_kind) == int(EVENT_BIND):
                    bind_events += 1
                elif int(event_kind) == int(EVENT_UNBIND):
                    unbind_events += 1
                elif int(event_kind) == int(EVENT_CLEAVE):
                    cleavage_events += 1
        x, y, z, n_hat = clamp_planar_pose(x, y, z, n_hat)
        bound_iz[:] = 0
        cleaved_iz[:n_cleaved] = 0
        hash_iz[:] = 0
        displacement = np.sqrt(
            (x - last_stored_x) ** 2
            + (y - last_stored_y) ** 2
            + (z - last_stored_z) ** 2
        )
        dot_product = (
            n_hat[0] * last_stored_n[0]
            + n_hat[1] * last_stored_n[1]
            + n_hat[2] * last_stored_n[2]
        )
        dot_product = min(1.0, max(-1.0, dot_product))
        angle_change = np.arccos(dot_product)
        if (
            displacement > displacement_threshold
            or angle_change > angle_threshold
            or t - last_stored_t >= time_threshold
        ):
            (
                last_stored_x,
                last_stored_y,
                last_stored_z,
                last_stored_t,
                last_stored_n,
            ) = _append_trajectory_samples_with_optional_backfill(
                times_list,
                positions_list,
                dimension="2d",
                last_t=last_stored_t,
                last_x=last_stored_x,
                last_y=last_stored_y,
                last_z=last_stored_z,
                last_n=last_stored_n,
                t=t,
                x=x,
                y=y,
                z=z,
                n_hat=n_hat,
                time_threshold=time_threshold,
                dense_backfill=dense_backfill_records,
            )
        if t >= t_final:
            done = True
            termination_code = TERM_T_FINAL
            break
        continue
    if not done and t >= t_final:
        done = True
        termination_code = TERM_T_FINAL
    if done:
        if len(times_list) == 0 or abs(times_list[-1] - t) > 1e-10:
            (
                last_stored_x,
                last_stored_y,
                last_stored_z,
                last_stored_t,
                last_stored_n,
            ) = _append_trajectory_samples_with_optional_backfill(
                times_list,
                positions_list,
                dimension="2d",
                last_t=last_stored_t,
                last_x=last_stored_x,
                last_y=last_stored_y,
                last_z=last_stored_z,
                last_n=last_stored_n,
                t=t,
                x=x,
                y=y,
                z=z,
                n_hat=n_hat,
                time_threshold=time_threshold,
                dense_backfill=dense_backfill_records,
            )
    times_chunk = (
        np.asarray(times_list, dtype=np.float64)
        if len(times_list) > 0
        else np.zeros(0, dtype=np.float64)
    )
    positions_chunk = (
        np.asarray(positions_list, dtype=np.float64)
        if len(positions_list) > 0
        else np.zeros((0, 3), dtype=np.float64)
    )
    x, y, z, n_hat = clamp_planar_pose(x, y, z, n_hat)
    last_stored_x, last_stored_y, last_stored_z, last_stored_n = clamp_planar_pose(
        last_stored_x, last_stored_y, last_stored_z, last_stored_n
    )
    pending_base_x, pending_base_y, pending_base_z, pending_base_n = clamp_planar_pose(
        pending_base_x, pending_base_y, pending_base_z, pending_base_n
    )
    pending_receptor_iz = 0
    clamp_planar_sparse_coord_state(
        bound_iz, cleaved_iz, hash_iz, pending_force_z, pending_bound_rz
    )
    state["x"] = float(x)
    state["y"] = float(y)
    state["z"] = float(z)
    state["n_hat"] = np.asarray(n_hat, dtype=np.float64)
    state["last_stored_x"] = float(last_stored_x)
    state["last_stored_y"] = float(last_stored_y)
    state["last_stored_z"] = float(last_stored_z)
    state["last_stored_n"] = np.asarray(last_stored_n, dtype=np.float64)
    state["last_stored_t"] = float(last_stored_t)
    state["t"] = float(t)
    state["attempts"] = int(attempts)
    state["step_counter"] = int(step_counter)
    state["bound_active"] = bound_active
    state["bound_ix"] = bound_ix
    state["bound_iy"] = bound_iy
    state["bound_iz"] = bound_iz
    state["n_bound_tracked"] = int(_count_bound_active(bound_active))
    state["cleaved_ix"] = cleaved_ix
    state["cleaved_iy"] = cleaved_iy
    state["cleaved_iz"] = cleaved_iz
    state["n_cleaved"] = int(n_cleaved)
    state["cleaved_hash_ix"] = hash_ix
    state["cleaved_hash_iy"] = hash_iy
    state["cleaved_hash_iz"] = hash_iz
    state["cleaved_hash_used"] = hash_used
    state["cleaved_block_ix"] = block_ix
    state["cleaved_block_iy"] = block_iy
    state["cleaved_block_iz"] = block_iz
    state["cleaved_block_bits"] = block_bits
    state["cleaved_block_used"] = block_used
    state["cleaved_block_count"] = int(block_count)
    state["cleaved_block_source_n"] = int(n_cleaved)
    state["pending_event_active"] = bool(pending_event_active)
    state["pending_tau_remaining"] = float(pending_tau_remaining)
    state["pending_event_kind"] = int(pending_event_kind)
    state["pending_ligand_idx"] = int(pending_ligand_idx)
    state["pending_receptor_ix"] = int(pending_receptor_ix)
    state["pending_receptor_iy"] = int(pending_receptor_iy)
    state["pending_receptor_iz"] = int(pending_receptor_iz)
    state["pending_n_bound"] = int(pending_n_bound)
    state["pending_tau_total"] = float(pending_tau_total)
    state["pending_tau_elapsed"] = float(pending_tau_elapsed)
    state["pending_base_x"] = float(pending_base_x)
    state["pending_base_y"] = float(pending_base_y)
    state["pending_base_z"] = float(pending_base_z)
    state["pending_base_n"] = np.asarray(pending_base_n, dtype=np.float64)
    state["pending_ligand_indices"] = pending_ligand_indices
    state["pending_force_x"] = pending_force_x
    state["pending_force_y"] = pending_force_y
    state["pending_force_z"] = pending_force_z
    state["pending_bound_rx"] = pending_bound_rx
    state["pending_bound_ry"] = pending_bound_ry
    state["pending_bound_rz"] = pending_bound_rz
    state["reaction_steps"] = int(state.get("reaction_steps", 0)) + int(reaction_steps)
    state["bind_events"] = int(state.get("bind_events", 0)) + int(
        bind_events
    )
    state["unbind_events"] = int(state.get("unbind_events", 0)) + int(
        unbind_events
    )
    state["cleavage_events"] = int(state.get("cleavage_events", 0)) + int(
        cleavage_events
    )
    state["motion_rule"] = MOTION_RULE_ATHERMAL
    state["thermal_brownian_enabled"] = False
    state["receptor_mobility_mode"] = RECEPTOR_MOBILITY_FIXED
    state["max_nearby_hint"] = int(max_nearby)
    return {
        "times": times_chunk,
        "positions": positions_chunk,
        "done": bool(done),
        "termination_code": int(termination_code),
        "termination_reason": _termination_reason_from_code(int(termination_code)),
        "n_cleaved": int(n_cleaved),
        "bind_events": int(bind_events),
        "unbind_events": int(unbind_events),
        "cleavage_events": int(cleavage_events),
    }


def run_simulation_chunk_3d(
    config: Dict[str, Any],
    ligand_types: np.ndarray,
    state: Dict[str, Any],
    max_steps: int,
    nearby_cutoff_alpha_mult: float = DEFAULT_NEARBY_CUTOFF_ALPHA_MULT,
    nearby_cutoff_validate_alpha_mult: float = DEFAULT_NEARBY_CUTOFF_VALIDATE_ALPHA_MULT,
    tail_propensity_eps: float = DEFAULT_TAIL_PROPENSITY_EPS,
    no_nearby_policy: str = NO_NEARBY_POLICY_VALIDATED_TERMINAL,
) -> Dict[str, Any]:
    """Advance one manuscript trajectory by at most ``max_steps`` events."""
    dimension = validate_dimension_config(config)
    backend = str(state.get("grid_backend", ""))
    is_uniform = str(config["GRADIENT_TYPE"]).lower() == "uniform" or np.isclose(
        float(config["GRADIENT_SCALE"]), 1.0
    )
    state["dimension"] = dimension
    state["nearby_cutoff_alpha_mult"] = float(nearby_cutoff_alpha_mult)
    state["nearby_cutoff_validate_alpha_mult"] = max(
        float(nearby_cutoff_alpha_mult),
        float(nearby_cutoff_validate_alpha_mult),
    )
    state["tail_propensity_eps"] = max(0.0, float(tail_propensity_eps))
    state["no_nearby_policy"] = normalize_no_nearby_policy(no_nearby_policy)
    if backend == GRID_BACKEND_UNIFORM_SPARSE_COORDS and is_uniform:
        if dimension == DIMENSION_3D:
            return _run_simulation_chunk_sparse_thermal(
                config=config,
                ligand_types=ligand_types,
                state=state,
                max_steps=max(1, int(max_steps)),
                nearby_cutoff_alpha_mult=float(nearby_cutoff_alpha_mult),
            )
        return _run_simulation_chunk_uniform_sparse_infinite(
            config=config,
            ligand_types=ligand_types,
            state=state,
            max_steps=max(1, int(max_steps)),
            nearby_cutoff_alpha_mult=float(nearby_cutoff_alpha_mult),
            nearby_cutoff_validate_alpha_mult=float(
                state["nearby_cutoff_validate_alpha_mult"]
            ),
            tail_propensity_eps=float(state["tail_propensity_eps"]),
            no_nearby_policy_mode=str(state["no_nearby_policy"]),
        )
    if (
        backend == GRID_BACKEND_GRADIENT_SPARSE_COORDS
        and not is_uniform
        and dimension == DIMENSION_3D
    ):
        return _run_simulation_chunk_sparse_thermal(
            config=config,
            ligand_types=ligand_types,
            state=state,
            max_steps=max(1, int(max_steps)),
            nearby_cutoff_alpha_mult=float(nearby_cutoff_alpha_mult),
        )
    raise ValueError("Unsupported geometry or receptor representation")
