from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numba import njit


GRADIENT_AXIS_LAW_MULTIPLICATIVE = "multiplicative_spacing"
DEFAULT_GRADIENT_MIN_SPACING = 0.1
DEFAULT_GRADIENT_MAX_SPACING = 5.0


def _clip_spacing(value: float, min_spacing: float, max_spacing: float) -> float:
    if value < min_spacing:
        return float(min_spacing)
    if value > max_spacing:
        return float(max_spacing)
    return float(value)


def _build_one_sided_prefix(
    start_spacing: float,
    factor: float,
    sign: float,
    min_spacing: float,
    max_spacing: float,
) -> Tuple[np.ndarray, float]:
    positions = [0.0]
    current_pos = 0.0
    current_spacing = _clip_spacing(float(start_spacing), min_spacing, max_spacing)

    while True:
        current_pos = current_pos + sign * current_spacing
        positions.append(float(current_pos))
        next_spacing = _clip_spacing(current_spacing * factor, min_spacing, max_spacing)
        if abs(next_spacing - current_spacing) <= 1.0e-15:
            return np.asarray(positions, dtype=np.float64), float(current_spacing)
        current_spacing = next_spacing


def build_gradient_axis_spec(
    spacing: float,
    gradient_scale: float,
    min_spacing: float = DEFAULT_GRADIENT_MIN_SPACING,
    max_spacing: float = DEFAULT_GRADIENT_MAX_SPACING,
) -> Dict[str, object]:
    spacing = float(spacing)
    gradient_scale = float(gradient_scale)
    min_spacing = float(min_spacing)
    max_spacing = float(max_spacing)

    if gradient_scale <= 0.0:
        raise ValueError("gradient_scale must be positive")

    pos_prefix, pos_tail_spacing = _build_one_sided_prefix(
        start_spacing=spacing,
        factor=gradient_scale,
        sign=1.0,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
    )
    neg_start = spacing / gradient_scale if gradient_scale != 0.0 else spacing
    neg_prefix, neg_tail_spacing = _build_one_sided_prefix(
        start_spacing=neg_start,
        factor=(1.0 / gradient_scale) if gradient_scale != 0.0 else 1.0,
        sign=-1.0,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
    )
    return {
        "law": GRADIENT_AXIS_LAW_MULTIPLICATIVE,
        "spacing": spacing,
        "gradient_scale": gradient_scale,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "pos_prefix": pos_prefix,
        "neg_prefix": neg_prefix,
        "pos_tail_spacing": float(pos_tail_spacing),
        "neg_tail_spacing": float(neg_tail_spacing),
    }


@njit(cache=True)
def axis_position_from_index_implicit(
    idx: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> float:
    if idx >= 0:
        if idx < len(pos_prefix):
            return pos_prefix[idx]
        tail_start_idx = len(pos_prefix) - 1
        tail_start_pos = pos_prefix[tail_start_idx]
        return tail_start_pos + (idx - tail_start_idx) * pos_tail_spacing

    k = -idx
    if k < len(neg_prefix):
        return neg_prefix[k]
    tail_start_idx = len(neg_prefix) - 1
    tail_start_pos = neg_prefix[tail_start_idx]
    return tail_start_pos - (k - tail_start_idx) * neg_tail_spacing


@njit(cache=True)
def axis_spacing_after_index_implicit(
    idx: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> float:
    p0 = axis_position_from_index_implicit(
        idx, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
    )
    p1 = axis_position_from_index_implicit(
        idx + 1, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
    )
    spacing = p1 - p0
    if spacing < 0.0:
        spacing = -spacing
    return spacing


@njit(cache=True)
def dense_coordinate_from_axis_position(coord: float, dense_sign: int) -> float:
    return float(coord) * float(dense_sign)


@njit(cache=True)
def dense_interval_from_axis_interval(
    coord_min: float,
    coord_max: float,
    dense_sign: int,
) -> Tuple[float, float]:
    u0 = dense_coordinate_from_axis_position(coord_min, dense_sign)
    u1 = dense_coordinate_from_axis_position(coord_max, dense_sign)
    if u0 <= u1:
        return u0, u1
    return u1, u0


@njit(cache=True)
def axis_first_dense_coordinate_with_spacing_at_most(
    target_spacing: float,
    dense_sign: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> float:
    if dense_sign == 0 or target_spacing <= 0.0:
        return np.nan

    step = 0
    tol = 1.0e-12
    while step < 1_000_000:
        idx = step if dense_sign > 0 else -(step + 1)
        spacing = axis_spacing_after_index_implicit(
            idx,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        left = axis_position_from_index_implicit(
            idx,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        right = axis_position_from_index_implicit(
            idx + 1,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        u0 = dense_coordinate_from_axis_position(left, dense_sign)
        u1 = dense_coordinate_from_axis_position(right, dense_sign)
        u_low = u0 if u0 < u1 else u1
        if spacing <= target_spacing:
            return u_low

        if dense_sign > 0:
            if step >= (len(pos_prefix) - 1) and abs(spacing - pos_tail_spacing) <= tol:
                return np.nan
        else:
            if step >= (len(neg_prefix) - 1) and abs(spacing - neg_tail_spacing) <= tol:
                return np.nan
        step += 1

    return np.nan


@njit(cache=True)
def axis_first_dense_coordinate_with_spacing_at_least_on_sparse_side(
    target_spacing: float,
    dense_sign: int,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> float:
    if dense_sign == 0 or target_spacing <= 0.0:
        return np.nan

    step = 0
    tol = 1.0e-12
    while step < 1_000_000:
        idx = -(step + 1) if dense_sign > 0 else step
        spacing = axis_spacing_after_index_implicit(
            idx,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        left = axis_position_from_index_implicit(
            idx,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        right = axis_position_from_index_implicit(
            idx + 1,
            pos_prefix,
            neg_prefix,
            pos_tail_spacing,
            neg_tail_spacing,
        )
        u0 = dense_coordinate_from_axis_position(left, dense_sign)
        u1 = dense_coordinate_from_axis_position(right, dense_sign)
        u_high = u0 if u0 > u1 else u1
        if spacing >= target_spacing:
            return u_high

        if dense_sign > 0:
            if step >= (len(neg_prefix) - 1) and abs(spacing - neg_tail_spacing) <= tol:
                return np.nan
        else:
            if step >= (len(pos_prefix) - 1) and abs(spacing - pos_tail_spacing) <= tol:
                return np.nan
        step += 1

    return np.nan


@njit(cache=True)
def axis_lower_bound_implicit(
    target: float,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> int:
    low = -1
    high = 1
    while (
        axis_position_from_index_implicit(
            low, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        >= target
    ):
        high = low
        low = low * 2
    while (
        axis_position_from_index_implicit(
            high, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        < target
    ):
        low = high
        high = high * 2

    while low + 1 < high:
        mid = (low + high) // 2
        if (
            axis_position_from_index_implicit(
                mid, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
            )
            < target
        ):
            low = mid
        else:
            high = mid
    return high


@njit(cache=True)
def axis_upper_bound_implicit(
    target: float,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> int:
    low = -1
    high = 1
    while (
        axis_position_from_index_implicit(
            low, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        > target
    ):
        high = low
        low = low * 2
    while (
        axis_position_from_index_implicit(
            high, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        <= target
    ):
        low = high
        high = high * 2

    while low + 1 < high:
        mid = (low + high) // 2
        if (
            axis_position_from_index_implicit(
                mid, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
            )
            <= target
        ):
            low = mid
        else:
            high = mid
    return low


@njit(cache=True)
def axis_min_spacing_in_interval_implicit(
    coord_min: float,
    coord_max: float,
    pos_prefix: np.ndarray,
    neg_prefix: np.ndarray,
    pos_tail_spacing: float,
    neg_tail_spacing: float,
) -> float:
    if coord_max < coord_min:
        tmp = coord_min
        coord_min = coord_max
        coord_max = tmp

    i_min = (
        axis_lower_bound_implicit(
            coord_min, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        - 1
    )
    i_max = axis_upper_bound_implicit(
        coord_max, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
    )
    if i_max < i_min:
        i_max = i_min

    min_spacing = 1.0e18
    for idx in range(i_min - 1, i_max + 2):
        left = axis_position_from_index_implicit(
            idx, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        right = axis_position_from_index_implicit(
            idx + 1, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
        seg_min = left if left < right else right
        seg_max = right if right > left else left
        if seg_max < coord_min or seg_min > coord_max:
            continue
        spacing = right - left
        if spacing < 0.0:
            spacing = -spacing
        if spacing < min_spacing:
            min_spacing = spacing
    if min_spacing >= 1.0e17:
        min_spacing = axis_spacing_after_index_implicit(
            0, pos_prefix, neg_prefix, pos_tail_spacing, neg_tail_spacing
        )
    return min_spacing
