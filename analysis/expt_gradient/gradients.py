from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import map_coordinates

from .io import MovieRecord, illumination_correct, read_channel_frame, read_metadata


@dataclass
class GradientField:
    movie: str
    pixel_size_um: float
    receptor_corrected: np.ndarray
    receptor_smooth: np.ndarray
    receptor_ring_mean: np.ndarray
    grad_x_per_um: np.ndarray
    grad_y_per_um: np.ndarray
    grad_unit_x: np.ndarray
    grad_unit_y: np.ndarray
    grad_strength_dimless: np.ndarray


def first_harmonic_cue_field(
    image: np.ndarray,
    *,
    radius_pixels: float,
    n_angles: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean and first angular harmonic on a ring around each pixel."""

    radius = float(radius_pixels)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_pixels must be finite and positive")
    if int(n_angles) < 8:
        raise ValueError("n_angles must be at least 8")
    values = np.asarray(image, dtype=float)
    mean = np.zeros_like(values)
    first_x = np.zeros_like(values)
    first_y = np.zeros_like(values)
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    for angle in angles:
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        sample = ndimage.shift(
            values,
            shift=(-radius * sine, -radius * cosine),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        mean += sample
        first_x += cosine * sample
        first_y += sine * sample
    scale = 1.0 / float(n_angles)
    return mean * scale, first_x * scale, first_y * scale


def sample_image(
    image: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    pixel_size_um: float,
    mode: str = "nearest",
) -> np.ndarray:
    coords = np.vstack([y_um / pixel_size_um, x_um / pixel_size_um])
    return map_coordinates(image, coords, order=1, mode=mode)


def build_gradient_field(record: MovieRecord, config: dict) -> GradientField:
    metadata = read_metadata(record)
    receptor = read_channel_frame(
        record, config["experiment"]["receptor_channel"], frame=0
    )
    receptor_corrected = illumination_correct(receptor, sigma=20.0)
    scale_um = float(config["physics"].get("gradient_smoothing_um", 0.25))
    sigma_px = max(scale_um / metadata.pixel_size_um, 0.5)
    smooth = ndimage.gaussian_filter(receptor_corrected.astype(float), sigma=sigma_px)
    radius_um = float(config["physics"].get("receptor_cue_radius_um", 0.5))
    n_angles = int(config["physics"].get("receptor_cue_angles", 96))
    ring_mean, first_x, first_y = first_harmonic_cue_field(
        smooth,
        radius_pixels=radius_um / metadata.pixel_size_um,
        n_angles=n_angles,
    )
    first_amplitude = np.hypot(first_x, first_y)
    unit_x = np.divide(
        first_x,
        first_amplitude,
        out=np.zeros_like(first_x),
        where=first_amplitude > 0.0,
    )
    unit_y = np.divide(
        first_y,
        first_amplitude,
        out=np.zeros_like(first_y),
        where=first_amplitude > 0.0,
    )
    grad_x = 2.0 * first_x / radius_um
    grad_y = 2.0 * first_y / radius_um
    grad_strength = 4.0 * np.divide(
        first_amplitude,
        ring_mean,
        out=np.full_like(first_amplitude, np.nan),
        where=ring_mean > 0.0,
    )
    return GradientField(
        movie=record.movie,
        pixel_size_um=metadata.pixel_size_um,
        receptor_corrected=receptor_corrected,
        receptor_smooth=smooth,
        receptor_ring_mean=ring_mean,
        grad_x_per_um=grad_x,
        grad_y_per_um=grad_y,
        grad_unit_x=unit_x,
        grad_unit_y=unit_y,
        grad_strength_dimless=grad_strength,
    )


def add_event_gradients(
    events: pd.DataFrame, fields: dict[str, GradientField]
) -> pd.DataFrame:
    frames = []
    for movie, group in events.groupby("movie", sort=False):
        field = fields[movie]
        x = group["p1_x_um"].to_numpy(float)
        y = group["p1_y_um"].to_numpy(float)
        gux = sample_image(field.grad_unit_x, x, y, field.pixel_size_um)
        guy = sample_image(field.grad_unit_y, x, y, field.pixel_size_um)
        gx = sample_image(field.grad_x_per_um, x, y, field.pixel_size_um)
        gy = sample_image(field.grad_y_per_um, x, y, field.pixel_size_um)
        g = sample_image(field.grad_strength_dimless, x, y, field.pixel_size_um)
        rho = sample_image(field.receptor_ring_mean, x, y, field.pixel_size_um)
        dx = group["dx_um"].to_numpy(float)
        dy = group["dy_um"].to_numpy(float)
        disp = group["displacement_um"].to_numpy(float)
        ux = np.divide(dx, disp, out=np.zeros_like(dx), where=disp > 0)
        uy = np.divide(dy, disp, out=np.zeros_like(dy), where=disp > 0)
        ci = ux * gux + uy * guy
        out = group.copy()
        out["grad_unit_x"] = gux
        out["grad_unit_y"] = guy
        out["grad_x_per_um"] = gx
        out["grad_y_per_um"] = gy
        out["grad_strength_dimless"] = g
        out["particle_scale_receptor_contrast_percent"] = 100.0 * g
        out["local_receptor_density"] = rho
        out["step_unit_x"] = ux
        out["step_unit_y"] = uy
        out["ci"] = np.clip(ci, -1.0, 1.0)
        out["uphill"] = out["ci"] > 0
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_gradient_qc(fields: dict[str, GradientField], qc_dir: Path) -> pd.DataFrame:
    rows = []
    for movie, field in fields.items():
        vals = field.grad_strength_dimless[np.isfinite(field.grad_strength_dimless)]
        rows.append(
            {
                "movie": movie,
                "grad_strength_median": float(np.median(vals)),
                "grad_strength_q75": float(np.quantile(vals, 0.75)),
                "grad_strength_q90": float(np.quantile(vals, 0.90)),
                "grad_strength_q99": float(np.quantile(vals, 0.99)),
                "particle_scale_contrast_percent_q90": float(
                    100.0 * np.quantile(vals, 0.90)
                ),
                "particle_scale_contrast_percent_q99": float(
                    100.0 * np.quantile(vals, 0.99)
                ),
                "receptor_smooth_median": float(np.median(field.receptor_smooth)),
            }
        )
    qc = pd.DataFrame(rows)
    qc.to_csv(qc_dir / "gradient_field_qc.csv", index=False)
    return qc
