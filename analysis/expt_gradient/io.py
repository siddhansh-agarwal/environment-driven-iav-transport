from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


def _nd2_reader():
    """Import the optional microscopy reader only when raw ND2 data are used."""
    from nd2reader import ND2Reader

    return ND2Reader


@dataclass(frozen=True)
class MovieRecord:
    movie: str
    nd2_path: Path


@dataclass
class MovieMetadata:
    movie: str
    channels: list[str]
    pixel_size_um: float
    timestamps_s: np.ndarray
    n_frames: int
    height_px: int
    width_px: int


def discover_movies(input_dir: str | Path) -> list[MovieRecord]:
    records = [
        MovieRecord(path.stem, path) for path in sorted(Path(input_dir).glob("*.nd2"))
    ]
    if not records:
        raise FileNotFoundError(f"No ND2 files found in {input_dir}")
    return records


def channel_index(channels: list[str], requested: str) -> int:
    for i, channel in enumerate(channels):
        if channel == requested:
            return i
    for i, channel in enumerate(channels):
        if requested in channel:
            return i
    raise ValueError(f"Channel {requested!r} not found in {channels}")


def read_metadata(record: MovieRecord) -> MovieMetadata:
    with _nd2_reader()(str(record.nd2_path)) as nd2:
        channels = list(nd2.metadata.get("channels", []))
        pixel_size = float(nd2.metadata.get("pixel_microns", 1.0))
        timestamps = np.asarray(nd2.get_timesteps(), dtype=float) / 1000.0
        n_frames = int(nd2.sizes.get("t", 1))
        frame0 = np.asarray(nd2[0])
    return MovieMetadata(
        movie=record.movie,
        channels=channels,
        pixel_size_um=pixel_size,
        timestamps_s=timestamps[:n_frames],
        n_frames=n_frames,
        height_px=int(frame0.shape[0]),
        width_px=int(frame0.shape[1]),
    )


def iter_channel_frames(record: MovieRecord, channel: str):
    with _nd2_reader()(str(record.nd2_path)) as nd2:
        channels = list(nd2.metadata.get("channels", []))
        nd2.default_coords["c"] = channel_index(channels, channel)
        n_frames = int(nd2.sizes.get("t", 1))
        for frame in range(n_frames):
            yield frame, np.asarray(nd2[frame], dtype=np.float32)


def read_channel_frame(record: MovieRecord, channel: str, frame: int = 0) -> np.ndarray:
    with _nd2_reader()(str(record.nd2_path)) as nd2:
        channels = list(nd2.metadata.get("channels", []))
        nd2.default_coords["c"] = channel_index(channels, channel)
        frame = min(frame, int(nd2.sizes.get("t", 1)) - 1)
        return np.asarray(nd2[frame], dtype=np.float64)


def illumination_correct(frame: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    frame_float = frame.astype(np.float64)
    illum = ndimage.gaussian_filter(frame_float, sigma=sigma)
    illum = np.maximum(illum, np.percentile(illum, 1))
    correction = np.mean(illum) / illum
    correction = np.clip(correction, 0.1, 10.0)
    return frame_float * correction
