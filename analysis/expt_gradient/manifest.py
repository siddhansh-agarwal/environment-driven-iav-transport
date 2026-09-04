from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CONDITIONS = ("SBA_noNAI", "SBA_NAI", "noSBA_noNAI", "noSBA_NAI")
SENSING_CONDITIONS = ("SBA_noNAI", "SBA_NAI")


@dataclass(frozen=True)
class ReplicateCondition:
    date: str
    raw_date_dir: str
    condition: str
    nai_dose_uM: float
    movies: tuple[str, ...]
    input_dir: Path
    output_dir: Path
    config_path: Path

    @property
    def role(self) -> str:
        return "sensing" if self.condition in SENSING_CONDITIONS else "motility"


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


def iter_replicate_conditions(
    manifest: dict[str, Any], project_root: Path
) -> list[ReplicateCondition]:
    raw_root = project_root / manifest["raw_data_root"]
    processed_root = project_root / manifest["processed_data_root"]
    config_root = project_root / manifest.get("config_root", "config/experimental")
    records: list[ReplicateCondition] = []
    condition_metadata = manifest.get("condition_metadata", {}) or {}
    for replicate in manifest["replicates"]:
        date = str(replicate["date"])
        raw_date_dir = str(replicate["raw_date_dir"])
        date_slug = date
        for condition in CONDITIONS:
            info = replicate["conditions"][condition]
            default_dose = condition_metadata.get(condition, {}).get("nai_dose_uM", 0)
            nai_dose_uM = float(info.get("nai_dose_uM", default_dose))
            movies = tuple(str(movie) for movie in info.get("movies", []))
            records.append(
                ReplicateCondition(
                    date=date,
                    raw_date_dir=raw_date_dir,
                    condition=condition,
                    nai_dose_uM=nai_dose_uM,
                    movies=movies,
                    input_dir=raw_root / raw_date_dir / condition,
                    output_dir=processed_root / "replicates" / date_slug / condition,
                    config_path=config_root / f"{date_slug}_{condition}.yaml",
                )
            )
    return records


def validate_manifest_records(
    records: list[ReplicateCondition],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    movie_uids: set[str] = set()
    for record in records:
        expected = set(record.movies)
        actual = {path.stem for path in record.input_dir.glob("*.nd2")}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if not record.input_dir.exists():
            errors.append(f"missing condition directory: {record.input_dir}")
        if missing:
            errors.append(
                f"{record.date} {record.condition}: missing ND2 files {missing}"
            )
        if extra:
            errors.append(
                f"{record.date} {record.condition}: unexpected ND2 files {extra}"
            )
        for movie in record.movies:
            movie_uid = f"{record.date}::{record.condition}::{movie}"
            if movie_uid in movie_uids:
                errors.append(f"duplicate movie_uid: {movie_uid}")
            movie_uids.add(movie_uid)
        rows.append(
            {
                "date": record.date,
                "condition": record.condition,
                "role": record.role,
                "nai_dose_uM": record.nai_dose_uM,
                "input_dir": str(record.input_dir),
                "output_dir": str(record.output_dir),
                "n_manifest_movies": len(record.movies),
                "n_nd2_files": len(actual),
                "movies": ",".join(record.movies),
            }
        )
    return rows, errors


def write_replicate_configs(
    records: list[ReplicateCondition],
    base_config_path: str | Path,
    *,
    particle_label: str,
    manifest: dict[str, Any] | None = None,
) -> None:
    with Path(base_config_path).open("r") as handle:
        base = yaml.safe_load(handle)
    tracking_policy = (manifest or {}).get("tracking_policy", {}) or {}
    tracking_defaults = tracking_policy.get("default_tracking", {}) or {}
    quality_defaults = tracking_policy.get("quality_threshold_defaults", {}) or {}
    condition_thresholds = quality_defaults.get("by_date_condition", {}) or {}
    movie_thresholds = quality_defaults.get("by_movie", {}) or {}
    parameter_overrides = tracking_policy.get("parameter_overrides", {}) or {}
    parameter_condition_overrides = (
        parameter_overrides.get("by_date_condition", {}) or {}
    )
    parameter_movie_overrides = parameter_overrides.get("by_movie", {}) or {}
    for record in records:
        config = yaml.safe_load(yaml.safe_dump(base))
        config["input_dir"] = str(record.input_dir.resolve())
        config["output_dir"] = str(record.output_dir.resolve())
        config.setdefault("experiment", {})
        config["experiment"]["condition"] = record.condition
        config["experiment"]["replicate_id"] = record.date
        config["experiment"]["particle_label"] = particle_label
        config["experiment"]["receptor_channel"] = "647"
        config["experiment"]["tracking_channel"] = "561"
        config["experiment"]["date"] = record.date
        config["experiment"]["analysis_role"] = record.role
        config["experiment"]["nai_dose_uM"] = record.nai_dose_uM
        if tracking_policy:
            config["tracking_policy"] = {
                "name": tracking_policy.get("name", ""),
                "rationale": tracking_policy.get("rationale", ""),
            }
        if tracking_defaults:
            config.setdefault("tracking", {}).update(tracking_defaults)
        default_threshold = quality_defaults.get("default")
        if default_threshold is not None:
            config.setdefault("tracking", {})["quality_threshold"] = float(
                default_threshold
            )
        date_condition_threshold = condition_thresholds.get(str(record.date), {}).get(
            str(record.condition)
        )
        if date_condition_threshold is not None:
            config.setdefault("tracking", {})["quality_threshold"] = float(
                date_condition_threshold
            )
        per_movie = movie_thresholds.get(str(record.date), {}).get(
            str(record.condition), {}
        )
        if per_movie:
            overrides = config.setdefault("tracking_overrides", {}).setdefault(
                "movies", {}
            )
            for movie, threshold in per_movie.items():
                if str(movie) in record.movies:
                    overrides.setdefault(str(movie), {})["quality_threshold"] = float(
                        threshold
                    )
        date_condition_params = parameter_condition_overrides.get(
            str(record.date), {}
        ).get(str(record.condition), {})
        if date_condition_params:
            config.setdefault("tracking", {}).update(date_condition_params)
        per_movie_params = parameter_movie_overrides.get(str(record.date), {}).get(
            str(record.condition), {}
        )
        if per_movie_params:
            overrides = config.setdefault("tracking_overrides", {}).setdefault(
                "movies", {}
            )
            for movie, movie_params in per_movie_params.items():
                if str(movie) in record.movies:
                    overrides.setdefault(str(movie), {}).update(movie_params)
        record.config_path.parent.mkdir(parents=True, exist_ok=True)
        with record.config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
