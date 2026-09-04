"""End-to-end checks for the numerical claims reported in the manuscript."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.population import (
    demand_sweep_allocation,
    projected_population_density,
    reader_cleaver_coordinate,
)
from analysis.expt_gradient import (
    assign_fast_component_probability,
    fit_untreated_diffusivity_mixture,
    trajectory_reversal_null,
    summarize_fast_component,
    within_replicate_label_permutation,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "figure_source"
ANALYSIS_INPUT = ROOT / "data" / "analysis_input"


def _semicolon_values(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in str(value).split(";")], dtype=float)


def test_experimental_ci_and_arrival_use_the_same_three_biological_replicates():
    tests = pd.read_csv(SOURCE / "experimental" / "statistical_tests.csv")
    families = {
        "conditioned_high_gradient_chemotactic_index": {
            "SBA_noNAI": (0.20254511971097866, 0.09887690740149184),
            "SBA_NAI": (0.15950313840152539, 0.06533808419526026),
        },
        "conditioned_high_gradient_net_first_arrival_score": {
            "SBA_noNAI": (0.006992279971070298, 0.0005497206212087043),
            "SBA_NAI": (0.002330175644592138, 0.00018741202073776727),
        },
    }
    for metric, expected in families.items():
        rows = tests.loc[
            tests["metric"].eq(metric)
            & tests["condition"].isin(expected)
            & tests["mean"].notna()
        ]
        assert set(rows["condition"]) == set(expected)
        for row in rows.itertuples(index=False):
            values = _semicolon_values(row.biological_replicate_values)
            assert values.size == 3
            assert row.n_biological_replicates == 3
            assert np.mean(values) == pytest.approx(
                expected[row.condition][0], abs=5.0e-9
            )
            assert np.std(values, ddof=1) == pytest.approx(
                expected[row.condition][1], abs=5.0e-9
            )
            assert row.high_gradient_cutoff_percent == pytest.approx(
                9.1966917756456
            )

    ci_pair = tests.loc[
        tests["metric"].eq("conditioned_high_gradient_chemotactic_index")
        & tests["condition"].eq("paired_NAI_minus_untreated")
        & tests["mean_delta"].notna()
    ].iloc[0]
    assert ci_pair["p_recording_stratified"] == pytest.approx(0.1694415279236038)

    arrival_pair = tests.loc[
        tests["metric"].eq("conditioned_high_gradient_net_first_arrival_score")
        & tests["condition"].eq("paired_NAI_minus_untreated")
        & tests["mean_delta"].notna()
    ].iloc[0]
    deltas = _semicolon_values(arrival_pair["biological_replicate_deltas"])
    assert np.all(deltas < 0.0)
    assert arrival_pair["p_t_two_sided"] == pytest.approx(0.002083, abs=1.0e-6)
    assert arrival_pair["p_recording_stratified"] == pytest.approx(
        0.0096995150242487
    )


def test_control_diffusivity_classification_is_shared_between_conditions():
    table = pd.read_csv(
        SOURCE / "experimental" / "panel_f_inset_fast_component_statistics.csv"
    )
    row = table.loc[table["metric"].eq("mean_fast_component_posterior")].iloc[0]
    assert row["n_biological_replicates"] == 3
    assert row["biological_replicate_mean_noNAI"] == pytest.approx(0.543162)
    assert row["biological_replicate_mean_NAI"] == pytest.approx(0.279967)
    assert np.all(_semicolon_values(row["biological_replicate_deltas"]) < 0.0)
    assert row["p_t_two_sided"] == pytest.approx(0.0457, abs=1.0e-6)
    assert row[
        "p_exact_biological_replicate_blocked_movie_permutation"
    ] == pytest.approx(1.998001998001998e-5)


def test_experimental_track_tables_rebuild_reported_statistics():
    base = ANALYSIS_INPUT / "experimental"
    expected = {
        "ci": (0.20254511971097866, 0.15950313840152539, 0.1694415279236038),
        "arrival": (0.006992279971070298, 0.002330175644592138, 0.0096995150242487),
    }
    specs = {
        "ci": (
            "ci_track_values.csv",
            "ci_recording_metrics.csv",
            "conditioned_ci",
        ),
        "arrival": (
            "arrival_track_values.csv",
            "arrival_recording_metrics.csv",
            "conditioned_net_first_passage_score_s",
        ),
    }
    for key, (track_name, recording_name, metric_name) in specs.items():
        tracks = pd.read_csv(base / track_name)
        support = pd.read_csv(base / recording_name)
        support = support.loc[
            support["conditioned_included_in_date_average"].astype(bool),
            ["date", "condition", "movie"],
        ]
        selected = tracks.merge(support, on=["date", "condition", "movie"])
        recording = selected.groupby(
            ["date", "condition", "movie"], as_index=False
        )["value"].mean()
        replicate = recording.groupby(["date", "condition"], as_index=False)[
            "value"
        ].mean()
        means = replicate.groupby("condition")["value"].mean()
        assert means["SBA_noNAI"] == pytest.approx(expected[key][0])
        assert means["SBA_NAI"] == pytest.approx(expected[key][1])
        test = within_replicate_label_permutation(
            recording,
            value_column="value",
            untreated_condition="SBA_noNAI",
            treated_condition="SBA_NAI",
            alternative="less",
            iterations=20_000,
            seed=417 + sum(ord(character) for character in metric_name),
        )
        assert test["exact"] is False
        assert test["p_value"] == pytest.approx(expected[key][2])


def test_experimental_main_panels_use_the_reported_local_contrast_axis():
    ci = pd.read_csv(SOURCE / "experimental" / "panel_e_ci_vs_receptor_contrast.csv")
    arrival = pd.read_csv(
        SOURCE / "experimental" / "panel_f_arrival_vs_receptor_contrast.csv"
    )
    assert len(ci) == len(arrival) == 30
    keys = ["condition", "gradient_bin", "contrast_threshold_percent"]
    pd.testing.assert_frame_equal(ci[keys], arrival[keys])
    assert set(ci["metric"]) == {"chemotactic_index"}
    assert set(ci["n_biological_replicates"]) == {3}
    assert set(arrival["n_biological_replicates"]) == {3}

    strongest = ci.loc[np.isclose(ci["contrast_threshold_percent"], 9.1966917756456)]
    observed = strongest.set_index("condition")["mean_value"]
    assert observed["SBA_noNAI"] == pytest.approx(0.20254511971097866)
    assert observed["SBA_NAI"] == pytest.approx(0.15950313840152539)


def test_experimental_metrics_share_recordings_cutoff_and_treatment_support():
    base = ANALYSIS_INPUT / "experimental"
    recording_keys: dict[str, set[tuple[str, str, str]]] = {}
    for name in ("ci", "arrival"):
        tracks = pd.read_csv(base / f"{name}_track_values.csv")
        recordings = pd.read_csv(base / f"{name}_recording_metrics.csv")
        included = recordings.loc[
            recordings["conditioned_included_in_date_average"].astype(bool)
        ]
        keys = ["date", "condition", "movie"]
        recording_keys[name] = set(
            included[keys].astype(str).itertuples(index=False, name=None)
        )
        track_keys = set(tracks[keys].astype(str).itertuples(index=False, name=None))
        assert track_keys == recording_keys[name]
        assert not tracks["global_track_uid"].duplicated().any()
        assert np.isfinite(tracks["value"]).all()
        assert np.allclose(included["conditioned_top_fraction"], 0.10)
        assert np.allclose(
            included["conditioned_high_gradient_cutoff_percent"],
            9.1966917756456,
        )
        assert included.groupby("date")["condition"].apply(set).to_dict() == {
            "2025-11-17": {"SBA_noNAI", "SBA_NAI"},
            "2025-11-24": {"SBA_noNAI", "SBA_NAI"},
            "2025-12-03": {"SBA_noNAI", "SBA_NAI"},
        }
    assert recording_keys["ci"] == recording_keys["arrival"]
    arrival = pd.read_csv(base / "arrival_recording_metrics.csv")
    assert np.allclose(arrival["conditioned_passage_distance_um"], 0.75)


def test_experimental_cue_settings_and_reversal_null_match_the_reported_analysis():
    import yaml

    settings = yaml.safe_load(
        (ROOT / "config" / "experimental_gradient_analysis.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert settings["physics"] == {
        "gradient_smoothing_um": 0.25,
        "receptor_cue_radius_um": 0.5,
        "receptor_cue_angles": 96,
        "ci_first_passage_displacement_um": 0.5,
        "arrival_distance_um": 0.75,
        "retained_receptor_contrast_fraction": 0.10,
    }

    tracks = pd.read_csv(
        ANALYSIS_INPUT / "experimental" / "ci_track_values.csv"
    )
    tracks = tracks.loc[tracks["condition"].eq("SBA_noNAI")].rename(
        columns={"value": "step_ci"}
    )
    null = trajectory_reversal_null(
        tracks,
        iterations=20_000,
        seed=417 + 901,
    )["step_ci"].to_numpy(float)
    archived = pd.read_csv(SOURCE / "experimental" / "statistical_tests.csv")
    row = archived.loc[
        archived["metric"].eq("conditioned_high_gradient_chemotactic_index")
        & archived["condition"].eq("SBA_noNAI")
        & archived["null_mean"].notna()
    ].iloc[0]
    assert np.mean(null) == pytest.approx(row["null_mean"], abs=5.0e-12)
    assert np.quantile(null, 0.025) == pytest.approx(row["null_ci_lo"], abs=5.0e-12)
    assert np.quantile(null, 0.975) == pytest.approx(row["null_ci_hi"], abs=5.0e-12)
    p_value = (1 + np.sum(null >= float(row["observed"]))) / (len(null) + 1)
    assert p_value == pytest.approx(row["p_randomization_greater"])


def test_control_track_table_rebuilds_fixed_mixture_and_exact_permutation():
    tracks = pd.read_csv(
        ANALYSIS_INPUT / "experimental" / "control_diffusivity_track_values.csv"
    )
    model, fast_component = fit_untreated_diffusivity_mixture(tracks)
    classified = assign_fast_component_probability(tracks, model, fast_component)
    recording, replicate = summarize_fast_component(classified)
    means = replicate.groupby("condition")["fast_component_probability"].mean()
    assert means["noSBA_noNAI"] == pytest.approx(0.543162, abs=5.0e-7)
    assert means["noSBA_NAI"] == pytest.approx(0.279967, abs=5.0e-7)
    test = within_replicate_label_permutation(
        recording,
        value_column="fast_component_probability",
        untreated_condition="noSBA_noNAI",
        treated_condition="noSBA_NAI",
        alternative="less",
    )
    assert test["exact"] is True
    assert test["n_permutations"] == 600_600
    assert test["p_value"] == pytest.approx(1.998001998001998e-5)


def test_fast_component_summary_uses_a_treatment_comparison_only():
    from scripts.rebuild_experimental_statistics import _condition_rows

    replicates = pd.DataFrame(
        {
            "date": ["r1", "r2", "r3"],
            "condition": ["untreated"] * 3,
            "value": [0.2, 0.4, 0.3],
        }
    )
    rows = _condition_rows(
        "control_fast_component_probability",
        replicates,
        compare_with_zero=False,
    )
    assert np.isnan(rows[0]["p_one_sample_greater_than_zero"])


def test_gradient_tables_match_the_submitted_source_contract():
    for name in (
        "panel_e_bias_vs_density_contrast.csv",
        "panel_f_arrival_vs_density_contrast.csv",
        "panel_g_guidance_vs_attachment_friction.csv",
        "panel_h_arrival_vs_attachment_friction.csv",
    ):
        table = pd.read_csv(SOURCE / "gradient_3d" / name)
        theory = table.loc[table["series"].eq("mean_field_theory")]
        assert not theory.empty
        assert set(theory["architecture"]) == {"polarized", "mixed"}
        assert set(theory["K_C"]) == {0.0, 10.0}
        assert np.allclose(theory["cleavage_rate_scale"], 0.002)
        assert np.allclose(theory["path_ci_correction_coefficient"], 1.0)
        assert np.all(np.isfinite(theory["mean"]))

    ci = pd.read_csv(
        SOURCE / "gradient_3d" / "panel_e_bias_vs_density_contrast.csv"
    )
    zero = ci.loc[
        ci["series"].eq("mean_field_theory")
        & np.isclose(ci["gradient_scale"], 1.0)
    ]
    assert np.allclose(zero["mean"], 0.0)


def test_gradient_affinity_axis_uses_the_manuscript_definition():
    """Panels g,h use one shared mean-attachments-over-K_D coordinate."""

    coordinates = []
    for name in (
        "panel_g_guidance_vs_attachment_friction.csv",
        "panel_h_arrival_vs_attachment_friction.csv",
    ):
        table = pd.read_csv(SOURCE / "gradient_3d" / name)
        coordinate = table["mean_attachments_over_K_D"].to_numpy(float)
        assert np.all(np.isfinite(coordinate))
        assert np.all(coordinate > 0.0)
        assert np.allclose(table["plotted_axis_coordinate"], coordinate)
        coordinates.append(
            table[["architecture", "K_C", "K_D", "mean_attachments_over_K_D"]]
            .drop_duplicates()
            .sort_values(["architecture", "K_C", "K_D"])
            .reset_index(drop=True)
        )
    pd.testing.assert_frame_equal(coordinates[0], coordinates[1])


def test_population_functions_and_displayed_distributions_are_consistent():
    maps = pd.read_csv(SOURCE / "population" / "panel_ab_function_maps.csv")
    assert maps.groupby("architecture").size().nunique() == 1
    scores = [
        "exploration_range_score",
        "gradient_guidance_score",
        "escape_score",
        "exploit_score",
        "passive_score",
    ]
    assert np.all(np.isfinite(maps[scores]))
    assert np.all((maps[scores] >= 0.0) & (maps[scores] <= 1.0))

    sweep = pd.read_csv(SOURCE / "population" / "panel_c_demand_sweep.csv")
    fractions = sweep[["explore_fraction", "escape_fraction", "exploit_fraction"]]
    assert np.allclose(fractions.sum(axis=1), 1.0)
    balanced = sweep.iloc[(sweep["explore_to_exploit_demand_ratio"] - 1.0).abs().argmin()]
    contributions = balanced[
        ["explore_contribution", "escape_contribution", "exploit_contribution"]
    ].to_numpy(float)
    assert np.ptp(contributions) < 1.0e-12

    distributions = pd.read_csv(
        SOURCE / "population" / "panel_c_population_distributions.csv"
    )
    for ratio, group in distributions.groupby("explore_to_exploit_demand_ratio"):
        x = group["reader_cleaver_coordinate"].to_numpy(float)
        for role in ("explore", "escape", "exploit"):
            area = np.trapz(group[f"{role}_density"].to_numpy(float), x)
            assert area == pytest.approx(float(group[f"{role}_fraction"].iloc[0]))
    explore_biased = distributions.loc[
        distributions["explore_to_exploit_demand_ratio"].eq(20.0)
    ].iloc[0]
    assert explore_biased["explore_fraction"] > explore_biased["escape_fraction"]
    exploit_biased = distributions.loc[
        distributions["explore_to_exploit_demand_ratio"].eq(0.05)
    ].iloc[0]
    assert exploit_biased["exploit_fraction"] > exploit_biased["escape_fraction"]


def test_population_guidance_interpolation_is_audited_on_held_out_states():
    validation = pd.read_csv(
        ANALYSIS_INPUT / "population" / "gradient_ci_emulator_validation.csv"
    )
    assert set(validation["validation_bin"]) == {
        "K_C_le_10",
        "K_C_10_to_50",
        "K_C_50_to_250_edge",
    }
    assert len(validation) == 72
    assert validation["absolute_error"].max() < 1.5e-3

    extrema = pd.read_csv(
        ANALYSIS_INPUT / "population" / "gradient_ci_extremum_evaluations.csv"
    )
    assert extrema.groupby("architecture").size().to_dict() == {
        "mixed": 16,
        "polarized": 16,
    }
    assert not extrema[
        ["architecture", "phi_index", "chi_C_index", "allocation_index"]
    ].duplicated().any()
    assert extrema["absolute_interpolation_difference"].max() < 1.0e-8


def test_population_map_uses_the_uniform_transport_kernel_unchanged():
    """Integer map states must reproduce the section-2 transport calculation."""

    from analysis.population.maps import _inputs as population_inputs
    from analysis.population.transport_mean_field import (
        OUTPUT_NAMES,
        evaluate_state,
        params_to_array,
    )
    from analysis.uniform_3d import load_geometries, load_inputs, predict

    inputs = load_inputs()
    geometries = load_geometries(inputs)
    population_parameters = params_to_array(population_inputs())
    output = {name: index for index, name in enumerate(OUTPUT_NAMES)}
    fields = {
        "speed": "speed",
        "persistence_time": "persistence_time",
        "active_diffusivity": "active_diffusivity",
        "effective_diffusivity": "effective_diffusivity",
        "mobile_fraction": "mobile_fraction",
    }
    for pattern_code, architecture in enumerate(("polarized", "mixed")):
        for n_binders in (4, 10, 16):
            for k_d in (1.0, 10.0, 100.0, 500.0):
                for k_c in (0.0, 10.0):
                    uniform = predict(
                        architecture,
                        k_d,
                        k_c,
                        n_binders,
                        inputs,
                        geometries,
                    )
                    population = evaluate_state(
                        pattern_code,
                        k_d,
                        k_c,
                        float(n_binders),
                        population_parameters,
                    )
                    for population_name, uniform_name in fields.items():
                        assert population[output[population_name]] == pytest.approx(
                            float(uniform[uniform_name]), rel=3.0e-10, abs=1.0e-12
                        )


def test_population_allocation_recalculates_panel_c():
    states = pd.read_csv(
        SOURCE / "population" / "panel_c_representative_states.csv"
    ).set_index("function")
    capacities = {
        role: float(states.loc[role, f"{role}_score"])
        for role in ("explore", "escape", "exploit")
    }
    archived = pd.read_csv(SOURCE / "population" / "panel_c_demand_sweep.csv")
    rebuilt = pd.DataFrame(
        demand_sweep_allocation(
            capacities,
            archived["explore_to_exploit_demand_ratio"].to_numpy(float),
        )
    )
    for column in (
        "explore_fraction",
        "escape_fraction",
        "exploit_fraction",
        "explore_contribution",
        "escape_contribution",
        "exploit_contribution",
    ):
        assert np.allclose(rebuilt[column], archived[column], atol=1.0e-12)

    coordinate = pd.read_csv(
        SOURCE / "population" / "panel_c_population_distributions.csv"
    )
    shared_center = reader_cleaver_coordinate(
        float(states.loc["explore", "phi_b"]),
        float(states.loc["explore", "chi_C"]),
    )
    exploit_center = reader_cleaver_coordinate(
        float(states.loc["exploit", "phi_b"]),
        float(states.loc["exploit", "chi_C"]),
    )
    for ratio, group in coordinate.groupby("explore_to_exploit_demand_ratio"):
        row = archived.iloc[
            np.argmin(
                np.abs(
                    archived["explore_to_exploit_demand_ratio"].to_numpy(float)
                    - float(ratio)
                )
            )
        ]
        fractions = {
            role: float(row[f"{role}_fraction"])
            for role in ("explore", "escape", "exploit")
        }
        rebuilt_density = projected_population_density(
            fractions,
            shared_center,
            exploit_center,
            group["reader_cleaver_coordinate"].to_numpy(float),
        )
        for column in (
            "explore_density",
            "escape_density",
            "exploit_density",
            "total_density",
        ):
            assert np.allclose(
                rebuilt_density[column], group[column].to_numpy(float), atol=1.0e-12
            )


def test_surface_range_conditions_follow_the_stated_regime_criteria():
    audit = pd.read_csv(
        ANALYSIS_INPUT / "surface_2d" / "surface_range_condition_audit.csv"
    )
    selected = (
        audit["K_C"].gt(0.0)
        & audit["detached_fraction"].ge(0.8)
        & audit["median_path_in_persistence_lengths"].ge(30.0)
        & audit["recurrence_range_screen"].lt(
            0.9 * audit["finite_time_free_range"]
        )
    )
    assert len(audit) == 144
    assert int(selected.sum()) == 38
    assert np.array_equal(
        selected.to_numpy(),
        audit["included_in_repeated_return_comparison"].astype(bool).to_numpy(),
    )
    keys = ["architecture", "K_C", "K_D", "n_binders"]
    plotted = pd.read_csv(
        SOURCE / "surface_2d" / "panel_d_recurrence_design_points.csv"
    )
    assert audit.loc[selected, keys].sort_values(keys).reset_index(drop=True).equals(
        plotted[keys].sort_values(keys).reset_index(drop=True)
    )
