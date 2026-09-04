from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis.expt_gradient import (
    add_gradient_frame_displacements,
    assign_fast_component_probability,
    fit_untreated_diffusivity_mixture,
    local_first_arrival_by_track,
    trajectory_reversal_null,
    shared_receptor_contrast_threshold,
    summarize_fast_component,
    summarize_first_arrival,
    summarize_step_ci,
    within_replicate_label_permutation,
)
from analysis.expt_gradient.gradients import first_harmonic_cue_field
from analysis.expt_gradient.events import exact_frame_events, first_passage_events
from analysis.gradient_3d import (
    binding_drift,
    binding_free_energy_force,
    binding_order,
    normalized_dense_arrival_rate,
    path_chemotactic_index,
    path_chemotactic_index_until,
)
from analysis.population import (
    OUTPUT_NAMES,
    attachment_probability,
    cleavage_probability,
    escape_probability,
    evaluate_state,
    optimal_allocation,
    params_to_array,
)
from analysis.surface_2d import predict_surface_range, recurrence_prediction
from analysis.uniform_3d import load_geometries as load_uniform_geometries
from analysis.uniform_3d import load_inputs as load_uniform_inputs
from analysis.uniform_3d import predict as predict_uniform_transport


ROOT = Path(__file__).resolve().parents[1]


def test_gradient_observables_have_expected_signs():
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert path_chemotactic_index(positions, np.asarray([1.0, 0.0])) == pytest.approx(
        1.0
    )
    assert normalized_dense_arrival_rate(
        np.asarray([0.0, 2.0, 4.0]),
        np.asarray([0.0, 0.4, 1.1]),
        dense_boundary=1.0,
        sparse_boundary=-1.0,
        observation_time=5.0,
    ) == pytest.approx(1.25)


def test_fixed_time_path_index_interpolates_the_endpoint():
    times = np.asarray([0.0, 1.0, 3.0])
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    observed = path_chemotactic_index_until(
        times,
        positions,
        np.asarray([1.0, 0.0]),
        query_time=2.0,
    )
    assert observed == pytest.approx(0.5)


def test_experimental_signed_first_arrival_matches_manuscript_definition():
    events = pd.DataFrame(
        {
            "date": ["r1"] * 6,
            "condition": ["untreated"] * 6,
            "movie": ["m1"] * 6,
            "track_uid": ["dense"] * 2 + ["sparse"] + ["neither"] * 3,
            "frame_start": [0, 1, 0, 0, 1, 3],
            "frame_end": [1, 2, 1, 1, 2, 4],
            "time_interval_s": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            "particle_scale_receptor_contrast_percent": [10.0] * 6,
            "gradient_parallel_displacement_um": [0.4, 0.4, -0.8, 0.1, 0.1, 0.1],
        }
    )
    scores = local_first_arrival_by_track(
        events,
        receptor_contrast_threshold=5.0,
        arrival_distance_um=0.75,
    ).set_index("track_uid")
    assert scores.loc["dense", "net_first_arrival_score_s"] == pytest.approx(0.5)
    assert scores.loc["sparse", "net_first_arrival_score_s"] == pytest.approx(-0.5)
    assert scores.loc["neither", "net_first_arrival_score_s"] == pytest.approx(0.0)
    assert scores.loc["neither", "n_episodes"] == 2

    recordings, replicates = summarize_first_arrival(scores.reset_index())
    assert len(recordings) == 1
    assert len(replicates) == 1
    assert recordings.loc[0, "net_first_arrival_score_s"] == pytest.approx(0.0)


def test_experimental_steps_are_resolved_in_the_start_gradient_frame():
    events = pd.DataFrame(
        {
            "dx_um": [3.0],
            "dy_um": [4.0],
            "grad_unit_x": [0.0],
            "grad_unit_y": [1.0],
        }
    )
    resolved = add_gradient_frame_displacements(events)
    assert resolved.loc[0, "gradient_parallel_displacement_um"] == pytest.approx(4.0)
    assert resolved.loc[0, "gradient_perpendicular_displacement_um"] == pytest.approx(
        -3.0
    )


def test_experimental_ci_uses_one_shared_cutoff_and_stated_hierarchy():
    events = pd.DataFrame(
        {
            "date": ["r1"] * 4 + ["r2"] * 4,
            "condition": ["untreated"] * 8,
            "movie": ["m1", "m1", "m2", "m2"] * 2,
            "track_uid": ["a", "a", "b", "c", "d", "d", "e", "f"],
            "particle_scale_receptor_contrast_percent": np.arange(1.0, 9.0),
            "ci": [1.0, 1.0, -1.0, 1.0, 0.5, 0.5, -0.5, 0.5],
        }
    )
    cutoff = shared_receptor_contrast_threshold(events, retained_fraction=0.5)
    assert cutoff == pytest.approx(4.5)
    track, recording, replicate = summarize_step_ci(
        events,
        receptor_contrast_threshold=cutoff,
    )
    assert len(track) == 3
    assert len(recording) == 2
    assert replicate.loc[0, "step_ci"] == pytest.approx(0.25)

    null = trajectory_reversal_null(track, iterations=12, seed=4)
    assert len(null) == 12
    assert set(null["condition"]) == {"untreated"}


def test_control_diffusivity_uses_one_untreated_mixture_for_both_conditions():
    rows = []
    distributions = {
        "noSBA_noNAI": np.r_[np.linspace(-4.2, -3.8, 8), np.linspace(-2.2, -1.8, 8)],
        "noSBA_NAI": np.r_[np.linspace(-4.2, -3.8, 12), np.linspace(-2.2, -1.8, 4)],
    }
    for condition, values in distributions.items():
        for replicate in ("r1", "r2"):
            for movie_index in range(2):
                for i, value in enumerate(values):
                    rows.append(
                        {
                            "date": replicate,
                            "condition": condition,
                            "movie": f"{replicate}_{condition}_{movie_index}",
                            "track_uid": f"{replicate}_{condition}_{movie_index}_{i}",
                            "d_eff_um2_s": 10.0**value,
                            "passes_diffusion_qc": True,
                        }
                    )
    diffusion = pd.DataFrame(rows)
    model, fast_component = fit_untreated_diffusivity_mixture(diffusion)
    classified = assign_fast_component_probability(diffusion, model, fast_component)
    recording, replicate = summarize_fast_component(classified)
    assert len(recording) == 8
    assert len(replicate) == 4
    means = replicate.groupby("condition")["fast_component_probability"].mean()
    assert means["noSBA_noNAI"] > means["noSBA_NAI"]

    test = within_replicate_label_permutation(
        recording,
        value_column="fast_component_probability",
        untreated_condition="noSBA_noNAI",
        treated_condition="noSBA_NAI",
    )
    assert test["n_replicates"] == 2
    assert test["observed_treated_minus_untreated"] < 0.0


def test_receptor_cue_uses_the_stated_ring_harmonic():
    y, x = np.mgrid[-50:51, -50:51]
    image = 100.0 + 2.0 * x
    mean, first_x, first_y = first_harmonic_cue_field(
        image,
        radius_pixels=5.0,
        n_angles=96,
    )
    center = (50, 50)
    assert mean[center] == pytest.approx(100.0, rel=1.0e-12)
    assert first_x[center] == pytest.approx(5.0, rel=1.0e-12)
    assert first_y[center] == pytest.approx(0.0, abs=1.0e-12)
    contrast = 4.0 * np.hypot(first_x[center], first_y[center]) / mean[center]
    assert contrast == pytest.approx(0.2, rel=1.0e-12)


def test_event_tables_preserve_replicate_and_condition_identity():
    trajectories = pd.DataFrame(
        {
            "date": ["r1"] * 3,
            "condition": ["untreated"] * 3,
            "movie": ["m1"] * 3,
            "particle": [7] * 3,
            "frame": [0, 1, 2],
            "time_s": [0.0, 1.0, 2.0],
            "x_um": [0.0, 0.3, 0.8],
            "y_um": [0.0, 0.0, 0.0],
        }
    )
    exact = exact_frame_events(trajectories)
    passage = first_passage_events(trajectories, body_length_um=0.5)
    for events in (exact, passage):
        assert set(events["date"]) == {"r1"}
        assert set(events["condition"]) == {"untreated"}
        assert set(events["global_track_uid"]) == {"r1::untreated::m1::7"}


def test_ligand_layouts_are_fixed_and_preserve_counts():
    from src.parameters import distribute_ligands

    polarized = distribute_ligands(20, 10, "polarized")
    mixed = distribute_ligands(20, 10, "mixed")
    np.testing.assert_array_equal(
        polarized,
        np.asarray([True] * 10 + [False] * 10, dtype=bool),
    )
    assert int(mixed.sum()) == 10
    assert not bool(mixed[0])
    assert not bool(mixed[-1])
    np.testing.assert_array_equal(mixed, distribute_ligands(20, 10, "mixed"))


def test_gradient_mean_field_points_toward_denser_receptors():
    order = binding_order(10.0, 4.0, dense_density=1.2, sparse_density=0.8)
    force = binding_free_energy_force(
        n_binders=10,
        bond_work=0.1,
        particle_length=20.0,
        k_d=10.0,
        accessible_receptors=4.0,
        dense_density=1.2,
        sparse_density=0.8,
    )
    velocity = binding_drift(
        n_binders=10,
        mean_force=force,
        attached_probability=0.3,
        particle_friction=0.1,
        attachment_friction=0.02,
    )
    assert order > 0.0
    assert force > 0.0
    assert velocity > 0.0


def test_population_probabilities_and_allocation():
    p_attachment = attachment_probability(n_binders=4.0, k_d=3.0, nu_b=2.0)
    p_cleavage = cleavage_probability(rate=0.2, persistence_time=5.0)
    p_escape = escape_probability(p_attachment, p_cleavage, p_local_depletion=0.1)
    assert 0.0 < p_attachment < 1.0
    assert 0.0 < p_cleavage < 1.0
    assert 0.0 < p_escape < 1.0

    fractions = optimal_allocation(
        {"explore": 0.4, "escape": 0.2, "exploit": 0.5},
        {"explore": 1.0, "escape": 1.0, "exploit": 1.0},
    )
    assert sum(fractions.values()) == pytest.approx(1.0)
    coverage = [
        fractions[name] * capacity
        for name, capacity in {
            "explore": 0.4,
            "escape": 0.2,
            "exploit": 0.5,
        }.items()
    ]
    assert max(coverage) - min(coverage) < 1.0e-12


def test_population_transport_map_is_finite_for_both_architectures():
    inputs = SimpleNamespace(
        L=20.0,
        alpha=0.5,
        d_rec=0.5,
        spring_k=1.0,
        kbt=0.125,
        cleavage_exposure_factor=0.002,
        gamma_parallel=0.1,
        gamma_perp=0.2,
        gamma_rot=5.0,
        D_parallel=0.0125,
        D_perp=0.00625,
        D_rot0=0.00025,
        nu_b=np.pi**1.5,
        nu_c=np.pi**1.5,
        bound_weight=1.0 + 2.0 ** (-1.5),
        mu=4.0 / (3.0 * np.pi),
        observation_time=1.0e6,
    )
    parameters = params_to_array(inputs)
    for architecture in (0, 1):
        values = evaluate_state(architecture, 100.0, 10.0, 10.0, parameters)
        result = dict(zip(OUTPUT_NAMES, values, strict=True))
        assert np.all(np.isfinite(values))
        assert result["speed"] >= 0.0
        assert result["effective_diffusivity"] >= 0.0
        assert 0.0 <= result["mobile_fraction"] <= 1.0


def test_population_source_tables_use_manuscript_terminology():
    table = pd.read_csv(
        ROOT / "data/figure_source/population/panel_ab_function_maps.csv"
    )
    expected = {
        "architecture",
        "phi_b",
        "chi_C",
        "exploration_range_score",
        "gradient_guidance_score",
        "escape_score",
        "exploit_score",
    }
    assert expected.issubset(table.columns)
    assert set(table["architecture"]) == {"polarized", "mixed"}


def test_surface_recurrence_returns_finite_range():
    result = recurrence_prediction(diffusion=1.0, tau=50.0, area=1.0, final_time=1.0e4)
    assert np.isfinite(result["R_finite"])
    assert result["R_finite"] > 0.0


def test_surface_mean_field_reproduces_every_range_row():
    """Tie the full surface range comparison to its molecular calculation."""

    from analysis.surface_2d import load_geometries, load_inputs

    inputs = load_inputs()
    geometries = load_geometries(inputs)
    table = pd.read_csv(
        ROOT / "data/figure_source/surface_2d/panel_d_recurrence_design_points.csv"
    )
    for observed in table.itertuples(index=False):
        predicted = predict_surface_range(
            observed.architecture,
            observed.K_D,
            observed.K_C,
            int(observed.n_binders),
            observed.observation_time,
            inputs,
            geometries,
        )
        for column, key in {
            "persistence_length": "ell_one_clock",
            "effective_trail_width": "effective_trail_width",
            "detachment_area": "detachment_area",
            "mean_field_range": "mean_field_range",
        }.items():
            assert float(getattr(observed, column)) == pytest.approx(
                float(predicted[key]), rel=3.0e-12, abs=1.0e-13
            )


def test_uniform_mean_field_reproduces_every_cleavage_curve_row():
    """Tie the complete plotted cleavage curves to the public calculation."""
    inputs = load_uniform_inputs()
    geometries = load_uniform_geometries(inputs)
    table = pd.read_csv(
        ROOT / "data/figure_source/uniform_3d/panel_f_cleavage_theory.csv"
    )
    expected = {
        "mean_attachments": "mean_attachments",
        "attachment_probability": "attachment_probability",
        "speed": "speed",
        "active_diffusivity": "active_diffusivity",
        "mobile_fraction": "mobile_fraction",
        "receptor_contrast": "contrast",
        "persistence_length": "persistence_length",
        "diffusivity_shift": "diffusivity_shift",
    }
    for observed in table.itertuples(index=False):
        predicted = predict_uniform_transport(
            observed.architecture,
            observed.K_D,
            observed.K_C,
            int(observed.n_binders),
            inputs,
            geometries,
        )
        for column, key in expected.items():
            assert predicted[key] == pytest.approx(
                float(getattr(observed, column)), rel=2.0e-12, abs=1.0e-13
            )
        assert predicted["tau_run"] == pytest.approx(
            float(observed.orientational_persistence_time),
            rel=2.0e-12,
            abs=1.0e-13,
        )


@pytest.mark.parametrize(
    ("architecture", "row_index"), [("polarized", 0), ("mixed", -1)]
)
def test_affinity_curves_reproduce_figure_source_rows(architecture, row_index):
    inputs = load_uniform_inputs()
    geometries = load_uniform_geometries(inputs)
    speed_table = pd.read_csv(
        ROOT / "data/figure_source/uniform_3d/panel_e_speed_theory.csv"
    )
    persistence_table = pd.read_csv(
        ROOT / "data/figure_source/uniform_3d/panel_d_tau_theory.csv"
    )
    speed_row = speed_table.loc[speed_table["architecture"].eq(architecture)].iloc[
        row_index
    ]
    persistence_row = persistence_table.loc[
        persistence_table["architecture"].eq(architecture)
    ].iloc[row_index]
    assert speed_row["K_D"] == pytest.approx(persistence_row["K_D"])
    predicted = predict_uniform_transport(
        architecture,
        float(speed_row["K_D"]),
        10.0,
        10,
        inputs,
        geometries,
    )
    assert predicted["speed"] == pytest.approx(
        float(speed_row["speed"]), rel=2.0e-12
    )
    assert predicted["directional_persistence_time"] == pytest.approx(
        float(persistence_row["directional_persistence_time"]), rel=2.0e-12
    )


def test_mean_field_closure_is_shared_across_calculations():
    from analysis.surface_2d import load_inputs as load_surface_inputs

    uniform = load_uniform_inputs()
    surface = load_surface_inputs()
    assert uniform.cleavage_exposure_factor == pytest.approx(0.002)
    assert surface.cleavage_exposure_factor == pytest.approx(
        uniform.cleavage_exposure_factor
    )


def test_shared_exposure_factor_is_the_declared_rounded_global_calibration():
    from analysis.uniform_3d import (
        exposure_factor_audit,
        load_uniform_calibration_points,
    )

    points = load_uniform_calibration_points(
        ROOT / "data/figure_source/uniform_3d"
    )
    result = exposure_factor_audit(points)
    assert result["point_count"] == 26
    assert result["selected_factor"] == pytest.approx(0.002)
    assert result["least_squares_factor"] == pytest.approx(0.00247152, rel=1.0e-4)
    assert result["rmse_ratio"] < 1.10
    assert result["correlation"] > 0.97


def test_figure_source_tables_are_finite_and_use_supported_architectures():
    for path in sorted((ROOT / "data/figure_source").rglob("*.csv")):
        table = pd.read_csv(path)
        assert len(table) > 0, path
        assert not any(
            str(column).startswith("Unnamed:") for column in table.columns
        ), path
        if "architecture" in table:
            values = set(table["architecture"].dropna().astype(str))
            assert values <= {"polarized", "mixed"}, (path, values)
