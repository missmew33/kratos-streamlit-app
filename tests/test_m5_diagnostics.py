import numpy as np
import pandas as pd

import kratos_diagnostics as kd
from kratos_core import ALL_GROUPS


def _toy_df():
    return pd.DataFrame(
        {
            "group": [group for group in ALL_GROUPS for _ in range(2)],
            "Cited by": [10, 1, 5, 2, 8, 0, 3, 1],
        }
    )


def test_primary_m5_constants_are_frozen():
    assert kd.PRIMARY_PERMUTATION_B == 50_000
    assert kd.PRIMARY_SEED == 20260831


def test_global_null_is_reproducible_and_uses_mc_correction():
    df = _toy_df()
    observed, draws_1 = kd.permutation_null_resolved(df, B=25, seed=77)
    _, draws_2 = kd.permutation_null_resolved(df, B=25, seed=77)
    pd.testing.assert_frame_equal(draws_1, draws_2)

    summary = kd.summarise_permutation_null(observed, draws_1)
    assert (summary["lower_tail_mc_p"] >= 1 / 26).all()
    assert {"null_sd", "null_z", "tie_share", "n_distinct_values"}.issubset(
        summary.columns
    )


def test_stratified_permutation_is_reproducible():
    groups = np.repeat(np.arange(4), 3)
    cites = np.arange(1, 13, dtype=float)
    strata = np.tile([2020, 2021, 2022], 4)
    first = kd.stratified_permutation(
        groups, cites, strata, B=30, seed=12, stats=("HC", "P")
    )
    second = kd.stratified_permutation(
        groups, cites, strata, B=30, seed=12, stats=("HC", "P")
    )
    assert first == second


def test_strata_diagnostics_identifies_frozen_monogroup_mass():
    groups = np.array([0, 0, 1, 2, 3, 3])
    cites = np.array([5, 5, 10, 20, 30, 30.0])
    strata = np.array([2020, 2020, 2021, 2022, 2023, 2023])
    diagnostics = kd.strata_diagnostics(cites, strata, groups)

    assert diagnostics["n_monogroup_strata"] == 4
    assert np.isclose(diagnostics["share_docs_in_monogroup_strata"], 1.0)
    assert np.isclose(diagnostics["share_citation_mass_frozen_monogroup"], 1.0)
    assert np.isclose(diagnostics["between_group_assignment_space_log"], 0.0)


def test_fixed_composition_bootstrap_has_valid_stability_range():
    groups = np.repeat(np.arange(4), 3)
    cites = np.array([1, 2, 9, 2, 3, 4, 1, 1, 8, 2, 5, 7.0])
    result = kd.bootstrap_observed_fixed_composition(
        groups, cites, B=30, seed=4, stats=("KJI",)
    )
    assert 0 <= result["KJI"]["boot_q025"] <= result["KJI"]["boot_q975"] <= 1


def test_logratio_sensitivity_preserves_architectural_bounds():
    groups = np.repeat(np.arange(4), 2)
    cites = np.array([2, 2, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75])
    result = kd.kratos_components_logratio(groups, cites)
    assert 0 <= result["P"] <= result["Abar"] <= 1
    assert result["KJI"] <= result["KCDI"] + 1e-12


def test_log_decomposition_is_exact():
    before = {
        "HD": 0.8,
        "HC": 0.7,
        "P": 0.3,
        "KJI": (0.8 * 0.7) ** 0.5 * 0.3,
        "Abar": 0.6,
    }
    after = {
        "HD": 0.9,
        "HC": 0.8,
        "P": 0.4,
        "KJI": (0.9 * 0.8) ** 0.5 * 0.4,
        "Abar": 0.7,
    }
    decomposition = kd.log_decomposition(before, after, 0.5)
    assert np.isclose(
        np.exp(decomposition["total_lnKJI"]), decomposition["ratio_KJI"]
    )
    ceiling = kd.ceiling_decomposition(before, after)
    assert np.isclose(
        ceiling["Abar"] + ceiling["P_over_Abar"], ceiling["total_lnP"]
    )


def test_publication_year_normalisation_uses_complete_corpus_cohort():
    canonical = pd.DataFrame(
        {"Year": [2020, 2020, 2021, 2021], "Cited by": [0, 10, 2, 6]}
    )
    resolved = pd.DataFrame({"Year": [2020, 2021], "Cited by": [10, 6]})
    weights = kd.publication_year_normalised_weights(canonical, resolved)
    median_weights = kd.publication_year_normalised_weights(
        canonical, resolved, denominator="median"
    )
    assert np.allclose(weights, [2.0, 1.5])
    assert np.allclose(median_weights, [2.0, 1.5])


def test_design_resolution_uses_same_lower_tail_mc_rule():
    groups = np.repeat(np.arange(4), 2)
    cites = np.array([10, 1, 5, 2, 8, 0, 3, 1.0])
    strata = np.tile([2020, 2021], 4)
    result = kd.design_resolution_curve(
        groups,
        cites,
        strata,
        effects=(0.0, 0.2),
        target_group=1,
        M=4,
        B=19,
        seed=3,
        stats=("HC", "P"),
    )
    assert result["criterion"].startswith("p_MC,L")
    assert result["M"] == 4 and result["B"] == 19
    for rates in result["detection_rate"].values():
        assert len(rates) == 2
        assert all(0 <= rate <= 1 for rate in rates)
