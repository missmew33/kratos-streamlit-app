import numpy as np
import pandas as pd

from kratos_diagnostics import (
    composition_diagnostics,
    permutation_null_resolved,
    summarise_permutation_null,
)
from kratos_core import ALL_GROUPS, compute_kratos_fixed_g


def _toy_df():
    groups = [
        ALL_GROUPS[0], ALL_GROUPS[0],
        ALL_GROUPS[1], ALL_GROUPS[1],
        ALL_GROUPS[2], ALL_GROUPS[2],
        ALL_GROUPS[3], ALL_GROUPS[3],
        "unknown x Global North",
    ]
    citations = [10, 1, 5, 2, 8, 0, 3, 1, 100]
    return pd.DataFrame({"group": groups, "Cited by": citations})


def test_composition_ceiling_is_one_under_equal_participation():
    df = _toy_df()
    diag = composition_diagnostics(df)
    assert np.isclose(diag["A_bar"], 1.0)
    assert 0.0 <= diag["P_S"] <= 1.0
    assert diag["n_resolved"] == 8


def test_permutation_freezes_exact_resolved_set():
    df = _toy_df()
    observed, draws = permutation_null_resolved(df, B=50, seed=123)
    _, details = compute_kratos_fixed_g(df)
    assert np.isclose(observed["KJI"], details["KJI"])
    assert np.isclose(observed["P"], details["P"])
    assert len(draws) == 50
    assert set(draws.columns) == {"draw", "H_C_prime", "KCDI", "P", "KJI"}


def test_permutation_is_reproducible():
    df = _toy_df()
    _, d1 = permutation_null_resolved(df, B=25, seed=77)
    _, d2 = permutation_null_resolved(df, B=25, seed=77)
    pd.testing.assert_frame_equal(d1, d2)


def test_summary_uses_monte_carlo_correction():
    df = _toy_df()
    observed, draws = permutation_null_resolved(df, B=20, seed=9)
    summary = summarise_permutation_null(observed, draws)
    assert set(summary["metric"]) == {"H_C_prime", "KCDI", "P", "KJI"}
    assert (summary["lower_tail_mc_p"] > 0).all()
    assert (summary["upper_tail_mc_p"] > 0).all()
    assert summary["lower_tail_mc_p"].max() <= 1.0
    assert summary["upper_tail_mc_p"].max() <= 1.0
