"""Robustness diagnostics for the KRATOS fixed-G=4 measurement regime.

This module extends the deterministic KRATOS core with diagnostics required for
comparability and validation work. It does not change the primary KCDI/P/KJI
architecture.

Implemented diagnostics
-----------------------
1. Resolved-set permutation null calibration: citation counts are permuted only
   among records already resolved into the substantive four-group universe. This
   keeps N_R, the four group sizes, and the empirical citation distribution fixed
   and removes only the association between group membership and citations.
2. Composition diagnostics: mean participation factor (A_bar) and a
   participation-weighted recognition-only summary P_S. These are supplementary
   quantities for decomposing P; they are not replacements for P or KJI.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from kratos_core import ALL_GROUPS, compute_kratos_fixed_g


NULL_METRICS: Tuple[str, ...] = ("H_C_prime", "KCDI", "P", "KJI")


def _resolved_frame(
    df: pd.DataFrame,
    *,
    group_col: str,
    weight_col: str,
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if weight_col not in df.columns:
        raise ValueError(f"Missing weight column: {weight_col}")

    work = df.loc[df[group_col].isin(ALL_GROUPS), [group_col, weight_col]].copy()
    if work.empty:
        raise ValueError("No records belong to the substantive G=4 universe")
    work[weight_col] = (
        pd.to_numeric(work[weight_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(float)
    )
    return work


def composition_diagnostics(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    weight_col: str = "Cited by",
) -> Dict[str, float]:
    """Return supplementary decomposition diagnostics for the resolved set.

    ``A_bar`` is the arithmetic mean of the four participation factors A(u) and
    is the maximum value P could attain if S(u)=1 for all groups under the fixed
    p*=1/4 reference.

    ``P_S`` is sum_u p_u S(u). It summarises proportional recognition without
    using the fixed p*=1/4 participation reference. It remains composition-
    weighted and is therefore reported only as a supplementary recognition
    diagnostic, not as a replacement for P.
    """
    work = _resolved_frame(df, group_col=group_col, weight_col=weight_col)
    n_total = len(work)
    total_citations = float(work[weight_col].sum())
    p_star = 1.0 / len(ALL_GROUPS)

    counts = work[group_col].value_counts().reindex(ALL_GROUPS, fill_value=0).astype(float)
    citations = (
        work.groupby(group_col)[weight_col]
        .sum()
        .reindex(ALL_GROUPS, fill_value=0.0)
        .astype(float)
    )

    a_values = []
    s_values = []
    p_values = []
    for group in ALL_GROUPS:
        p_u = float(counts.loc[group]) / n_total if n_total else 0.0
        s_u = float(citations.loc[group]) / total_citations if total_citations > 0 else 0.0
        a_u = max(0.0, 1.0 - abs(p_u - p_star) / p_star)
        s_factor = 0.0 if p_u == 0 else max(0.0, 1.0 - abs((s_u / p_u) - 1.0))
        p_values.append(p_u)
        a_values.append(a_u)
        s_values.append(s_factor)

    a_bar = float(np.mean(a_values))
    p_s = float(np.dot(np.asarray(p_values), np.asarray(s_values)))
    return {
        "A_bar": a_bar,
        "P_S": p_s,
        "n_resolved": float(n_total),
    }


def _entropy_from_totals(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    shares = values / total
    positive = shares > 0.0
    entropy = -float(np.sum(shares[positive] * np.log(shares[positive])))
    return entropy / math.log(len(ALL_GROUPS))


def _kcdi(h_d_prime: float, h_c_prime: float, lambda_param: float) -> float:
    if lambda_param == 0.0:
        return float(h_c_prime)
    if lambda_param == 1.0:
        return float(h_d_prime)
    return float((h_d_prime ** lambda_param) * (h_c_prime ** (1.0 - lambda_param)))


def permutation_null_resolved(
    df: pd.DataFrame,
    *,
    B: int = 5000,
    seed: int = 20260831,
    group_col: str = "group",
    weight_col: str = "Cited by",
    lambda_param: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Generate a null distribution conditional on the exact resolved set.

    The substantive G=4 records are frozen. Their group labels and group sizes
    remain fixed, and the observed citation counts are randomly permuted across
    those same records. This is equivalent to permuting the four group labels
    while preserving group sizes, but citation permutation is computationally
    simpler and leaves the exact resolved record set explicit.

    The procedure removes only the association between group membership and
    citation outcome. It does not model corpus-boundary uncertainty, demographic
    misclassification, or unresolved-metadata mechanisms.
    """
    if B <= 0:
        raise ValueError("B must be a positive integer")
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must lie in [0, 1]")

    resolved = _resolved_frame(df, group_col=group_col, weight_col=weight_col)
    group_index = {group: idx for idx, group in enumerate(ALL_GROUPS)}
    codes = resolved[group_col].map(group_index).to_numpy(dtype=int)
    weights = resolved[weight_col].to_numpy(dtype=float)
    n_resolved = len(resolved)

    counts = np.bincount(codes, minlength=len(ALL_GROUPS)).astype(float)
    p = counts / n_resolved
    p_star = 1.0 / len(ALL_GROUPS)
    a = np.maximum(0.0, 1.0 - np.abs(p - p_star) / p_star)
    h_d_prime = _entropy_from_totals(counts)

    _, observed_details = compute_kratos_fixed_g(
        df,
        group_col=group_col,
        weight_col=weight_col,
        lambda_param=lambda_param,
    )
    observed = {metric: float(observed_details[metric]) for metric in NULL_METRICS}

    total_citations = float(weights.sum())
    rng = np.random.default_rng(seed)
    h_c_draws = np.empty(B, dtype=float)
    kcdi_draws = np.empty(B, dtype=float)
    p_draws = np.empty(B, dtype=float)
    kji_draws = np.empty(B, dtype=float)

    for draw in range(B):
        permuted = rng.permutation(weights)
        group_citations = np.bincount(codes, weights=permuted, minlength=len(ALL_GROUPS)).astype(float)
        h_c = _entropy_from_totals(group_citations)
        kcdi = _kcdi(h_d_prime, h_c, lambda_param)

        if total_citations > 0.0:
            s = group_citations / total_citations
        else:
            s = np.zeros(len(ALL_GROUPS), dtype=float)

        recognition = np.zeros(len(ALL_GROUPS), dtype=float)
        nonzero = p > 0.0
        recognition[nonzero] = np.maximum(
            0.0,
            1.0 - np.abs((s[nonzero] / p[nonzero]) - 1.0),
        )
        p_factor = float(np.mean(a * recognition))
        kji = kcdi * p_factor

        h_c_draws[draw] = h_c
        kcdi_draws[draw] = kcdi
        p_draws[draw] = p_factor
        kji_draws[draw] = kji

    draws = pd.DataFrame(
        {
            "draw": np.arange(B, dtype=int),
            "H_C_prime": h_c_draws,
            "KCDI": kcdi_draws,
            "P": p_draws,
            "KJI": kji_draws,
        }
    )
    return observed, draws


def summarise_permutation_null(
    observed: Mapping[str, float],
    draws: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise observed metrics relative to the resolved-set null baseline."""
    rows = []
    B = len(draws)
    if B == 0:
        raise ValueError("Permutation draws are empty")

    for metric in NULL_METRICS:
        values = pd.to_numeric(draws[metric], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No valid permutation values for metric {metric}")

        obs = float(observed[metric])
        lower_count = int((values <= obs).sum())
        upper_count = int((values >= obs).sum())
        p025 = float(values.quantile(0.025))
        p975 = float(values.quantile(0.975))
        median = float(values.median())
        tie_share = float(np.mean(np.isclose(values.to_numpy(dtype=float), obs, rtol=1e-12, atol=1e-12)))

        if obs < p025:
            position = "below 95% null range"
        elif obs > p975:
            position = "above 95% null range"
        else:
            position = "within 95% null range"

        rows.append(
            {
                "metric": metric,
                "observed": obs,
                "null_median": median,
                "null_p025": p025,
                "null_p975": p975,
                "observed_minus_null_median": obs - median,
                "lower_tail_mc_p": (lower_count + 1) / (len(values) + 1),
                "upper_tail_mc_p": (upper_count + 1) / (len(values) + 1),
                "tie_share": tie_share,
                "null_position": position,
            }
        )
    return pd.DataFrame(rows)


def null_snapshot(
    *,
    df: pd.DataFrame,
    B: int,
    seed: int,
    group_col: str,
    weight_col: str,
    lambda_param: float,
) -> Dict[str, float | int | str]:
    resolved = _resolved_frame(df, group_col=group_col, weight_col=weight_col)
    return {
        "n_input": int(len(df)),
        "n_resolved": int(len(resolved)),
        "B": int(B),
        "seed": int(seed),
        "lambda_param": float(lambda_param),
        "group_col": group_col,
        "weight_col": weight_col,
        "null_design": "citation permutation within exact resolved G=4 set; group labels and group sizes fixed",
    }
