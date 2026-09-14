"""KRATOS v2.3 diagnostics used by the Scientometrics manuscript.

This module supplements the deterministic fixed-G=4 KRATOS core. It implements
the locked M5 calibration, null-design diagnostics, empirical design-resolution
assessment, finite-sample stability procedures, recognition-functional-form
sensitivity, citation-age normalisation, and decomposition diagnostics. None of
these functions changes the primary KCDI/P/KJI architecture.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from kratos_core import ALL_GROUPS, compute_kratos_fixed_g

G = 4
P_STAR = 1.0 / G
NULL_METRICS: Tuple[str, ...] = ("H_C_prime", "KCDI", "P", "KJI")
PRIMARY_PERMUTATION_B = 50_000
PRIMARY_SEED = 20260831


def _resolved_frame(df: pd.DataFrame, *, group_col: str, weight_col: str) -> pd.DataFrame:
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
    """Return the composition ceiling A_bar and supplementary P_S diagnostic."""
    work = _resolved_frame(df, group_col=group_col, weight_col=weight_col)
    n_total = len(work)
    total_citations = float(work[weight_col].sum())
    counts = work[group_col].value_counts().reindex(ALL_GROUPS, fill_value=0).astype(float)
    citations = (
        work.groupby(group_col)[weight_col]
        .sum()
        .reindex(ALL_GROUPS, fill_value=0.0)
        .astype(float)
    )

    p_values, a_values, s_values = [], [], []
    for group in ALL_GROUPS:
        p_u = float(counts.loc[group]) / n_total if n_total else 0.0
        s_u = float(citations.loc[group]) / total_citations if total_citations > 0 else 0.0
        a_u = max(0.0, 1.0 - abs(p_u - P_STAR) / P_STAR)
        s_factor = 0.0 if p_u == 0 else max(0.0, 1.0 - abs((s_u / p_u) - 1.0))
        p_values.append(p_u)
        a_values.append(a_u)
        s_values.append(s_factor)

    return {
        "A_bar": float(np.mean(a_values)),
        "P_S": float(np.dot(p_values, s_values)),
        "n_resolved": float(n_total),
    }


def _entropy_from_totals(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    shares = values / total
    positive = shares > 0.0
    return -float(np.sum(shares[positive] * np.log(shares[positive]))) / math.log(G)


def _kcdi(h_d_prime: float, h_c_prime: float, lambda_param: float) -> float:
    if lambda_param == 0.0:
        return float(h_c_prime)
    if lambda_param == 1.0:
        return float(h_d_prime)
    return float((h_d_prime ** lambda_param) * (h_c_prime ** (1.0 - lambda_param)))


def permutation_null_resolved(
    df: pd.DataFrame,
    *,
    B: int = PRIMARY_PERMUTATION_B,
    seed: int = PRIMARY_SEED,
    group_col: str = "group",
    weight_col: str = "Cited by",
    lambda_param: float = 0.5,
):
    """Global citation-permutation baseline on the exact resolved set.

    ``B=50_000`` is the locked primary manuscript specification. Smaller values
    may be passed explicitly for exploratory use and automated tests.
    """
    if B <= 0:
        raise ValueError("B must be a positive integer")
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must lie in [0, 1]")

    resolved = _resolved_frame(df, group_col=group_col, weight_col=weight_col)
    group_index = {group: i for i, group in enumerate(ALL_GROUPS)}
    codes = resolved[group_col].map(group_index).to_numpy(dtype=int)
    weights = resolved[weight_col].to_numpy(dtype=float)
    counts = np.bincount(codes, minlength=G).astype(float)
    p = counts / len(resolved)
    a = np.maximum(0.0, 1.0 - np.abs(p - P_STAR) / P_STAR)
    h_d_prime = _entropy_from_totals(counts)

    _, details = compute_kratos_fixed_g(
        df,
        group_col=group_col,
        weight_col=weight_col,
        lambda_param=lambda_param,
    )
    observed = {metric: float(details[metric]) for metric in NULL_METRICS}

    total_citations = float(weights.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty((B, len(NULL_METRICS)), dtype=float)
    for b in range(B):
        permuted = rng.permutation(weights)
        group_citations = np.bincount(codes, weights=permuted, minlength=G).astype(float)
        h_c_prime = _entropy_from_totals(group_citations)
        kcdi = _kcdi(h_d_prime, h_c_prime, lambda_param)
        s = group_citations / total_citations if total_citations > 0 else np.zeros(G)
        recognition = np.zeros(G)
        nonzero = p > 0
        recognition[nonzero] = np.maximum(
            0.0, 1.0 - np.abs((s[nonzero] / p[nonzero]) - 1.0)
        )
        p_factor = float(np.mean(a * recognition))
        draws[b] = [h_c_prime, kcdi, p_factor, kcdi * p_factor]

    frame = pd.DataFrame(draws, columns=NULL_METRICS)
    frame.insert(0, "draw", np.arange(B, dtype=int))
    return observed, frame


def summarise_permutation_null(
    observed: Mapping[str, float], draws: pd.DataFrame
) -> pd.DataFrame:
    """Summarise the observed metrics relative to one permutation baseline."""
    if len(draws) == 0:
        raise ValueError("Permutation draws are empty")
    rows = []
    for metric in NULL_METRICS:
        values = pd.to_numeric(draws[metric], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No valid permutation values for metric {metric}")
        obs = float(observed[metric])
        arr = values.to_numpy(dtype=float)
        median = float(np.median(arr))
        sd = float(np.std(arr, ddof=1))
        p025 = float(np.percentile(arr, 2.5))
        p975 = float(np.percentile(arr, 97.5))
        rows.append(
            {
                "metric": metric,
                "observed": obs,
                "null_median": median,
                "null_p025": p025,
                "null_p975": p975,
                "observed_minus_null_median": obs - median,
                "null_sd": sd,
                "null_z": (obs - median) / sd if sd > 0 else np.nan,
                "lower_tail_mc_p": (1 + int(np.sum(arr <= obs))) / (len(arr) + 1),
                "upper_tail_mc_p": (1 + int(np.sum(arr >= obs))) / (len(arr) + 1),
                "tie_share": float(np.mean(np.isclose(arr, obs, rtol=0.0, atol=1e-12))),
                "n_distinct_values": int(np.unique(np.round(arr, 12)).size),
                "null_position": (
                    "below 95% null range"
                    if obs < p025
                    else "above 95% null range"
                    if obs > p975
                    else "within 95% null range"
                ),
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
):
    resolved = _resolved_frame(df, group_col=group_col, weight_col=weight_col)
    return {
        "n_input": int(len(df)),
        "n_resolved": int(len(resolved)),
        "B": int(B),
        "seed": int(seed),
        "lambda_param": float(lambda_param),
        "group_col": group_col,
        "weight_col": weight_col,
        "null_design": (
            "citation permutation within exact resolved G=4 set; group labels/group "
            "sizes and empirical citation multiset fixed"
        ),
    }


# Array-level M5 implementation -------------------------------------------------
def _norm_entropy(shares: np.ndarray) -> float:
    values = np.asarray(shares, dtype=float)
    values = values[values > 0]
    return float(-(values * np.log(values)).sum() / np.log(G)) if values.size else 0.0


def kratos_components(groups, cites, lam: float = 0.5) -> dict:
    """Calculate fixed-G=4 components from integer group codes and citation weights."""
    groups = np.asarray(groups, dtype=int)
    cites = np.asarray(cites, dtype=float)
    n = np.bincount(groups, minlength=G).astype(float)
    if n.sum() == 0:
        return dict(HD=0.0, HC=0.0, KCDI=0.0, Abar=0.0, P=0.0, P_S=0.0, KJI=0.0)
    c = np.bincount(groups, weights=cites, minlength=G).astype(float)
    p = n / n.sum()
    s = c / c.sum() if c.sum() > 0 else np.zeros(G)
    hd, hc = _norm_entropy(p), _norm_entropy(s)
    a = np.maximum(0.0, 1.0 - np.abs(p - P_STAR) / P_STAR)
    ratio = np.divide(s, p, out=np.zeros(G), where=p > 0)
    recognition = np.where(p > 0, np.maximum(0.0, 1.0 - np.abs(ratio - 1.0)), 0.0)
    kcdi = _kcdi(hd, hc, lam)
    p_factor = float(np.mean(a * recognition))
    return {
        "HD": hd,
        "HC": hc,
        "KCDI": kcdi,
        "Abar": float(np.mean(a)),
        "P": p_factor,
        "P_S": float(np.sum(p * recognition)),
        "KJI": float(kcdi * p_factor),
    }


def kratos_components_logratio(
    groups, cites, lam: float = 0.5, cap: float = np.log(2.0)
) -> dict:
    """Sensitivity calculation using reciprocal-symmetric log-ratio alignment."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    groups = np.asarray(groups, dtype=int)
    cites = np.asarray(cites, dtype=float)
    n = np.bincount(groups, minlength=G).astype(float)
    if n.sum() == 0:
        return dict(HD=0.0, HC=0.0, KCDI=0.0, Abar=0.0, P=0.0, P_S=0.0, KJI=0.0)
    c = np.bincount(groups, weights=cites, minlength=G).astype(float)
    p = n / n.sum()
    s = c / c.sum() if c.sum() > 0 else np.zeros(G)
    hd, hc = _norm_entropy(p), _norm_entropy(s)
    a = np.maximum(0.0, 1.0 - np.abs(p - P_STAR) / P_STAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            (p > 0) & (s > 0),
            np.log(np.divide(s, p, out=np.ones(G), where=p > 0)),
            -np.inf,
        )
    recognition = np.where(
        np.isfinite(log_ratio), np.maximum(0.0, 1.0 - np.abs(log_ratio) / cap), 0.0
    )
    kcdi = _kcdi(hd, hc, lam)
    p_factor = float(np.mean(a * recognition))
    return {
        "HD": hd,
        "HC": hc,
        "KCDI": kcdi,
        "Abar": float(np.mean(a)),
        "P": p_factor,
        "P_S": float(np.sum(p * recognition)),
        "KJI": float(kcdi * p_factor),
    }


def stratified_permutation(
    groups,
    cites,
    strata=None,
    B: int = PRIMARY_PERMUTATION_B,
    lam: float = 0.5,
    seed: int = PRIMARY_SEED,
    stats=("HC", "KCDI", "P", "P_S", "KJI"),
) -> dict:
    """Permutation calibration globally or within supplied temporal strata."""
    if B <= 0:
        raise ValueError("B must be positive")
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups, dtype=int)
    cites = np.asarray(cites, dtype=float)
    observed = kratos_components(groups, cites, lam)
    if strata is None:
        blocks = [np.arange(len(cites))]
    else:
        strata = np.asarray(strata)
        blocks = [np.flatnonzero(strata == value) for value in np.unique(strata)]

    draws = np.empty((B, len(stats)), dtype=float)
    for b in range(B):
        permuted = cites.copy()
        for idx in blocks:
            if idx.size > 1:
                permuted[idx] = rng.permutation(cites[idx])
        components = kratos_components(groups, permuted, lam)
        draws[b] = [components[key] for key in stats]

    out = {}
    for j, key in enumerate(stats):
        col = draws[:, j]
        median = float(np.median(col))
        sd = float(np.std(col, ddof=1))
        q025 = float(np.percentile(col, 2.5))
        q975 = float(np.percentile(col, 97.5))
        out[key] = {
            "observed": observed[key],
            "null_median": median,
            "null_q025": q025,
            "null_q975": q975,
            "p_mc_lower": float((1 + np.sum(col <= observed[key])) / (B + 1)),
            "below_range": bool(observed[key] < q025),
            "null_sd": sd,
            "null_delta": float(observed[key] - median),
            "null_z": float((observed[key] - median) / sd) if sd > 0 else float("nan"),
            "tie_share": float(np.mean(np.isclose(col, observed[key], rtol=0.0, atol=1e-12))),
            "n_distinct_values": int(np.unique(np.round(col, 12)).size),
        }
    out["_Abar"] = observed["Abar"]
    return out


def bootstrap_observed(
    groups,
    cites,
    B: int = 5000,
    lam: float = 0.5,
    seed: int = PRIMARY_SEED,
    stats=("HD", "HC", "KCDI", "Abar", "P", "P_S", "KJI"),
) -> dict:
    """Overall document bootstrap allowing analytical-group composition to vary."""
    if B <= 0:
        raise ValueError("B must be positive")
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups, dtype=int)
    cites = np.asarray(cites, dtype=float)
    n = len(groups)
    observed = kratos_components(groups, cites, lam)
    draws = np.empty((B, len(stats)), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        components = kratos_components(groups[idx], cites[idx], lam)
        draws[b] = [components[key] for key in stats]
    return {
        key: {
            "observed": observed[key],
            "boot_median": float(np.median(draws[:, j])),
            "boot_q025": float(np.percentile(draws[:, j], 2.5)),
            "boot_q975": float(np.percentile(draws[:, j], 97.5)),
        }
        for j, key in enumerate(stats)
    }


def bootstrap_observed_stratified(
    groups,
    cites,
    B: int = 5000,
    lam: float = 0.5,
    seed: int = PRIMARY_SEED,
    stats=("HC", "KCDI", "P", "P_S", "KJI"),
) -> dict:
    """Fixed-composition bootstrap, resampling with replacement within groups."""
    if B <= 0:
        raise ValueError("B must be positive")
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups, dtype=int)
    cites = np.asarray(cites, dtype=float)
    by_group = [np.flatnonzero(groups == u) for u in range(G)]
    observed = kratos_components(groups, cites, lam)
    draws = np.empty((B, len(stats)), dtype=float)
    for b in range(B):
        selection = np.concatenate(
            [rng.choice(idx, idx.size, replace=True) for idx in by_group if idx.size]
        )
        components = kratos_components(groups[selection], cites[selection], lam)
        draws[b] = [components[key] for key in stats]
    return {
        key: {
            "observed": observed[key],
            "boot_median": float(np.median(draws[:, j])),
            "boot_q025": float(np.percentile(draws[:, j], 2.5)),
            "boot_q975": float(np.percentile(draws[:, j], 97.5)),
        }
        for j, key in enumerate(stats)
    }


def bootstrap_observed_fixed_composition(
    groups,
    cites,
    B: int = 5000,
    lam: float = 0.5,
    seed: int = PRIMARY_SEED,
    stats=("HC", "KCDI", "P", "P_S", "KJI"),
) -> dict:
    """Alias using the terminology of the manuscript's stability analysis."""
    return bootstrap_observed_stratified(
        groups, cites, B=B, lam=lam, seed=seed, stats=stats
    )


def log_decomposition(before: dict, after: dict, lam: float = 0.5) -> dict:
    """Exact log-scale decomposition of a change in KJI."""
    contribution = {
        "HD": lam * (np.log(after["HD"]) - np.log(before["HD"])),
        "HC": (1.0 - lam) * (np.log(after["HC"]) - np.log(before["HC"])),
        "P": np.log(after["P"]) - np.log(before["P"]),
    }
    contribution["total_lnKJI"] = sum(contribution.values())
    contribution["ratio_KJI"] = after["KJI"] / before["KJI"]
    return {key: float(value) for key, value in contribution.items()}


def ceiling_decomposition(before: dict, after: dict) -> dict:
    """Decompose a change in P into A_bar and P/A_bar components."""
    if (
        before["P"] <= 0
        or after["P"] <= 0
        or before["Abar"] <= 0
        or after["Abar"] <= 0
    ):
        return {"Abar": float("nan"), "P_over_Abar": float("nan"), "total_lnP": float("nan")}
    d_abar = float(np.log(after["Abar"]) - np.log(before["Abar"]))
    d_eff = float(
        np.log(after["P"] / after["Abar"])
        - np.log(before["P"] / before["Abar"])
    )
    return {"Abar": d_abar, "P_over_Abar": d_eff, "total_lnP": d_abar + d_eff}


def strata_diagnostics(cites, strata, groups) -> dict:
    """Describe the counterfactual reassignment space of a stratified null design."""
    cites = np.asarray(cites, dtype=float)
    strata = np.asarray(strata)
    groups = np.asarray(groups, dtype=int)
    if not (len(cites) == len(strata) == len(groups)):
        raise ValueError("cites, strata, and groups must have equal length")
    values, counts = np.unique(strata, return_counts=True)
    total = float(cites.sum())

    def log_factorial(k: int) -> float:
        return float(math.lgamma(k + 1))

    singleton_values, monogroup_values, groups_per_stratum = [], [], []
    raw_space, group_space = 0.0, 0.0
    for value, n_s in zip(values, counts):
        idx = np.flatnonzero(strata == value)
        stratum_groups = groups[idx]
        n_groups = len(np.unique(stratum_groups))
        groups_per_stratum.append(n_groups)
        if n_s == 1:
            singleton_values.append(value)
        if n_groups == 1:
            monogroup_values.append(value)
        raw_space += log_factorial(int(n_s))
        group_space += log_factorial(int(n_s)) - sum(
            log_factorial(int(m)) for m in np.bincount(stratum_groups, minlength=G)
        )

    singleton_mask = np.isin(strata, singleton_values)
    monogroup_mask = np.isin(strata, monogroup_values)
    return {
        "n_strata": int(len(values)),
        "size_min": int(counts.min()),
        "size_median": float(np.median(counts)),
        "size_max": int(counts.max()),
        "n_singleton_strata": int(len(singleton_values)),
        "share_docs_in_singletons": float(singleton_mask.mean()),
        "share_citation_mass_frozen_singletons": (
            float(cites[singleton_mask].sum() / total) if total > 0 else 0.0
        ),
        "n_monogroup_strata": int(len(monogroup_values)),
        "share_docs_in_monogroup_strata": float(monogroup_mask.mean()),
        "share_citation_mass_frozen_monogroup": (
            float(cites[monogroup_mask].sum() / total) if total > 0 else 0.0
        ),
        "groups_per_stratum_min": int(min(groups_per_stratum)),
        "groups_per_stratum_median": float(np.median(groups_per_stratum)),
        "log_permutation_space_raw": float(raw_space),
        "between_group_assignment_space_log": float(group_space),
        "n_u": np.bincount(groups, minlength=G).tolist(),
    }


def compare_null_designs(
    groups,
    cites,
    year,
    B: int = PRIMARY_PERMUTATION_B,
    lam: float = 0.5,
    seed: int = PRIMARY_SEED,
    stats=("HC", "P", "KJI"),
) -> dict:
    """Run global, biennium-stratified, and exact-year calibrations."""
    year = np.asarray(year)
    designs = {"global": None, "biennium": (year // 2) * 2, "exact_year": year}
    out = {}
    for name, strata in designs.items():
        result = stratified_permutation(
            groups, cites, strata, B=B, lam=lam, seed=seed, stats=stats
        )
        out[name] = {key: result[key] for key in stats}
    return out


def design_resolution_curve(
    groups,
    cites,
    strata,
    effects=(0.0, 0.20, 0.40, 0.60),
    target_group: int = 1,
    M: int = 500,
    B: int = 999,
    lam: float = 0.5,
    seed: int = PRIMARY_SEED,
    stats=("HC", "P", "KJI"),
    alpha: float = 0.025,
) -> dict:
    """Empirical design-resolution curve from null-compatible outer configurations.

    For each outer replicate, citation weights are first permuted within strata to
    create a null-compatible configuration. A known group-specific citation
    reduction is then planted and recalibrated using the same lower-tail Monte
    Carlo criterion as the primary analysis. These rates diagnose design
    resolution; they are not additional tests of the observed configuration.
    """
    if M <= 0 or B <= 0:
        raise ValueError("M and B must be positive")
    groups = np.asarray(groups, dtype=int)
    cites = np.asarray(cites, dtype=float)
    strata = np.asarray(strata)
    blocks = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    mask = groups == target_group
    if not mask.any():
        raise ValueError(f"target group {target_group} has no documents")

    root = np.random.SeedSequence([seed, target_group])
    children = root.spawn(len(effects) * M)
    rejects = {key: np.zeros(len(effects)) for key in stats}

    def permute_within(values, rng):
        out = values.copy()
        for idx in blocks:
            if idx.size > 1:
                out[idx] = rng.permutation(values[idx])
        return out

    for j, effect in enumerate(effects):
        for r in range(M):
            rng = np.random.default_rng(children[j * M + r])
            null_compatible = permute_within(cites, rng)
            planted = null_compatible.copy()
            planted[mask] *= 1.0 - effect
            observed = kratos_components(groups, planted, lam)
            draws = np.empty((B, len(stats)), dtype=float)
            for b in range(B):
                permuted = permute_within(planted, rng)
                components = kratos_components(groups, permuted, lam)
                draws[b] = [components[key] for key in stats]
            for i, key in enumerate(stats):
                p_mc = (1.0 + np.sum(draws[:, i] <= observed[key])) / (B + 1)
                rejects[key][j] += int(p_mc <= alpha)

    return {
        "target_group": int(target_group),
        "effects": [float(value) for value in effects],
        "M": int(M),
        "B": int(B),
        "alpha": float(alpha),
        "criterion": "p_MC,L = (1 + #{T_b <= T_obs}) / (B + 1) <= alpha",
        "detection_rate": {key: (rejects[key] / M).tolist() for key in stats},
    }


def publication_year_normalised_weights(
    canonical_df: pd.DataFrame,
    resolved_df: pd.DataFrame,
    *,
    year_col: str = "Year",
    canonical_weight_col: str = "Cited by",
    resolved_weight_col: str = "Cited by",
    denominator: str = "mean",
) -> np.ndarray:
    """Return within-corpus publication-year-normalised citation weights for D_R.

    The locked primary specification uses the mean raw citation count in each
    publication-year cohort of the complete canonical corpus D. ``median`` is
    exposed only for the prespecified Trade Fairs/MICE denominator sensitivity
    check. Cohorts with a zero or missing denominator receive weight zero.
    """
    for frame, column in (
        (canonical_df, year_col),
        (canonical_df, canonical_weight_col),
        (resolved_df, year_col),
        (resolved_df, resolved_weight_col),
    ):
        if column not in frame.columns:
            raise ValueError(f"Missing column: {column}")
    if denominator not in {"mean", "median"}:
        raise ValueError("denominator must be 'mean' or 'median'")

    years = pd.to_numeric(canonical_df[year_col], errors="coerce")
    raw = (
        pd.to_numeric(canonical_df[canonical_weight_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    grouped = (
        pd.DataFrame({"year": years, "weight": raw})
        .dropna(subset=["year"])
        .groupby("year")["weight"]
    )
    denominators = grouped.mean() if denominator == "mean" else grouped.median()

    resolved_year = pd.to_numeric(resolved_df[year_col], errors="coerce")
    resolved_weight = (
        pd.to_numeric(resolved_df[resolved_weight_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    normalised = []
    for year, weight in zip(resolved_year, resolved_weight):
        cohort_denominator = denominators.get(year, np.nan)
        normalised.append(
            float(weight / cohort_denominator)
            if pd.notna(cohort_denominator) and cohort_denominator > 0
            else 0.0
        )
    return np.asarray(normalised, dtype=float)
