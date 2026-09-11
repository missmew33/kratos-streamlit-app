#!/usr/bin/env python3
"""Corpus-conditional permutation baselines for KRATOS G=4 metrics.

Input is a record-level demographic audit produced by
``scripts/kratos_analyze_csv.py``. Licensed Scopus source data are not stored in
this repository.

The null baseline preserves, for each corpus:
- the resolved analytical set size ``N_R``;
- substantive G=4 group labels and group composition;
- the empirical citation-count distribution among resolved records.

It removes only the association between analytical group and citation outcome
by randomly permuting citation counts across resolved documents. The resulting
distributions are diagnostic baselines, not population-level confidence
intervals or significance tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kratos_core import ALL_GROUPS, compute_kratos_fixed_g


NULL_METRICS = ["H_C_prime", "KCDI", "P", "KJI"]


def _resolved_mask(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    return df[group_col].isin(ALL_GROUPS)


def observed_kratos(
    df: pd.DataFrame,
    *,
    group_col: str,
    weight_col: str,
    lambda_param: float,
) -> dict[str, float]:
    """Return the observed KRATOS quantities for one frozen corpus."""
    _, details = compute_kratos_fixed_g(
        df,
        group_col=group_col,
        weight_col=weight_col,
        lambda_param=lambda_param,
    )
    return details


def permutation_null(
    df: pd.DataFrame,
    *,
    B: int,
    seed: int,
    group_col: str = "group",
    weight_col: str = "Cited by",
    lambda_param: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Generate a corpus-conditional null distribution by citation permutation.

    Citation counts are permuted only among records belonging to the substantive
    G=4 universe. Unresolved records remain unchanged and outside the primary
    KRATOS calculation. This preserves ``N_R``, group composition and the exact
    empirical citation distribution of the resolved analytical set.
    """
    if B <= 0:
        raise ValueError("B must be a positive integer")
    if weight_col not in df.columns:
        raise ValueError(f"Missing weight column: {weight_col}")

    resolved = _resolved_mask(df, group_col)
    n_resolved = int(resolved.sum())
    if n_resolved == 0:
        raise ValueError("No records belong to the substantive G=4 universe")

    work = df.copy()
    resolved_weights = (
        pd.to_numeric(work.loc[resolved, weight_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )

    observed = observed_kratos(
        work,
        group_col=group_col,
        weight_col=weight_col,
        lambda_param=lambda_param,
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for draw in range(B):
        permuted = work.copy()
        permuted.loc[resolved, weight_col] = rng.permutation(resolved_weights)
        _, details = compute_kratos_fixed_g(
            permuted,
            group_col=group_col,
            weight_col=weight_col,
            lambda_param=lambda_param,
        )
        rows.append(
            {
                "draw": draw,
                **{metric: float(details[metric]) for metric in NULL_METRICS},
            }
        )

    return observed, pd.DataFrame(rows)


def summarise_null(
    observed: dict[str, float],
    draws: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise observed values against their permutation distributions."""
    rows = []
    for metric in NULL_METRICS:
        values = pd.to_numeric(draws[metric], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No valid permutation values for metric {metric}")

        obs = float(observed[metric])
        rows.append(
            {
                "metric": metric,
                "observed": obs,
                "null_median": float(values.median()),
                "null_p025": float(values.quantile(0.025)),
                "null_p975": float(values.quantile(0.975)),
                "proportion_null_le_observed": float((values <= obs).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corpus-conditional KRATOS permutation baselines."
    )
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kratos_null_baseline"),
    )
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--weight-col", default="Cited by")
    parser.add_argument("--year-col", default="Year")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument("--lambda-param", type=float, default=0.5)
    parser.add_argument("--B", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    df = pd.read_csv(args.audit_csv)

    if args.year_min is not None:
        if args.year_col not in df.columns:
            raise ValueError(f"Missing year column: {args.year_col}")
        years = pd.to_numeric(df[args.year_col], errors="coerce")
        df = df[(years >= args.year_min) & (years <= args.year_max)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    observed, draws = permutation_null(
        df,
        B=args.B,
        seed=args.seed,
        group_col=args.group_col,
        weight_col=args.weight_col,
        lambda_param=args.lambda_param,
    )
    summary = summarise_null(observed, draws)

    draws.to_csv(args.output_dir / "permutation_null_draws.csv", index=False)
    summary.to_csv(args.output_dir / "permutation_null_summary.csv", index=False)

    snapshot = pd.DataFrame(
        [
            {
                "n_input": int(len(df)),
                "n_resolved": int(_resolved_mask(df, args.group_col).sum()),
                "B": int(args.B),
                "seed": int(args.seed),
                "lambda_param": float(args.lambda_param),
                "group_col": args.group_col,
                "weight_col": args.weight_col,
                "year_min": args.year_min,
                "year_max": args.year_max,
            }
        ]
    )
    snapshot.to_csv(args.output_dir / "permutation_null_snapshot.csv", index=False)


if __name__ == "__main__":
    main()
