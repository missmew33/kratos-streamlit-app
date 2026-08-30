#!/usr/bin/env python3
"""Sensitivity analyses for the substantive KRATOS G=4 measurement regime.

Input is a record-level demographic audit produced by kratos_analyze_csv.py.
No licensed source data are stored in the repository.

The script provides two diagnostics:
1. stratified stochastic imputation of unresolved gender, conditional on the
   observed female/male distribution within each known region;
2. matched-size resampling of documents before KRATOS computation.

The imputation is a measurement-sensitivity scenario, not an estimate of an
individual author's self-identified gender.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kratos_core import compute_kratos_fixed_g


METRICS = [
    "demographic_coverage",
    "H_D_prime",
    "H_C_prime",
    "KCDI",
    "P",
    "R",
    "KJI",
]


def summarise(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "median": draws[metric].median(),
                "p025": draws[metric].quantile(0.025),
                "p975": draws[metric].quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def stratified_gender_sensitivity(
    df: pd.DataFrame,
    *,
    B: int,
    seed: int,
    weight_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute only unresolved gender for records with known GN/GS geography.

    Within each region, female/male draws follow the observed resolved-gender
    proportion in that corpus. Geography-unknown records remain unresolved.
    """
    rng = np.random.default_rng(seed)
    probabilities: dict[str, float] = {}
    for region in ("Global North", "Global South"):
        resolved = df[(df["region"] == region) & df["gender_category"].isin(["female", "male"])]
        probabilities[region] = (
            float((resolved["gender_category"] == "female").mean()) if len(resolved) else 0.5
        )

    rows = []
    for draw in range(B):
        work = df.copy()
        gender = work["gender_category"].copy()
        for region, p_female in probabilities.items():
            mask = (work["region"] == region) & (gender == "unknown")
            gender.loc[mask] = np.where(
                rng.random(int(mask.sum())) < p_female,
                "female",
                "male",
            )
        work["group_sensitivity"] = gender.astype(str) + " x " + work["region"].astype(str)
        _, details = compute_kratos_fixed_g(
            work,
            group_col="group_sensitivity",
            weight_col=weight_col,
        )
        rows.append({"draw": draw, **{metric: details[metric] for metric in METRICS}})

    probability_table = pd.DataFrame(
        [
            {"region": region, "observed_resolved_female_share": p}
            for region, p in probabilities.items()
        ]
    )
    return pd.DataFrame(rows), probability_table


def matched_size_sensitivity(
    df: pd.DataFrame,
    *,
    n: int,
    B: int,
    seed: int,
    weight_col: str,
) -> pd.DataFrame:
    """Sample documents without replacement before G=4 computation."""
    if n > len(df):
        raise ValueError(f"matched n={n} exceeds corpus size {len(df)}")
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(B):
        if n == len(df):
            sample = df
        else:
            idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
            sample = df.loc[idx]
        _, details = compute_kratos_fixed_g(sample, weight_col=weight_col)
        rows.append({"draw": draw, **{metric: details[metric] for metric in METRICS}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("kratos_g4_sensitivity"))
    parser.add_argument("--weight-col", default="Cited by")
    parser.add_argument("--year-col", default="Year")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument("--matched-n", type=int, default=None)
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    df = pd.read_csv(args.audit_csv)
    if args.year_min is not None:
        years = pd.to_numeric(df[args.year_col], errors="coerce")
        df = df[(years >= args.year_min) & (years <= args.year_max)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    imputation_draws, probabilities = stratified_gender_sensitivity(
        df, B=args.B, seed=args.seed, weight_col=args.weight_col
    )
    imputation_draws.to_csv(args.output_dir / "gender_sensitivity_draws.csv", index=False)
    summarise(imputation_draws).to_csv(
        args.output_dir / "gender_sensitivity_summary.csv", index=False
    )
    probabilities.to_csv(args.output_dir / "gender_sensitivity_probabilities.csv", index=False)

    if args.matched_n is not None:
        matched = matched_size_sensitivity(
            df,
            n=args.matched_n,
            B=args.B,
            seed=args.seed + 1,
            weight_col=args.weight_col,
        )
        matched.to_csv(args.output_dir / "matched_size_draws.csv", index=False)
        summarise(matched).to_csv(args.output_dir / "matched_size_summary.csv", index=False)


if __name__ == "__main__":
    main()
