#!/usr/bin/env python3
"""Resolved-set permutation baselines for KRATOS G=4 metrics.

Input is a record-level demographic audit produced by
``scripts/kratos_analyze_csv.py`` or exported by the Streamlit app. Licensed
Scopus source data are not stored in this repository.

The null calibration freezes the exact resolved G=4 analytical set, its four
group sizes, and the empirical citation-count distribution. Citation counts are
then permuted across the resolved records, removing only the association between
analytical group and citation outcome.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kratos_diagnostics import (
    composition_diagnostics,
    null_snapshot,
    permutation_null_resolved,
    summarise_permutation_null,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate resolved-set KRATOS permutation baselines."
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

    observed, draws = permutation_null_resolved(
        df,
        B=args.B,
        seed=args.seed,
        group_col=args.group_col,
        weight_col=args.weight_col,
        lambda_param=args.lambda_param,
    )
    summary = summarise_permutation_null(observed, draws)
    composition = composition_diagnostics(
        df,
        group_col=args.group_col,
        weight_col=args.weight_col,
    )
    snapshot = null_snapshot(
        df=df,
        B=args.B,
        seed=args.seed,
        group_col=args.group_col,
        weight_col=args.weight_col,
        lambda_param=args.lambda_param,
    )

    draws.to_csv(args.output_dir / "permutation_null_draws.csv", index=False)
    summary.to_csv(args.output_dir / "permutation_null_summary.csv", index=False)
    pd.DataFrame([composition]).to_csv(
        args.output_dir / "composition_diagnostics.csv", index=False
    )
    pd.DataFrame([snapshot]).to_csv(
        args.output_dir / "permutation_null_snapshot.csv", index=False
    )


if __name__ == "__main__":
    main()
