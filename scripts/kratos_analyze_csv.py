#!/usr/bin/env python3
"""Run the revised KRATOS pipeline on a Scopus CSV export.

The script writes:
- record-level demographic audit CSV;
- four-cell substantive group metrics CSV;
- corpus summary JSON including demographic coverage and full-corpus citation concentration.

Scopus source data are not stored in the GitHub repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kratos_core import (
    GenderComputerResolver,
    compute_citation_concentration,
    compute_kratos_fixed_g,
    enrich_first_author_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("kratos_output"))
    parser.add_argument("--author-col", default="Author full names")
    parser.add_argument("--affiliation-col", default="Authors with affiliations")
    parser.add_argument("--weight-col", default="Cited by")
    parser.add_argument("--lambda-param", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv, dtype=str)

    resolver = GenderComputerResolver()
    enriched = enrich_first_author_metadata(
        df,
        author_col=args.author_col,
        authors_with_affiliations_col=args.affiliation_col,
        gender_resolver=resolver,
    )

    group_table, details = compute_kratos_fixed_g(
        enriched,
        group_col="group",
        weight_col=args.weight_col,
        lambda_param=args.lambda_param,
    )
    # Concentration is a document-level full-corpus diagnostic and therefore
    # does not depend on demographic resolution coverage.
    concentration = compute_citation_concentration(enriched[args.weight_col])

    audit_columns = [
        args.author_col,
        args.affiliation_col,
        "first_author",
        "given_name",
        "first_author_affiliation",
        "country",
        "country_iso3",
        "region",
        "region_method",
        "gender_category",
        "gender_raw_result",
        "gender_method",
        "gender_resolution_status",
        "group",
        args.weight_col,
    ]
    available = [column for column in audit_columns if column in enriched.columns]
    enriched[available].to_csv(args.output_dir / "demographic_audit.csv", index=False)
    group_table.to_csv(args.output_dir / "group_metrics.csv", index=False)

    snapshot = {
        "input_file": args.input_csv.name,
        "lambda": args.lambda_param,
        "demographic_method": "genderComputer@f626761 + exact first-author first-listed affiliation country",
        "measurement_regime": {
            "substantive_G": 4,
            "parity_reference": 0.25,
            "unknown_treatment": "audit/coverage state; excluded from primary parity universe",
            "document_component": "H_D_prime: normalised Shannon entropy of document shares over fixed G=4",
            "citation_component": "H_C_prime: normalised Shannon entropy of citation shares over fixed G=4",
            "kcdi_identity": "KCDI = H_D_prime^lambda * H_C_prime^(1-lambda)",
            "kji_identity": "KJI = KCDI * P; P = mean(A*S)",
        },
        "KRATOS": details,
        "citation_concentration_full_corpus": concentration,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
