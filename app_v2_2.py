"""KRATOS v2.2 release-candidate Streamlit entrypoint.

This migration entrypoint preserves the existing UI while replacing the legacy
metadata enrichment and variable-G metric functions with the auditable pure
implementation in ``kratos_core.py``. It is intentionally separate from
``app.py`` until the revised measurement regime has been validated on the
canonical corpora.
"""

from io import BytesIO
import json
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

# Importing the legacy UI does not execute main() because it is not __main__.
# It does, however, configure the Streamlit page before any other UI call.
import app as legacy

from kratos_core import (
    ALL_GROUPS,
    GLOBAL_NORTH_ISO3,
    GenderComputerResolver,
    compute_kratos_fixed_g,
    enrich_first_author_metadata,
)


legacy.APP_VERSION = "2.2.0-rc1"
legacy.GLOBAL_NORTH = set(GLOBAL_NORTH_ISO3)
legacy.DEFAULT_COLUMNS = {
    "author": ["Author full names", "Authors", "Author(s)", "Author Names"],
    # Do not auto-map the author-linked affiliation field as a generic country.
    "country": ["Country", "First Author Country"],
    "affiliations": ["Authors with affiliations", "Affiliations", "Affiliation"],
    "year": ["Year", "Publication Year", "Pub Year"],
    "weight": ["Cited by", "Citations", "Times Cited", "Citation Count"],
    "source": ["Source title", "Source Title", "Journal", "Publication Name"],
}

_GENDER_RESOLVER = None


def _get_gender_resolver():
    global _GENDER_RESOLVER
    if _GENDER_RESOLVER is None:
        _GENDER_RESOLVER = GenderComputerResolver()
    return _GENDER_RESOLVER


def load_and_enrich_data_v2(
    file_bytes: bytes,
    file_name: str,
    author_col: str,
    country_col: Optional[str],
    affiliations_col: Optional[str],
    year_col: str,
    weight_col: str,
    source_col: Optional[str],
) -> pd.DataFrame:
    """Load Scopus data and apply first-author demographic resolution v2."""
    separator = legacy.detect_separator(file_bytes)
    df = pd.read_csv(
        BytesIO(file_bytes),
        sep=separator,
        encoding="utf-8",
        on_bad_lines="skip",
        dtype=str,
    )

    if df.columns.duplicated().any():
        df = legacy._make_unique_columns(df)

    required = [author_col, year_col, weight_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # For raw Scopus exports, author-linked affiliations are required to support
    # first-author geography. A generic aggregate Country column is not used to
    # overwrite first-author geography.
    author_aff_col = affiliations_col
    if not author_aff_col or author_aff_col not in df.columns:
        if "Authors with affiliations" in df.columns:
            author_aff_col = "Authors with affiliations"
        else:
            raise ValueError(
                "Demographic resolution v2 requires the Scopus 'Authors with affiliations' field "
                "to assign first-author geography reproducibly."
            )

    df["_weight_numeric"] = df[weight_col].apply(legacy.parse_weight_safe)
    df["_year_numeric"] = df[year_col].apply(legacy.parse_year_safe)

    enriched = enrich_first_author_metadata(
        df,
        author_col=author_col,
        authors_with_affiliations_col=author_aff_col,
        gender_resolver=_get_gender_resolver(),
    )

    # Compatibility aliases expected by the existing Streamlit views.
    enriched["_given_name"] = enriched["given_name"]
    enriched["Gender"] = enriched["gender_category"]
    enriched["Region"] = enriched["region"]
    enriched["Gender_Region"] = enriched["group"]

    enriched["First_Author"] = enriched["first_author"]
    enriched["First_Author_Affiliation"] = enriched["first_author_affiliation"]
    enriched["First_Author_Country"] = enriched["country"]
    enriched["Country_ISO3"] = enriched["country_iso3"]
    enriched["Gender_Raw_Result"] = enriched["gender_raw_result"]
    enriched["Gender_Method"] = enriched["gender_method"]
    enriched["Gender_Resolution_Status"] = enriched["gender_resolution_status"]
    enriched["Region_Method"] = enriched["region_method"]

    if source_col and source_col in enriched.columns:
        enriched["_source_title"] = enriched[source_col].astype(str)
    else:
        enriched["_source_title"] = "Unknown"

    return enriched


def compute_kcdi_corpus_v2(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str,
    lambda_param: float = 0.5,
):
    _, details = compute_kratos_fixed_g(df, group_col, weight_col, lambda_param)
    return details["KCDI"], {
        "H_prime": details["H_prime"],
        "W_norm": details["W_norm"],
        "n_groups": 9,
        "lambda": details["lambda"],
        "KCDI_corpus": details["KCDI"],
    }


def compute_group_justice_metrics_v2(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str,
    kcdi_corpus: float,
):
    # A(u) and S(u) do not depend on lambda. We derive those factors once from
    # the fixed-G core, then rescale group KJI with the KCDI supplied by the UI
    # so the selected lambda is respected consistently.
    groups, _ = compute_kratos_fixed_g(df, group_col, weight_col, 0.5)
    groups["KJI_group"] = kcdi_corpus * groups["A_factor"] * groups["S_factor"]
    groups["abs_gap"] = groups["signed_gap"].abs()
    return groups[
        [
            "Group",
            "n_docs",
            "doc_share",
            "total_weight",
            "weight_share",
            "signed_gap",
            "abs_gap",
            "A_factor",
            "S_factor",
            "KJI_group",
        ]
    ]


def compute_corpus_kji_v2(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str,
    lambda_param: float = 0.5,
):
    groups, details = compute_kratos_fixed_g(df, group_col, weight_col, lambda_param)
    return details["KJI"], {
        "H_prime": details["H_prime"],
        "W_norm": details["W_norm"],
        "n_groups": 9,
        "lambda": details["lambda"],
        "KCDI_corpus": details["KCDI"],
        "KJI_mean": details["KJI"],
        "A_mean": float(groups["A_factor"].mean()),
        "S_mean": float(groups["S_factor"].mean()),
        "P": details["P"],
        "R": details["R"],
        "Delta": details["Delta"],
    }


def generate_snapshot_export_v2(lambda_param: float, column_mappings: Dict, corpus_info: Dict[str, Dict]) -> str:
    snapshot = {
        "app_version": legacy.APP_VERSION,
        "timestamp": datetime.now().isoformat(),
        "lambda": lambda_param,
        "measurement_regime": {
            "group_universe": list(ALL_GROUPS),
            "G": 9,
            "parity_reference": "1/9",
            "geography_unit": "first author; first listed affiliation",
            "gender_method": "genderComputer@f626761; conservative unknown",
            "gender_interpretation": "metadata-derived proxy, not self-identified gender",
            "kji_identity": "KJI = KCDI * mean(A*S)",
        },
        "column_mappings": column_mappings,
        "global_north_hash": legacy.compute_global_north_hash(),
        "global_north_n_iso3": len(GLOBAL_NORTH_ISO3),
        "corpora": corpus_info,
    }
    return json.dumps(snapshot, indent=2)


# Replace legacy computational functions before running the existing UI.
legacy.load_and_enrich_data = load_and_enrich_data_v2
legacy.compute_kcdi_corpus = compute_kcdi_corpus_v2
legacy.compute_group_justice_metrics = compute_group_justice_metrics_v2
legacy.compute_corpus_kji = compute_corpus_kji_v2
legacy.generate_snapshot_export = generate_snapshot_export_v2


if __name__ == "__main__":
    legacy.main()
