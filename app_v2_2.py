"""KRATOS v2.2 release-candidate Streamlit entrypoint.

This migration entrypoint preserves the existing UI while replacing the legacy
metadata enrichment and variable-G metric functions with the auditable pure
implementation in ``kratos_core.py``. It supports both raw Scopus exports and
KRATOS canonical/audit CSV files containing pre-resolved first-author country
metadata. It remains separate from ``app.py`` until the revised measurement
regime has been validated on the canonical corpora.
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
    classify_region,
    compute_kratos_fixed_g,
    enrich_first_author_metadata,
    extract_first_author,
    extract_given_name,
)


legacy.APP_VERSION = "2.2.0-rc1"
legacy.GLOBAL_NORTH = set(GLOBAL_NORTH_ISO3)
legacy.DEFAULT_COLUMNS = {
    # Canonical KRATOS exports use underscores; raw Scopus uses spaces.
    "author": [
        "Author_Full_Names",
        "Author full names",
        "Authors",
        "Author(s)",
        "Author Names",
    ],
    # Prefer a first-author country if already resolved. Never auto-map Region as Country.
    "country": [
        "First_Author_Country",
        "First Author Country",
        "Country",
    ],
    "affiliations": [
        "Authors with affiliations",
        "Affiliations",
        "Affiliation",
    ],
    "year": ["Year", "Publication Year", "Pub Year"],
    "weight": ["Cited by", "Citations", "Times Cited", "Citation Count"],
    "source": ["Source_Title", "Source title", "Source Title", "Journal", "Publication Name"],
}

_GENDER_RESOLVER = None


def _get_gender_resolver():
    global _GENDER_RESOLVER
    if _GENDER_RESOLVER is None:
        _GENDER_RESOLVER = GenderComputerResolver()
    return _GENDER_RESOLVER


def _country_to_iso3(country: object) -> str:
    """Resolve a country label already attributed to the first author."""
    if not isinstance(country, str) or not country.strip() or country.strip().lower() == "unknown":
        return "unknown"
    try:
        iso3 = legacy.coco.CountryConverter().convert(
            names=country.strip(), to="ISO3", not_found=None
        )
    except Exception:
        return "unknown"
    if isinstance(iso3, (list, tuple)):
        iso3 = iso3[0] if iso3 else None
    if iso3 is None:
        return "unknown"
    iso3 = str(iso3).strip()
    return iso3 if len(iso3) == 3 and iso3.isalpha() else "unknown"


def _enrich_from_precomputed_first_author_country(
    df: pd.DataFrame,
    author_col: str,
    country_col: str,
) -> pd.DataFrame:
    """Enrich a canonical KRATOS CSV that already contains first-author country.

    This path is intentionally distinct from raw-Scopus affiliation parsing. The
    country is accepted only from an explicitly first-author-labelled field and
    the provenance is recorded as ``precomputed_first_author_country``.
    """
    out = df.copy()
    resolver = _get_gender_resolver()

    first_authors = []
    given_names = []
    countries = []
    iso3_values = []
    regions = []
    gender_categories = []
    gender_raw = []
    gender_methods = []
    gender_status = []
    groups = []

    has_given_name = "First_Author_Given_Name" in out.columns

    for _, row in out.iterrows():
        full_names = row.get(author_col, "")
        first_author = extract_first_author(full_names)
        if has_given_name and isinstance(row.get("First_Author_Given_Name"), str):
            given_name = str(row.get("First_Author_Given_Name") or "").strip()
        else:
            given_name = extract_given_name(full_names)

        country = str(row.get(country_col) or "").strip()
        if not country or country.lower() == "nan":
            country = "unknown"
        iso3 = _country_to_iso3(country)
        region = classify_region(iso3)

        result = resolver.resolve(first_author, given_name, None if country == "unknown" else country)
        gender = result.get("category", "unknown")
        if gender not in {"female", "male", "unknown"}:
            gender = "unknown"

        first_authors.append(first_author)
        given_names.append(given_name)
        countries.append(country)
        iso3_values.append(iso3)
        regions.append(region)
        gender_categories.append(gender)
        gender_raw.append(result.get("raw_result", ""))
        gender_methods.append(result.get("method", "unknown"))
        gender_status.append(result.get("status", "unresolved"))
        groups.append(f"{gender} x {region}")

    out["first_author"] = first_authors
    out["given_name"] = given_names
    out["first_author_affiliation"] = ""
    out["country"] = countries
    out["country_iso3"] = iso3_values
    out["region"] = regions
    out["region_method"] = "precomputed_first_author_country"
    out["gender_category"] = gender_categories
    out["gender_raw_result"] = gender_raw
    out["gender_method"] = gender_methods
    out["gender_resolution_status"] = gender_status
    out["group"] = groups
    return out


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
    """Load data and apply first-author demographic resolution v2.

    Supported inputs:
    1. raw Scopus CSV with ``Authors with affiliations``;
    2. canonical KRATOS CSV with ``First_Author_Country`` (or an explicitly
       selected equivalent first-author country field).
    """
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

    df["_weight_numeric"] = df[weight_col].apply(legacy.parse_weight_safe)
    df["_year_numeric"] = df[year_col].apply(legacy.parse_year_safe)

    # Prefer raw author-linked affiliations when available. Otherwise accept a
    # precomputed first-author country field from a canonical/audit KRATOS CSV.
    author_aff_col = affiliations_col if affiliations_col in df.columns else None
    if author_aff_col is None and "Authors with affiliations" in df.columns:
        author_aff_col = "Authors with affiliations"

    if author_aff_col:
        enriched = enrich_first_author_metadata(
            df,
            author_col=author_col,
            authors_with_affiliations_col=author_aff_col,
            gender_resolver=_get_gender_resolver(),
        )
    elif country_col and country_col in df.columns and country_col.lower() not in {"region"}:
        enriched = _enrich_from_precomputed_first_author_country(
            df,
            author_col=author_col,
            country_col=country_col,
        )
    else:
        raise ValueError(
            "Demographic resolution v2 requires either Scopus 'Authors with affiliations' "
            "or an explicit first-author country field such as 'First_Author_Country'. "
            "Do not map a Region column as Country."
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
    # A(u) and S(u) do not depend on lambda. Derive those factors once from the
    # fixed-G core, then rescale group KJI with the KCDI supplied by the UI.
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
            "geography_unit": "first author; raw affiliation or precomputed first-author country",
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
