"""Production Streamlit interface for KRATOS v2.2.

The application delegates all substantive metric computation to ``kratos_core``.
The substantive analytical universe is fixed at G=4 (female/male x Global
North/Global South). Unresolved demographic metadata remain audit states and are
reported through demographic coverage.

KCDI = H_D_prime**lambda * H_C_prime**(1-lambda)
P    = mean(A(u) * S(u)) over the four substantive cells
KJI  = KCDI * P

H_D_prime is normalised Shannon entropy of resolved document shares and
H_C_prime is normalised Shannon entropy of resolved citation shares. The former
range-normalised W_norm component is deprecated and is not computed here.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import hashlib
import json
from typing import Dict, Optional, Tuple

import country_converter as coco
import pandas as pd
import streamlit as st

from kratos_core import (
    ALL_GROUPS,
    GLOBAL_NORTH_ISO3,
    GenderComputerResolver,
    classify_region,
    compute_citation_concentration,
    compute_kratos_fixed_g,
    enrich_first_author_metadata,
    extract_first_author,
    extract_given_name,
)

APP_VERSION = "2.2.0"
_CC = coco.CountryConverter()

AUTHOR_CANDIDATES = [
    "Author full names",
    "Author_Full_Names",
    "Authors",
    "Author(s)",
    "First_Author",
    "First Author",
]
AFFILIATION_CANDIDATES = [
    "Authors with affiliations",
    "Affiliations",
    "Affiliation",
]
COUNTRY_CANDIDATES = [
    "First_Author_Country",
    "First Author Country",
    "Country",
]
YEAR_CANDIDATES = ["Year", "Publication Year", "Pub Year"]
WEIGHT_CANDIDATES = ["Cited by", "Citations", "Times Cited", "Citation Count"]


@st.cache_resource
def _gender_resolver() -> GenderComputerResolver:
    return GenderComputerResolver()


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lowered = {str(col).casefold(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    return None


def _read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Read a Scopus-style CSV using delimiter inference with UTF-8 fallbacks."""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
                on_bad_lines="skip",
            )
        except Exception as exc:  # pragma: no cover - depends on user input
            last_error = exc
    raise ValueError(f"CSV could not be parsed: {last_error}")


def _explicit_country_to_iso3(country: object) -> str:
    """Convert an explicit country field to ISO3.

    This helper is used only when the input already contains a country attributed
    to the first author. Raw affiliation strings are resolved by the stricter
    exact-country parser in ``kratos_core``.
    """
    if not isinstance(country, str) or not country.strip():
        return "unknown"
    text = country.strip()
    if text.casefold() in {"unknown", "nan", "none"}:
        return "unknown"
    try:
        iso3 = _CC.convert(names=text, to="ISO3", not_found=None)
    except Exception:
        return "unknown"
    if isinstance(iso3, (list, tuple)):
        iso3 = iso3[0] if iso3 else None
    if not isinstance(iso3, str):
        return "unknown"
    iso3 = iso3.strip().upper()
    return iso3 if len(iso3) == 3 and iso3.isalpha() else "unknown"


def _enrich_from_explicit_country(
    df: pd.DataFrame,
    author_col: str,
    country_col: str,
) -> pd.DataFrame:
    """Resolve first-author gender proxy and region from an explicit country field."""
    out = df.copy()
    resolver = _gender_resolver()

    first_authors = []
    given_names = []
    countries = []
    iso3_values = []
    regions = []
    genders = []
    gender_raw = []
    gender_methods = []
    gender_status = []
    groups = []

    for _, row in out.iterrows():
        author_value = row.get(author_col, "")
        first_author = extract_first_author(author_value)
        given_name = extract_given_name(author_value)
        country = str(row.get(country_col) or "").strip()
        if not country or country.casefold() == "nan":
            country = "unknown"
        iso3 = _explicit_country_to_iso3(country)
        region = classify_region(iso3)
        result = resolver.resolve(
            first_author,
            given_name,
            None if country == "unknown" else country,
        )
        gender = result.get("category", "unknown")
        if gender not in {"female", "male", "unknown"}:
            gender = "unknown"

        first_authors.append(first_author)
        given_names.append(given_name)
        countries.append(country)
        iso3_values.append(iso3)
        regions.append(region)
        genders.append(gender)
        gender_raw.append(result.get("raw_result", ""))
        gender_methods.append(result.get("method", "unknown"))
        gender_status.append(result.get("status", "unresolved"))
        groups.append(f"{gender} x {region}")

    out["first_author"] = first_authors
    out["given_name"] = given_names
    out["country"] = countries
    out["country_iso3"] = iso3_values
    out["region"] = regions
    out["region_method"] = "precomputed_first_author_country"
    out["gender_category"] = genders
    out["gender_raw_result"] = gender_raw
    out["gender_method"] = gender_methods
    out["gender_resolution_status"] = gender_status
    out["group"] = groups
    return out


def _prepare_corpus(
    df: pd.DataFrame,
    *,
    author_col: Optional[str],
    affiliations_col: Optional[str],
    country_col: Optional[str],
    weight_col: str,
) -> pd.DataFrame:
    """Create the auditable G=4 demographic layer required by KRATOS."""
    if weight_col not in df.columns:
        raise ValueError(f"Citation column not found: {weight_col}")

    out = df.copy()
    out["_weight_numeric"] = (
        pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    # Already enriched record-level audit files can be analysed directly.
    if {"group", "gender_category", "region"}.issubset(out.columns):
        return out

    if not author_col or author_col not in out.columns:
        raise ValueError(
            "An author-name column is required unless the file already contains "
            "KRATOS demographic audit fields."
        )

    if affiliations_col and affiliations_col in out.columns:
        enriched = enrich_first_author_metadata(
            out,
            author_col=author_col,
            authors_with_affiliations_col=affiliations_col,
            gender_resolver=_gender_resolver(),
        )
        enriched["_weight_numeric"] = out["_weight_numeric"].to_numpy()
        return enriched

    if country_col and country_col in out.columns:
        enriched = _enrich_from_explicit_country(out, author_col, country_col)
        enriched["_weight_numeric"] = out["_weight_numeric"].to_numpy()
        return enriched

    raise ValueError(
        "Demographic resolution requires either 'Authors with affiliations', "
        "an explicit first-author country field, or precomputed KRATOS audit fields."
    )


def _analyse_corpus(
    enriched: pd.DataFrame,
    *,
    lambda_param: float,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    group_table, details = compute_kratos_fixed_g(
        enriched,
        group_col="group",
        weight_col="_weight_numeric",
        lambda_param=lambda_param,
    )
    concentration = compute_citation_concentration(enriched["_weight_numeric"])
    return group_table, details, concentration


def _summary_record(
    name: str,
    details: Dict[str, float],
    concentration: Dict[str, float],
) -> Dict[str, float | str]:
    return {
        "Corpus": name,
        "N": int(details["n_docs_input"]),
        "N_resolved": int(details["n_docs_resolved"]),
        "Coverage": details["demographic_coverage"],
        "H_D_prime": details["H_D_prime"],
        "H_C_prime": details["H_C_prime"],
        "KCDI": details["KCDI"],
        "P": details["P"],
        "R": details["R"],
        "KJI": details["KJI"],
        "Gini": concentration["Gini"],
        "HHI": concentration["HHI"],
        "Top10_share": concentration["top10_share"],
    }


def _snapshot(
    *,
    lambda_param: float,
    mappings: Dict[str, Optional[str]],
    summaries: list[Dict[str, float | str]],
) -> str:
    payload = {
        "app_version": APP_VERSION,
        "generated_at": datetime.now().isoformat(),
        "measurement_regime": {
            "substantive_G": 4,
            "groups": list(ALL_GROUPS),
            "parity_reference": 0.25,
            "unknown_treatment": "audit/coverage state; excluded from primary G=4 calculation",
            "gender_interpretation": "metadata-derived proxy, not self-identified gender",
            "geography_rule": "first author; first listed affiliation when raw Scopus affiliation is used",
            "global_north_n_iso3": len(GLOBAL_NORTH_ISO3),
            "H_D_prime": "normalised Shannon entropy of resolved document shares over fixed G=4",
            "H_C_prime": "normalised Shannon entropy of resolved citation shares over fixed G=4",
            "KCDI": "H_D_prime^lambda * H_C_prime^(1-lambda)",
            "P": "mean(A(u)*S(u)) over fixed G=4",
            "KJI": "KCDI * P",
            "R": "1 - P",
            "Delta": "KCDI-KJI = KCDI*(1-P); derived identity, not an empirical test",
        },
        "lambda": lambda_param,
        "column_mappings": mappings,
        "corpora": summaries,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _metric_row(details: Dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", f"{int(details['n_docs_input']):,}")
    c2.metric("Resolved", f"{int(details['n_docs_resolved']):,}")
    c3.metric("Coverage", f"{details['demographic_coverage']:.1%}")
    c4.metric("KJI", f"{details['KJI']:.3f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("H′D", f"{details['H_D_prime']:.3f}")
    c2.metric("H′C", f"{details['H_C_prime']:.3f}")
    c3.metric("KCDI", f"{details['KCDI']:.3f}")
    c4.metric("P", f"{details['P']:.3f}")


def main() -> None:
    st.set_page_config(page_title="KRATOS", page_icon="K", layout="wide")
    st.title("KRATOS bibliometric recognition diagnostic")
    st.caption(
        "G=4 substantive universe; unresolved demographic metadata are retained as audit states. "
        "KJI is a measurement diagnostic and should not be interpreted as a direct ranking of epistemic justice."
    )

    with st.sidebar:
        st.header("Analysis")
        lambda_param = st.slider(
            "KCDI balance parameter (lambda)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Primary manuscript specification: 0.50. Sensitivity analyses use alternative values.",
        )
        uploads = st.file_uploader(
            "Upload one or more Scopus/KRATOS CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )
        st.caption(f"Version {APP_VERSION}")

    if not uploads:
        st.info(
            "Upload a CSV containing citation counts plus either raw Scopus first-author affiliations, "
            "an explicit first-author country field, or precomputed KRATOS demographic audit fields."
        )
        return

    try:
        first_bytes = uploads[0].getvalue()
        first_df = _read_csv_bytes(first_bytes)
    except Exception as exc:
        st.error(str(exc))
        return

    inferred_author = _find_column(first_df, AUTHOR_CANDIDATES)
    inferred_aff = _find_column(first_df, AFFILIATION_CANDIDATES)
    inferred_country = _find_column(first_df, COUNTRY_CANDIDATES)
    inferred_weight = _find_column(first_df, WEIGHT_CANDIDATES)

    with st.expander("Column mapping", expanded=not bool(inferred_weight)):
        columns = list(first_df.columns)
        author_options = ["<pre-enriched / none>"] + columns
        author_default = (
            author_options.index(inferred_author) if inferred_author in author_options else 0
        )
        author_selection = st.selectbox("Author names", author_options, index=author_default)
        author_col = None if author_selection.startswith("<") else author_selection

        aff_options = ["<none>"] + columns
        aff_default = aff_options.index(inferred_aff) if inferred_aff in aff_options else 0
        aff_selection = st.selectbox("Authors with affiliations", aff_options, index=aff_default)
        affiliations_col = None if aff_selection == "<none>" else aff_selection

        country_options = ["<none>"] + columns
        country_default = (
            country_options.index(inferred_country) if inferred_country in country_options else 0
        )
        country_selection = st.selectbox("First-author country", country_options, index=country_default)
        country_col = None if country_selection == "<none>" else country_selection

        if not columns:
            st.error("The uploaded CSV has no columns.")
            return
        weight_default = columns.index(inferred_weight) if inferred_weight in columns else 0
        weight_col = st.selectbox("Citation count", columns, index=weight_default)

    mappings = {
        "author": author_col,
        "affiliations": affiliations_col,
        "country": country_col,
        "weight": weight_col,
    }

    analyses: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]] = {}
    summaries: list[Dict[str, float | str]] = []

    for upload in uploads:
        name = upload.name.rsplit(".", 1)[0]
        try:
            raw = _read_csv_bytes(upload.getvalue())
            for selected in (author_col, affiliations_col, country_col, weight_col):
                if selected and selected not in raw.columns:
                    raise ValueError(
                        f"{upload.name}: mapped column '{selected}' is absent. "
                        "Use files with a common schema or analyse them separately."
                    )
            enriched = _prepare_corpus(
                raw,
                author_col=author_col,
                affiliations_col=affiliations_col,
                country_col=country_col,
                weight_col=weight_col,
            )
            group_table, details, concentration = _analyse_corpus(
                enriched, lambda_param=lambda_param
            )
        except Exception as exc:
            st.error(f"{upload.name}: {exc}")
            continue

        analyses[name] = (raw, enriched, details, concentration)
        summaries.append(_summary_record(name, details, concentration))

    if not analyses:
        return

    if len(analyses) > 1:
        st.subheader("Cross-corpus comparison")
        comparison = pd.DataFrame(summaries)
        display = comparison.copy()
        for col in ["Coverage", "H_D_prime", "H_C_prime", "KCDI", "P", "R", "KJI", "Gini", "HHI", "Top10_share"]:
            display[col] = display[col].astype(float).round(4)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Corpus ordering should be interpreted only after the common-window, matched-size, "
            "parameter, and unresolved-metadata sensitivity analyses used in the manuscript."
        )

    for name, (raw, enriched, details, concentration) in analyses.items():
        st.divider()
        st.subheader(name)
        _metric_row(details)

        c1, c2, c3 = st.columns(3)
        c1.metric("Citation Gini", f"{concentration['Gini']:.3f}")
        c2.metric("Citation HHI", f"{concentration['HHI']:.4f}")
        c3.metric("Top 10% citation share", f"{concentration['top10_share']:.1%}")

        groups, _, _ = _analyse_corpus(enriched, lambda_param=lambda_param)
        group_display = groups[
            [
                "Group",
                "n_docs",
                "doc_share",
                "total_weight",
                "citation_share",
                "A_factor",
                "S_factor",
                "A_times_S",
            ]
        ].copy()
        for col in ["doc_share", "citation_share", "A_factor", "S_factor", "A_times_S"]:
            group_display[col] = group_display[col].round(4)
        st.dataframe(group_display, use_container_width=True, hide_index=True)

        if details["demographic_coverage"] < 0.8:
            st.warning(
                "Demographic coverage is below 80%. Primary KCDI/KJI values are complete-case diagnostics; "
                "unresolved-metadata sensitivity should be consulted before cross-corpus interpretation."
            )

        with st.expander("Metric definitions"):
            st.latex(r"H'_D=-\frac{\sum_{u=1}^{4} p_u\ln p_u}{\ln 4}")
            st.latex(r"H'_C=-\frac{\sum_{u=1}^{4} s_u\ln s_u}{\ln 4}")
            st.latex(r"KCDI=(H'_D)^{\lambda}(H'_C)^{1-\lambda}")
            st.latex(r"A(u)=\max\left(0,1-\frac{|p_u-1/4|}{1/4}\right)")
            st.latex(r"S(u)=\max\left(0,1-\left|\frac{s_u}{p_u}-1\right|\right)")
            st.latex(r"P=\frac{1}{4}\sum_{u=1}^{4}A(u)S(u),\qquad KJI=KCDI\,P")
            st.markdown(
                "`H′D` measures evenness of resolved document participation; `H′C` measures evenness "
                "of resolved citation shares. `P` evaluates participation-recognition alignment. "
                "`KJI <= KCDI` follows from the index architecture and is not evidence of canonical closure, "
                "epistemic stigma, or epistemic injustice."
            )

        export_cols = [
            col
            for col in [
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
                "_weight_numeric",
            ]
            if col in enriched.columns
        ]
        audit_csv = enriched[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download demographic audit CSV",
            data=audit_csv,
            file_name=f"{name}_kratos_demographic_audit.csv",
            mime="text/csv",
            key=f"audit_{name}",
        )

    snapshot_json = _snapshot(
        lambda_param=lambda_param,
        mappings=mappings,
        summaries=summaries,
    )
    st.download_button(
        "Download analysis snapshot JSON",
        data=snapshot_json.encode("utf-8"),
        file_name=f"kratos_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

    st.caption(
        "Source Scopus records remain local to the user session and are not committed to the public repository."
    )


if __name__ == "__main__":
    main()
