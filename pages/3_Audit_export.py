"""Record-level demographic audit export for KRATOS v2.3.

This page reconstructs the auditable G=4 demographic layer from raw Scopus
exports (or accepts already-enriched KRATOS files), reports the primary metrics,
and allows the exact record-level audit used by downstream robustness analyses
to be downloaded and archived.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import hashlib
import json

import pandas as pd
import streamlit as st

import app as base


st.set_page_config(page_title="KRATOS audit export", page_icon="K", layout="wide")
st.title("KRATOS demographic audit export")
st.caption(
    "Reconstruct and archive the exact record-level G=4 analytical layer before "
    "running manuscript-grade robustness diagnostics."
)

uploads = st.file_uploader(
    "Upload one or more Scopus/KRATOS CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploads:
    st.info(
        "Upload the frozen corpus CSVs. The page resolves first-author geography and "
        "the metadata-derived gender proxy using the production KRATOS pipeline, then "
        "exports the exact record-level audit used by the fixed G=4 calculation."
    )
    st.stop()

try:
    first_bytes = uploads[0].getvalue()
    first_df = base._read_csv_bytes(first_bytes)
except Exception as exc:
    st.error(str(exc))
    st.stop()

inferred_author = base._find_column(first_df, base.AUTHOR_CANDIDATES)
inferred_aff = base._find_column(first_df, base.AFFILIATION_CANDIDATES)
inferred_country = base._find_column(first_df, base.COUNTRY_CANDIDATES)
inferred_weight = base._find_column(first_df, base.WEIGHT_CANDIDATES)

with st.expander("Column mapping", expanded=not bool(inferred_weight)):
    columns = list(first_df.columns)
    if not columns:
        st.error("The uploaded CSV has no columns.")
        st.stop()

    author_options = ["<pre-enriched / none>"] + columns
    author_default = author_options.index(inferred_author) if inferred_author in author_options else 0
    author_selection = st.selectbox("Author names", author_options, index=author_default)
    author_col = None if author_selection.startswith("<") else author_selection

    aff_options = ["<none>"] + columns
    aff_default = aff_options.index(inferred_aff) if inferred_aff in aff_options else 0
    aff_selection = st.selectbox("Authors with affiliations", aff_options, index=aff_default)
    affiliations_col = None if aff_selection == "<none>" else aff_selection

    country_options = ["<none>"] + columns
    country_default = country_options.index(inferred_country) if inferred_country in country_options else 0
    country_selection = st.selectbox("First-author country", country_options, index=country_default)
    country_col = None if country_selection == "<none>" else country_selection

    weight_default = columns.index(inferred_weight) if inferred_weight in columns else 0
    weight_col = st.selectbox("Citation count", columns, index=weight_default)

lambda_param = st.slider(
    "KCDI balance parameter (lambda)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

run = st.button("Build demographic audits", type="primary")
if not run:
    st.stop()

manifest = {
    "generated_at": datetime.now().isoformat(),
    "lambda": float(lambda_param),
    "column_mapping": {
        "author": author_col,
        "affiliations": affiliations_col,
        "country": country_col,
        "weight": weight_col,
    },
    "corpora": [],
}

for upload in uploads:
    name = upload.name.rsplit(".", 1)[0]
    st.divider()
    st.subheader(name)

    try:
        raw_bytes = upload.getvalue()
        raw = base._read_csv_bytes(raw_bytes)
        for selected in (author_col, affiliations_col, country_col, weight_col):
            if selected and selected not in raw.columns:
                raise ValueError(
                    f"{upload.name}: mapped column '{selected}' is absent. "
                    "Use files with a common schema or analyse them separately."
                )

        enriched = base._prepare_corpus(
            raw,
            author_col=author_col,
            affiliations_col=affiliations_col,
            country_col=country_col,
            weight_col=weight_col,
        )
        group_table, details, concentration = base._analyse_corpus(
            enriched,
            lambda_param=float(lambda_param),
        )
    except Exception as exc:
        st.error(f"{upload.name}: {exc}")
        continue

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input N", f"{int(details['n_docs_input']):,}")
    c2.metric("Resolved N", f"{int(details['n_docs_resolved']):,}")
    c3.metric("Coverage", f"{details['demographic_coverage']:.1%}")
    c4.metric("KJI", f"{details['KJI']:.3f}")

    metric_table = pd.DataFrame(
        [
            {
                "H_D_prime": details["H_D_prime"],
                "H_C_prime": details["H_C_prime"],
                "KCDI": details["KCDI"],
                "P": details["P"],
                "KJI": details["KJI"],
                "Gini": concentration["Gini"],
                "HHI": concentration["HHI"],
                "Top10_share": concentration["top10_share"],
            }
        ]
    ).round(6)
    st.dataframe(metric_table, use_container_width=True, hide_index=True)

    audit_id_cols = [
        col
        for col in ["EID", "DOI", "Title", "Year", "Source title", "Document Type"]
        if col in enriched.columns
    ]
    audit_resolution_cols = [
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
    audit_cols = audit_id_cols + [c for c in audit_resolution_cols if c not in audit_id_cols]
    audit = enriched[audit_cols].copy()

    resolved = audit[audit.get("group", pd.Series(index=audit.index, dtype=str)).isin(base.ALL_GROUPS)].copy()
    stable_cols = [c for c in ["EID", "DOI", "group", "_weight_numeric"] if c in resolved.columns]
    if stable_cols:
        stable_payload = resolved[stable_cols].fillna("").astype(str).sort_values(stable_cols).to_csv(index=False)
        resolved_set_sha256 = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()
    else:
        resolved_set_sha256 = ""

    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    st.code(
        f"input_sha256={input_sha256}\nresolved_set_sha256={resolved_set_sha256}",
        language="text",
    )

    st.download_button(
        "Download demographic audit CSV",
        data=audit.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}_kratos_demographic_audit.csv",
        mime="text/csv",
        key=f"audit_{name}",
    )

    manifest["corpora"].append(
        {
            "name": name,
            "input_sha256": input_sha256,
            "resolved_set_sha256": resolved_set_sha256,
            "n_input": int(details["n_docs_input"]),
            "n_resolved": int(details["n_docs_resolved"]),
            "coverage": float(details["demographic_coverage"]),
            "H_D_prime": float(details["H_D_prime"]),
            "H_C_prime": float(details["H_C_prime"]),
            "KCDI": float(details["KCDI"]),
            "P": float(details["P"]),
            "KJI": float(details["KJI"]),
        }
    )

st.divider()
st.download_button(
    "Download audit manifest JSON",
    data=json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name=f"kratos_audit_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
)

st.caption(
    "Archive the audit CSVs and manifest together with the software commit used for the manuscript. "
    "Raw licensed Scopus records remain outside the public repository."
)
