"""Interactive robustness diagnostics for KRATOS v2.3.

This Streamlit page exposes the global resolved-set permutation baseline and
composition diagnostics without changing the primary KRATOS measurement
architecture implemented in ``app.py`` and ``kratos_core.py``. The archived M5
workflow additionally includes biennium- and exact-year conditioning and design-
resolution assessment through ``kratos_diagnostics.py``.
"""

from __future__ import annotations

from datetime import datetime
import json

import pandas as pd
import streamlit as st

import app as base
from kratos_diagnostics import (
    composition_diagnostics,
    null_snapshot,
    permutation_null_resolved,
    summarise_permutation_null,
)


st.set_page_config(page_title="KRATOS robustness", page_icon="K", layout="wide")
st.title("KRATOS robustness diagnostics")
st.caption(
    "Resolved-set global null calibration and composition diagnostics for the fixed G=4 regime. "
    "The Scientometrics M5 release uses B=50,000 for the principal global, biennium, and exact-year calibrations."
)

with st.sidebar:
    st.header("Calibration settings")
    lambda_param = st.slider(
        "KCDI balance parameter (lambda)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    B = st.select_slider(
        "Permutation draws",
        options=[500, 1000, 5000, 10000, 50000],
        value=50000,
        help=(
            "Locked Scientometrics manuscript specification: 50,000 draws. "
            "Lower values are exploratory only."
        ),
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=20260831,
        step=1,
    )
    uploads = st.file_uploader(
        "Upload one or more Scopus/KRATOS CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

if not uploads:
    st.info(
        "Upload raw Scopus CSVs or record-level KRATOS demographic-audit CSVs. "
        "For manuscript-grade null calibration, use the exact frozen audit files so that "
        "the resolved analytical set D_R is identical to the primary analysis."
    )
    st.stop()

try:
    first_df = base._read_csv_bytes(uploads[0].getvalue())
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

run = st.button("Run robustness diagnostics", type="primary")
if not run:
    st.stop()

all_summaries = []
all_composition = []
export_payload = {
    "generated_at": datetime.now().isoformat(),
    "B": int(B),
    "seed": int(seed),
    "lambda": float(lambda_param),
    "null_design": (
        "citation permutation within the exact resolved G=4 analytical set; "
        "N_R, group labels/group sizes, and the empirical citation distribution are fixed"
    ),
    "corpora": [],
}

for upload in uploads:
    name = upload.name.rsplit(".", 1)[0]
    st.divider()
    st.subheader(name)

    try:
        raw = base._read_csv_bytes(upload.getvalue())
        enriched = base._prepare_corpus(
            raw,
            author_col=author_col,
            affiliations_col=affiliations_col,
            country_col=country_col,
            weight_col=weight_col,
        )
        observed, draws = permutation_null_resolved(
            enriched,
            B=int(B),
            seed=int(seed),
            group_col="group",
            weight_col="_weight_numeric",
            lambda_param=float(lambda_param),
        )
        summary = summarise_permutation_null(observed, draws)
        composition = composition_diagnostics(
            enriched,
            group_col="group",
            weight_col="_weight_numeric",
        )
        snapshot = null_snapshot(
            df=enriched,
            B=int(B),
            seed=int(seed),
            group_col="group",
            weight_col="_weight_numeric",
            lambda_param=float(lambda_param),
        )
    except Exception as exc:
        st.error(f"{upload.name}: {exc}")
        continue

    summary.insert(0, "Corpus", name)
    all_summaries.append(summary)
    all_composition.append(
        {
            "Corpus": name,
            "A_bar": composition["A_bar"],
            "P_S": composition["P_S"],
            "N_resolved": int(composition["n_resolved"]),
        }
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Resolved N", f"{int(composition['n_resolved']):,}")
    c2.metric("Composition ceiling A_bar", f"{composition['A_bar']:.3f}")
    c3.metric("Recognition-only diagnostic P_S", f"{composition['P_S']:.3f}")

    display = summary.copy()
    numeric = [
        "observed",
        "null_median",
        "null_p025",
        "null_p975",
        "observed_minus_null_median",
        "lower_tail_mc_p",
        "upper_tail_mc_p",
        "null_sd",
        "null_z",
        "tie_share",
    ]
    for col in numeric:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(4)
    st.dataframe(display, use_container_width=True, hide_index=True)

    p_row = summary.loc[summary["metric"] == "P"].iloc[0]
    if p_row["null_position"] == "below 95% null range":
        st.info(
            "Observed P lies below the corpus-specific 95% permutation range. "
            "This indicates stronger participation-recognition misalignment than is typical "
            "under exchangeability of citation counts across the resolved groups. It does not "
            "identify a causal mechanism, discrimination, epistemic stigma, or epistemic injustice."
        )
    elif p_row["null_position"] == "within 95% null range":
        st.info(
            "Observed P lies within the corpus-specific 95% permutation range. The observed "
            "misalignment is not clearly separated from the resolved-set exchangeability baseline."
        )
    else:
        st.info(
            "Observed P lies above the corpus-specific 95% permutation range, indicating greater "
            "alignment than is typical under the resolved-set exchangeability baseline."
        )

    c1, c2 = st.columns(2)
    c1.download_button(
        "Download null summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}_kratos_null_summary.csv",
        mime="text/csv",
        key=f"null_summary_{name}",
    )
    c2.download_button(
        "Download null draws CSV",
        data=draws.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}_kratos_null_draws.csv",
        mime="text/csv",
        key=f"null_draws_{name}",
    )

    export_payload["corpora"].append(
        {
            "name": name,
            "snapshot": snapshot,
            "composition": composition,
            "null_summary": summary.drop(columns=["Corpus"]).to_dict(orient="records"),
        }
    )

if all_summaries:
    st.divider()
    st.subheader("Cross-corpus calibrated comparison")
    combined = pd.concat(all_summaries, ignore_index=True)
    kji = combined.loc[combined["metric"] == "KJI"].copy()
    ptab = combined.loc[combined["metric"] == "P"].copy()

    left, right = st.columns(2)
    with left:
        st.markdown("**KJI relative to corpus-specific null**")
        kji_display = kji[
            [
                "Corpus",
                "observed",
                "null_median",
                "null_p025",
                "null_p975",
                "observed_minus_null_median",
                "null_position",
            ]
        ].copy()
        for col in ["observed", "null_median", "null_p025", "null_p975", "observed_minus_null_median"]:
            kji_display[col] = pd.to_numeric(kji_display[col], errors="coerce").round(4)
        st.dataframe(kji_display, use_container_width=True, hide_index=True)

    with right:
        st.markdown("**P relative to corpus-specific null**")
        p_display = ptab[
            [
                "Corpus",
                "observed",
                "null_median",
                "null_p025",
                "null_p975",
                "observed_minus_null_median",
                "lower_tail_mc_p",
                "null_position",
            ]
        ].copy()
        for col in ["observed", "null_median", "null_p025", "null_p975", "observed_minus_null_median", "lower_tail_mc_p"]:
            p_display[col] = pd.to_numeric(p_display[col], errors="coerce").round(4)
        st.dataframe(p_display, use_container_width=True, hide_index=True)

    st.caption(
        "Use null-centred differences and permutation positions to calibrate raw cross-corpus estimates. "
        "Do not interpret Monte Carlo tail probabilities as evidence of a causal demographic effect."
    )

    composition_df = pd.DataFrame(all_composition)
    st.markdown("**Composition diagnostics**")
    for col in ["A_bar", "P_S"]:
        composition_df[col] = composition_df[col].astype(float).round(4)
    st.dataframe(composition_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download robustness snapshot JSON",
    data=json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name=f"kratos_robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
)

st.caption(
    "For publication use, archive the exact record-level audit CSV, software commit, seed, draw count, "
    "and output snapshot used to generate the reported results."
)
