import io
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import gender_guesser.detector as gender
import country_converter as coco


# ------------------------------------------------------------
# Streamlit basic config
# ------------------------------------------------------------
st.set_page_config(
    page_title="KRATOS – Knowledge Justice Analytics",
    layout="wide"
)

# ------------------------------------------------------------
# Hero image (front banner)
# ------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent / "assets"
HERO_IMAGE = ASSETS_DIR / "kratos_front.png"

if HERO_IMAGE.exists():
    st.image(str(HERO_IMAGE), use_column_width=True)
else:
    st.info("Tip: añade 'assets/kratos_front.png' al repositorio para visualizar la portada.")


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def read_scopus_csv(file_obj) -> pd.DataFrame:
    """
    Robust reader for Scopus-style CSV files.

    - Tries comma-separated first; if it fails, tries semicolon.
    - Skips badly formatted lines instead of raising a ParserError.
    """
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj, engine="python", on_bad_lines="skip")
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep=";", engine="python", on_bad_lines="skip")


def load_demo_file() -> io.BytesIO | None:
    """Load demo_scopus.csv from data/ if it exists."""
    demo_path = Path(__file__).parent / "data" / "demo_scopus.csv"
    if demo_path.exists():
        return io.BytesIO(demo_path.read_bytes())
    return None


# ------------------------------------------------------------
# Author / gender / country / region helpers
# ------------------------------------------------------------
def extract_first_author_given_name(raw: str) -> str:
    """
    Extract the given name of the first author from a Scopus-style
    'Authors' or 'Author full names' field.

    Examples:
    - "Bao, Guotai"
    - "Divya; M., Narwal, Mahabir"
    - "Jog, Deepti; N.A., Alcasoas, Nelissa Andrea"
    """
    if pd.isna(raw) or not isinstance(raw, str):
        return ""

    s = str(raw).strip()
    s = s.splitlines()[0]  # keep first line only

    # First author (authors separated by ';')
    first_author = s.split(";")[0].strip()

    # Usually "Surname, Given names"
    if "," in first_author:
        given_part = first_author.split(",", 1)[1].strip()
    else:
        given_part = first_author.strip()

    # Remove parentheses content, digits
    given_part = re.sub(r"\(.*?\)", "", given_part)
    given_part = re.sub(r"\d", "", given_part)

    # Tokenise and clean
    tokens = [
        re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", t)
        for t in given_part.replace("-", " ").split()
    ]
    clean_tokens = [t for t in tokens if len(t) >= 2]

    if not clean_tokens:
        return ""

    # Often the last token is the given name
    return clean_tokens[-1]


def _infer_gender(detector: gender.Detector, author: str) -> str:
    """
    Infer gender from given name using gender-guesser.
    'andy' mapped to 'unknown'.
    """
    given_name = extract_first_author_given_name(author)
    if not given_name:
        return "unknown"

    g = detector.get_gender(given_name)
    if g == "andy":
        return "unknown"
    return g


def add_gender_column(
    df: pd.DataFrame,
    author_col: str,
    new_col: str = "gender",
) -> pd.DataFrame:
    det = gender.Detector(case_sensitive=False)
    out = df.copy()

    if author_col not in out.columns:
        st.warning(f"Column '{author_col}' not found. Gender set to 'unknown'.")
        out[new_col] = "unknown"
        return out

    out[new_col] = out[author_col].astype(str).apply(lambda x: _infer_gender(det, x))
    out[new_col] = out[new_col].replace({"mostly_female": "female", "mostly_male": "male"})
    return out


def extract_country_from_affiliation(s: str) -> str:
    """
    Simple heuristic: take the last comma-separated element
    as country name (e.g. 'Evora, Portugal' → 'Portugal').
    """
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts[-1] if parts else ""


def ensure_country_column(df: pd.DataFrame, country_col: str = "Country") -> pd.DataFrame:
    """
    If 'Country' does not exist but 'Affiliations' does, create it
    from the last element of the affiliation string.
    """
    out = df.copy()
    if country_col in out.columns:
        return out

    if "Affiliations" in out.columns:
        out[country_col] = out["Affiliations"].astype(str).apply(extract_country_from_affiliation)
        return out

    st.warning("No 'Country' or 'Affiliations' column found; region will be 'Unknown'.")
    out[country_col] = ""
    return out


def add_region_column(
    df: pd.DataFrame,
    country_col: str = "Country",
    new_col: str = "region",
) -> pd.DataFrame:
    out = df.copy()

    if country_col not in out.columns:
        st.warning(f"Column '{country_col}' not found. Region set to 'Unknown'.")
        out[new_col] = "Unknown"
        return out

    iso3 = coco.convert(out[country_col].fillna("").astype(str).tolist(), to="ISO3")
    out["_iso3"] = iso3

    north = {
        "USA", "CAN", "GBR", "FRA", "DEU", "ESP", "ITA", "NLD", "SWE", "NOR",
        "DNK", "CHE", "AUS", "NZL", "JPN", "FIN", "BEL", "AUT", "IRL", "PRT"
    }

    def classify(code) -> str:
        if not isinstance(code, str):
            return "Unknown"
        c = code.strip()
        if c in {"", "not found", "nan", "None"}:
            return "Unknown"
        return "Global North" if c in north else "Global South"

    out[new_col] = out["_iso3"].apply(classify)
    out.drop(columns=["_iso3"], inplace=True)
    return out


def to_numeric_weight(series: pd.Series) -> pd.Series:
    """Coerce weight column to numeric, fill NA with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


# ------------------------------------------------------------
# CORE METRICS: KCDI, KJI (KRATOS), PPI (Pleonasm)
# ------------------------------------------------------------
def compute_kcdi(
    df: pd.DataFrame,
    group_cols: list[str],
    weight_col: str | None = None,
    lambda_entropy: float = 0.5,
) -> pd.DataFrame:
    """
    KCDI(u) = H'(u)^λ * W_norm(u)^(1-λ)

    H'  : normalised Shannon entropy on the within-group weight distribution
    W̄  : mean weight within group
    W_norm: min–max normalisation of W̄ across groups
    """
    data = df.copy()

    if weight_col is None or weight_col not in data.columns:
        data["_weight"] = 1.0
        weight_col = "_weight"
    else:
        data[weight_col] = to_numeric_weight(data[weight_col])

    records = []
    for gvals, sub in data.groupby(group_cols, dropna=False):
        if not isinstance(gvals, tuple):
            gvals = (gvals,)

        w = to_numeric_weight(sub[weight_col]).values
        total_w = float(w.sum())

        if total_w <= 0.0 or w.size == 0:
            h_prime = 0.0
            w_bar = 0.0
        else:
            p = w / total_w
            H = -np.sum(p * np.log(p + 1e-12))
            n = len(p)
            h_prime = (H / np.log(n)) if n > 1 else 0.0
            w_bar = float(w.mean())

        records.append(
            {
                **{c: v for c, v in zip(group_cols, gvals)},
                "H_prime": float(h_prime),
                "W_bar": float(w_bar),
            }
        )

    res = pd.DataFrame(records)
    if res.empty:
        return res

    w_min, w_max = float(res["W_bar"].min()), float(res["W_bar"].max())
    if w_max > w_min:
        res["W_norm"] = (res["W_bar"] - w_min) / (w_max - w_min)
    else:
        res["W_norm"] = 1.0

    lam = float(lambda_entropy)
    lam = max(0.0, min(1.0, lam))
    res["KCDI"] = (res["H_prime"] ** lam) * (res["W_norm"] ** (1.0 - lam))

    return res


def compute_kratos(
    df: pd.DataFrame,
    group_cols: list[str],
    weight_col: str | None = None,
    lambda_entropy: float = 0.5,
) -> pd.DataFrame:
    """
    Returns per group:
    H_prime, W_bar, W_norm, KCDI,
    A_factor, S_factor, KJI,
    plus n_docs and total_weight.
    """
    data = df.copy()

    if weight_col is None or weight_col not in data.columns:
        data["_weight"] = 1.0
        weight_col = "_weight"
    else:
        data[weight_col] = to_numeric_weight(data[weight_col])

    summary = (
        data.groupby(group_cols, dropna=False)[weight_col]
        .agg(n_docs="count", total_weight="sum", mean_weight="mean")
        .reset_index()
    )

    if summary.empty:
        return pd.DataFrame(
            columns=group_cols
            + ["H_prime", "W_bar", "W_norm", "KCDI", "n_docs", "total_weight", "A_factor", "S_factor", "KJI"]
        )

    kcdi_df = compute_kcdi(data, group_cols, weight_col=weight_col, lambda_entropy=lambda_entropy)
    res = pd.merge(kcdi_df, summary[group_cols + ["n_docs", "total_weight"]], on=group_cols, how="left")

    # Initialise
    res["A_factor"] = 0.0
    res["S_factor"] = 0.0

    N = float(res["n_docs"].sum())
    C = float(res["total_weight"].sum())
    G = int((res["n_docs"] > 0).sum())

    # Participation fairness (A)
    if N > 0 and G > 0:
        p_star = 1.0 / G

        def A(nu: float) -> float:
            pu = nu / N
            return max(0.0, 1.0 - abs(pu - p_star) / p_star)

        res["A_factor"] = res["n_docs"].astype(float).apply(A)

    # Recognition fairness (S)
    if N > 0 and C > 0:

        def S(row) -> float:
            nu = float(row["n_docs"])
            cu = float(row["total_weight"])
            if nu <= 0:
                return 0.0
            pu = nu / N
            su = cu / C
            if pu <= 0:
                return 0.0
            r = su / pu
            return max(0.0, 1.0 - abs(r - 1.0))

        res["S_factor"] = res.apply(S, axis=1)

    res["KJI"] = res["KCDI"] * res["A_factor"] * res["S_factor"]
    res = res.sort_values(by="KJI", ascending=False).reset_index(drop=True)
    return res


def pleonasm_hhi(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    return float(np.sum(p**2))


def top_share(weights: np.ndarray, top_pct: float = 0.10) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    w_sorted = np.sort(w)[::-1]
    k = max(1, int(np.ceil(top_pct * w_sorted.size)))
    return float(w_sorted[:k].sum() / w_sorted.sum())


def compute_pleonasm(
    df: pd.DataFrame,
    group_cols: list[str],
    weight_col: str | None = None,
) -> pd.DataFrame:
    """
    PPI (Pleonasm / canon concentration) diagnostics:
    - PPI_HHI: Herfindahl concentration on weights within group
    - Top10_share: share of total weight held by top 10% items within group
    """
    data = df.copy()

    if weight_col is None or weight_col not in data.columns:
        data["_weight"] = 1.0
        weight_col = "_weight"
    else:
        data[weight_col] = to_numeric_weight(data[weight_col])

    rows = []
    for gvals, sub in data.groupby(group_cols, dropna=False):
        if not isinstance(gvals, tuple):
            gvals = (gvals,)

        w = to_numeric_weight(sub[weight_col]).values

        rows.append(
            {
                **{c: v for c, v in zip(group_cols, gvals)},
                "PPI_HHI": pleonasm_hhi(w),
                "Top10_share": top_share(w, 0.10),
                "Top05_share": top_share(w, 0.05),
                "n_docs": int(sub.shape[0]),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Author summary
# ------------------------------------------------------------
def summarise_authors(df: pd.DataFrame, author_col: str, citations_col: str):
    """Aggregate at author level: docs, citations, modal gender/region."""
    d = df.copy()
    if author_col not in d.columns:
        return pd.DataFrame()

    if citations_col not in d.columns:
        d[citations_col] = 1.0

    d[citations_col] = to_numeric_weight(d[citations_col])

    g = (
        d.groupby(author_col)
        .agg(
            total_cites=(citations_col, "sum"),
            n_docs=("Year", "count") if "Year" in d.columns else (citations_col, "count"),
            gender_mode=(
                "gender",
                lambda x: x.mode().iat[0] if not x.mode().empty else "unknown",
            ),
            region_mode=(
                "region",
                lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown",
            ),
        )
        .reset_index()
    )
    return g


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    st.title("KRATOS – Knowledge Justice Analytics for Scholarly Data")
    st.markdown(
        """
        This app implements **KRATOS / KJI** to analyse epistemic diversity and justice across **gender-coded proxies**
        and **geographical position** (e.g., Global North/South proxy). The interface is designed to make key parameters
        and intermediate outputs auditable prior to interpretation.
        """
    )

    # ---------------- Sidebar: data input ----------------
    st.sidebar.header("1. Data input")
    use_demo = st.sidebar.checkbox("Use demo_scopus.csv from data/", value=True)
    uploaded = st.sidebar.file_uploader("Upload a Scopus-style CSV file", type=["csv"])

    if use_demo and uploaded is None:
        uploaded = load_demo_file()

    if uploaded is None:
        st.info("Upload a CSV file or enable the demo dataset to start.")
        return

    df_raw = read_scopus_csv(uploaded)

    st.subheader("Raw data (first 5 rows)")
    st.dataframe(df_raw.head(), use_container_width=True)

    # ---------------- Sidebar: parameters ----------------
    st.sidebar.header("2. Columns and parameters")
    author_col = st.sidebar.text_input("Author column", "Author full names")
    country_col = st.sidebar.text_input("Country column", "Country")
    citations_col = st.sidebar.text_input("Citations / impact column", "Cited by")

    lambda_entropy = st.sidebar.slider(
        "λ (weight of entropy in KCDI)",
        0.0,
        1.0,
        0.5,
        0.05,
    )

    # ---------------- Enrichment ----------------
    df = ensure_country_column(df_raw, country_col=country_col)
    df = add_gender_column(df, author_col=author_col, new_col="gender")
    df = add_region_column(df, country_col=country_col, new_col="region")

    st.subheader("Enriched data (author, gender, country, region)")
    cols_show = [c for c in [author_col, "gender", country_col, "region"] if c in df.columns]
    if cols_show:
        st.dataframe(df[cols_show].head(), use_container_width=True)
    else:
        st.warning("No expected columns available for enriched preview.")

    # Weight column resolution (single source of truth)
    weight_col = citations_col if citations_col in df.columns else None

    # Pre-compute tables once (so tabs can be visited in any order)
    kcdi_table = compute_kcdi(df, group_cols=["gender", "region"], weight_col=weight_col, lambda_entropy=lambda_entropy)
    kji_table = compute_kratos(df, group_cols=["gender", "region"], weight_col=weight_col, lambda_entropy=lambda_entropy)
    ppi_table = compute_pleonasm(df, group_cols=["gender", "region"], weight_col=weight_col)

    # ---------------- Tabs ----------------
    tab_kcdi, tab_kji, tab_pleo, tab_trends, tab_authors, tab_notes = st.tabs(
        [
            "KCDI",
            "KJI / KRATOS",
            "Pleonasm (PPI)",
            "Trends",
            "Author / Institution explorer",
            "Methodological notes",
        ]
    )

    # --- Tab: KCDI ---
    with tab_kcdi:
        st.markdown("### KCDI by gender and region")
        st.dataframe(kcdi_table, use_container_width=True)

        if not kcdi_table.empty:
            fig = px.bar(
                kcdi_table.sort_values("KCDI", ascending=False),
                x="region",
                y="KCDI",
                color="gender",
                barmode="group",
                hover_data=["H_prime", "W_bar", "W_norm"],
            )
            fig.update_layout(xaxis_title="Region", yaxis_title="KCDI")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No KCDI results available for the selected configuration.")

    # --- Tab: KJI / KRATOS ---
    with tab_kji:
        st.markdown("### KJI – Knowledge Justice Index")
        st.dataframe(kji_table, use_container_width=True)

        if not kji_table.empty:
            fig = px.scatter(
                kji_table,
                x="KCDI",
                y="KJI",
                size="n_docs",
                color="region",
                hover_data=["gender", "region", "A_factor", "S_factor", "n_docs", "total_weight"],
            )
            fig.update_layout(
                xaxis_title="KCDI (Epistemic diversity)",
                yaxis_title="KJI (Knowledge justice)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No KJI results available for the selected configuration.")

    # --- Tab: Pleonasm (PPI) ---
    with tab_pleo:
        st.markdown("### Pleonasm & canon concentration diagnostics (PPI)")
        st.dataframe(ppi_table, use_container_width=True)

        if not ppi_table.empty and not kji_table.empty:
            merged = ppi_table.merge(
                kji_table[["gender", "region", "S_factor", "KJI", "KCDI", "n_docs"]],
                on=["gender", "region"],
                how="left",
            )

            fig_diag = px.scatter(
                merged,
                x="PPI_HHI",
                y="S_factor",
                size="n_docs",
                color="region",
                hover_data=["gender", "region", "Top10_share", "Top05_share", "KJI", "KCDI"],
                title="Diagnostic: Pleonasm (HHI concentration) vs Recognition fairness (S)",
            )
            fig_diag.update_layout(
                xaxis_title="PPI_HHI (recognition concentration within group)",
                yaxis_title="S_factor (recognition fairness)",
            )
            st.plotly_chart(fig_diag, use_container_width=True)
        else:
            st.info("Pleonasm diagnostics require non-empty PPI and KJI tables.")

    # --- Tab: Trends ---
    with tab_trends:
        st.markdown("### Temporal trends")
        if "Year" not in df.columns:
            st.info("No 'Year' column found; temporal trends are unavailable.")
        else:
            d = df.copy()
            d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
            d = d.dropna(subset=["Year"])
            if d.empty:
                st.info("Year values are missing/invalid; temporal trends are unavailable.")
            else:
                d["Year"] = d["Year"].astype(int)
                trend = d.groupby(["Year", "gender"]).size().reset_index(name="n_docs")
                fig = px.line(trend, x="Year", y="n_docs", color="gender", markers=True)
                fig.update_layout(xaxis_title="Year", yaxis_title="Documents (count)")
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab: Author explorer ---
    with tab_authors:
        st.markdown("### Author / Institution explorer")

        search = st.text_input("Filter by author/institution/ID (partial match)", "")

        cols_interest = [
            "Authors",
            "Author full names",
            "Author(s) ID",
            "Affiliations",
            "gender",
            "region",
            citations_col,
            "Year",
            "Source title",
        ]
        cols_available = [c for c in cols_interest if c in df.columns]
        df_show = df[cols_available].copy() if cols_available else df.copy()

        if search:
            s = search.lower()
            mask = None
            for col in ["Authors", "Author full names", "Affiliations", "Author(s) ID"]:
                if col in df_show.columns:
                    m = df_show[col].astype(str).str.lower().str.contains(s, na=False)
                    mask = m if mask is None else (mask | m)
            if mask is not None:
                df_show = df_show[mask]

        st.dataframe(df_show.head(500), use_container_width=True)
        st.caption("Showing up to 500 rows (filtered).")

        # Top authors
        if author_col in df.columns:
            st.markdown("#### Top 20 authors by citations / impact proxy")
            author_summary = summarise_authors(df, author_col=author_col, citations_col=citations_col)
            if not author_summary.empty:
                top = author_summary.sort_values("total_cites", ascending=False).head(20)
                fig = px.bar(
                    top,
                    x=author_col,
                    y="total_cites",
                    color="gender_mode",
                    hover_data=["region_mode", "n_docs"],
                )
                fig.update_layout(xaxis_tickangle=-60, margin=dict(b=160))
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab: Methodological notes ---
    with tab_notes:
        st.markdown(
            """
            ### KRATOS / KJI methodological notes (operational)

            1. **KCDI (epistemic diversity)** combines:
               - **H'**: normalised Shannon entropy computed on the within-group weight distribution.
               - **W_norm**: min–max normalised mean weight across groups.
               - **λ** controls the relative emphasis of entropy vs weight.

            2. **KJI (Knowledge Justice Index)**:
               \\[
               KJI = KCDI \\times A \\times S
               \\]
               - **A (Participation fairness)**: proximity to parity in document share across groups.
               - **S (Recognition fairness)**: proximity to parity between document share and recognition share.

            3. **Pleonasm / canon concentration (PPI)**:
               - **PPI_HHI** captures concentration of recognition within each group.
               - **TopShare** quantifies dominance of the most recognised items (top 10% / 5%) within each group.

            Interpretation:
            - Citation/impact fields function as **recognition proxies inside the canon** (not direct measures of intrinsic quality).
            - Pleonasm diagnostics are treated as operational evidence compatible with canon closure and concentrated recognition regimes,
              and are used to interpret degradation in **S**.
            """
        )


if __name__ == "__main__":
    main()
