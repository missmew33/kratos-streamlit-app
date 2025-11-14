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

def read_scopus_csv(file_obj) -> pd.DataFrame:
    """
    Robust reader for Scopus-style CSV files.
    Tries comma-separated first; if it fails, tries semicolon.
    Skips badly formatted lines instead of raising a ParserError.
    """
    # First attempt: comma-separated
    try:
        file_obj.seek(0)
        return pd.read_csv(
            file_obj,
            engine="python",
            on_bad_lines="skip"
        )
    except Exception:
        pass

    # Second attempt: semicolon-separated
    file_obj.seek(0)
    return pd.read_csv(
        file_obj,
        sep=";",
        engine="python",
        on_bad_lines="skip"
    )

def extract_first_author_given_name(raw: str) -> str:
    """
    Extract the given name of the first author from a Scopus-style
    'Authors' or 'Author full names' field.

    Examples of inputs:
    - "Bao, Guotai"
    - "Divya; M., Narwal, Mahabir"
    - "Jog, Deepti; N.A., Alcasoas, Nelissa Andrea"
    """
    if pd.isna(raw) or not isinstance(raw, str):
        return ""

    s = raw.strip()

    # Some Scopus exports include line breaks; keep only the first line
    s = s.splitlines()[0]

    # 1) Take the first author (authors separated by ';')
    first_author = s.split(";")[0].strip()

    # 2) Scopus pattern is usually "Surname, Given names"
    if "," in first_author:
        parts = first_author.split(",", 1)
        given_part = parts[1].strip()
    else:
        given_part = first_author

    # 3) Remove IDs, parentheses, numbers
    given_part = re.sub(r"\(.*?\)", "", given_part)
    given_part = re.sub(r"\d", "", given_part)

    # 4) Split into tokens, remove punctuation, keep tokens with ≥2 letters
    tokens = given_part.replace("-", " ").split()
    clean_tokens = [
        re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", t) for t in tokens
    ]
    clean_tokens = [t for t in clean_tokens if len(t) >= 2]

    if not clean_tokens:
        return ""

    # Often the last token is the actual given name ("Mahabir", "Deepti")
    return clean_tokens[-1]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_demo_file() -> io.BytesIO | None:
    """Load demo_scopus.csv from data/ if it exists."""
    demo_path = Path(__file__).parent / "data" / "demo_scopus.csv"
    if demo_path.exists():
        return io.BytesIO(demo_path.read_bytes())
    return None


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
    # Some Scopus exports include line breaks; keep only the first line
    s = s.splitlines()[0]

    # 1) First author (authors separated by ';')
    first_author = s.split(";")[0].strip()

    # 2) Usually "Surname, Given names"
    if "," in first_author:
        parts = first_author.split(",", 1)
        given_part = parts[1].strip()
    else:
        given_part = first_author

    # 3) Remove IDs, parentheses, digits
    given_part = re.sub(r"\(.*?\)", "", given_part)
    given_part = re.sub(r"\d", "", given_part)

    # 4) Clean tokens, keep those with ≥2 letters
    tokens = given_part.replace("-", " ").split()
    clean_tokens = [re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", t) for t in tokens]
    clean_tokens = [t for t in clean_tokens if len(t) >= 2]

    if not clean_tokens:
        return ""

    # Often the last token is the actual given name
    return clean_tokens[-1]


def _infer_gender(detector: gender.Detector, author: str) -> str:
    """
    Infer gender from the first author's given name using gender-guesser.
    Ambiguous ('andy') is mapped to 'unknown'.
    """
    given_name = extract_first_author_given_name(author)
    if not given_name:
        return "unknown"

    g = detector.get_gender(given_name)
    if g == "andy":
        return "unknown"
    return g


def add_gender_column(df: pd.DataFrame,
                      author_col: str = "Authors",
                      new_col: str = "gender") -> pd.DataFrame:
    det = gender.Detector(case_sensitive=False)
    df = df.copy()

    if author_col not in df.columns:
        st.warning(f"Column '{author_col}' not found. Gender set to 'unknown'.")
        df[new_col] = "unknown"
        return df

    # Infer gender for each row
    df[new_col] = df[author_col].astype(str).apply(
        lambda x: _infer_gender(det, x)
    )

    # Collapse 'mostly_female'/'mostly_male' into female/male
    mapping = {
        "mostly_female": "female",
        "mostly_male": "male",
    }
    df[new_col] = df[new_col].replace(mapping)

    return df


def extract_country_from_affiliation(s: str) -> str:
    """
    Very simple heuristic: take the last comma-separated element
    as country name (e.g. 'Evora, Portugal' → 'Portugal').
    """
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts[-1] if parts else ""


def ensure_country_column(df: pd.DataFrame,
                          country_col: str = "Country") -> pd.DataFrame:
    """
    If 'Country' does not exist but 'Affiliations' does, create it
    from the last element of the affiliation string.
    """
    df = df.copy()
    if country_col in df.columns:
        return df
    if "Affiliations" in df.columns:
        df[country_col] = df["Affiliations"].astype(str).apply(
            extract_country_from_affiliation
        )
        return df
    st.warning(
        "No 'Country' or 'Affiliations' column found; region will be 'Unknown'."
    )
    df[country_col] = ""
    return df


def add_region_column(df: pd.DataFrame,
                      country_col: str = "Country",
                      new_col: str = "region") -> pd.DataFrame:
    df = df.copy()
    if country_col not in df.columns:
        st.warning(f"Column '{country_col}' not found. Region set to 'Unknown'.")
        df[new_col] = "Unknown"
        return df

    iso3 = coco.convert(df[country_col].tolist(), to="ISO3")
    df["_iso3"] = iso3

    north = {
        "USA", "CAN", "GBR", "FRA", "DEU", "ESP", "ITA", "NLD", "SWE", "NOR",
        "DNK", "CHE", "AUS", "NZL", "JPN", "FIN", "BEL", "AUT", "IRL", "PRT"
    }

    def classify(code):
        if not isinstance(code, str) or code in {"", "nan", "None"}:
            return "Unknown"
        if code in north:
            return "Global North"
        return "Global South"

    df[new_col] = df["_iso3"].apply(classify)
    df.drop(columns=["_iso3"], inplace=True)
    return df


# ------------------------------------------------------------
# KCDI
# ------------------------------------------------------------
def compute_kcdi(
    df: pd.DataFrame,
    group_cols: list[str],
    weight_col: str | None = None,
    lambda_entropy: float = 0.5
) -> pd.DataFrame:
    """
    KCDI(u) = H'(u)^λ * W_norm(u)^(1-λ),
    with H' normalised Shannon entropy and W_norm min–max of mean weight.
    """
    data = df.copy()

    if weight_col is None or weight_col not in data.columns:
        data["_weight"] = 1.0
        weight_col = "_weight"

    grouped = data.groupby(group_cols)

    records = []
    for group_values, subdf in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        weights = pd.to_numeric(
            subdf[weight_col], errors="coerce"
        ).fillna(0.0).values
        total_w = weights.sum()

        if total_w <= 0 or len(weights) == 0:
            H_prime = 0.0
            W_bar = 0.0
        else:
            p = weights / total_w
            H = -np.sum(p * np.log(p + 1e-12))
            n = len(p)
            H_prime = H / np.log(n) if n > 1 else 0.0
            W_bar = float(weights.mean())

        record = {
            **{col: val for col, val in zip(group_cols, group_values)},
            "H_prime": H_prime,
            "W_bar": W_bar,
        }
        records.append(record)

    res = pd.DataFrame(records)
    if res.empty:
        res["W_norm"] = []
        res["KCDI"] = []
        return res

    w_min = res["W_bar"].min()
    w_max = res["W_bar"].max()
    if w_max > w_min:
        res["W_norm"] = (res["W_bar"] - w_min) / (w_max - w_min)
    else:
        res["W_norm"] = 1.0

    lam = float(lambda_entropy)
    lam = max(0.0, min(1.0, lam))

    res["KCDI"] = (res["H_prime"] ** lam) * (res["W_norm"] ** (1.0 - lam))
    return res


# ------------------------------------------------------------
# KJI / KRATOS
# ------------------------------------------------------------
def compute_kratos(
    df: pd.DataFrame,
    group_cols: list[str],
    weight_col: str | None = None,
    lambda_entropy: float = 0.5
) -> pd.DataFrame:
    """
    Returns per group:
    H_prime, W_bar, W_norm, KCDI,
    A_factor, S_factor, KJI.
    """
    data = df.copy()

    if weight_col is None or weight_col not in data.columns:
        data["_weight"] = 1.0
        weight_col = "_weight"

    g = data.groupby(group_cols)
    summary = g[weight_col].agg(
        n_docs="count",
        total_weight="sum",
        mean_weight="mean"
    ).reset_index()

    if summary.empty:
        cols = group_cols + [
            "H_prime", "W_bar", "W_norm", "KCDI",
            "A_factor", "S_factor", "KJI"
        ]
        return pd.DataFrame(columns=cols)

    kcdi_df = compute_kcdi(
        df=data,
        group_cols=group_cols,
        weight_col=weight_col,
        lambda_entropy=lambda_entropy
    )

    res = pd.merge(
        kcdi_df,
        summary[group_cols + ["n_docs", "total_weight"]],
        on=group_cols,
        how="left"
    )

    N = res["n_docs"].sum()
    G = (res["n_docs"] > 0).sum()
    if N <= 0 or G == 0:
        res["A_factor"] = 0.0
    else:
        p_star = 1.0 / G

        def participation_fairness(nu: float) -> float:
            pu = nu / N
            return max(0.0, 1.0 - abs(pu - p_star) / p_star)

        res["A_factor"] = res["n_docs"].apply(participation_fairness)

    C = res["total_weight"].sum()
    if C <= 0:
        res["S_factor"] = 0.0
    else:

        def recognition_fairness(row) -> float:
            nu = row["n_docs"]
            cu = row["total_weight"]
            if nu <= 0:
                return 0.0
            pu = nu / N if N > 0 else 0.0
            su = cu / C if C > 0 else 0.0
            if pu <= 0:
                return 0.0
            r = su / pu
            return max(0.0, 1.0 - abs(r - 1.0))

        res["S_factor"] = res.apply(recognition_fairness, axis=1)

    res["KJI"] = res["KCDI"] * res["A_factor"] * res["S_factor"]
    res = res.sort_values(by="KJI", ascending=False).reset_index(drop=True)
    return res


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def plot_kcdi_bar(df: pd.DataFrame, gender_col="gender", region_col="region"):
    if df.empty:
        st.info("No KCDI data to plot.")
        return
    df = df.copy()
    df["group"] = df[gender_col].astype(str) + " – " + df[region_col].astype(str)
    df = df.sort_values("KCDI", ascending=False)

    fig = px.bar(df, x="group", y="KCDI", text="KCDI")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Group (gender – region)",
        yaxis_title="KCDI",
        xaxis_tickangle=-45,
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_kji_bar(df: pd.DataFrame, gender_col="gender", region_col="region"):
    if df.empty:
        st.info("No KJI data to plot.")
        return
    df = df.copy()
    df["group"] = df[gender_col].astype(str) + " – " + df[region_col].astype(str)
    df = df.sort_values("KJI", ascending=False)

    fig = px.bar(df, x="group", y="KJI", text="KJI")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Group (gender – region)",
        yaxis_title="KJI (Knowledge Justice Index)",
        xaxis_tickangle=-45,
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_group_share(df: pd.DataFrame,
                     gender_col: str = "gender",
                     region_col: str = "region"):
    """
    Show how many documents belong to each gender–region group (counts + share).
    """
    if df.empty:
        st.info("No data to plot.")
        return

    g = (
        df.groupby([gender_col, region_col])
          .size()
          .reset_index(name="n_docs")
    )
    total = g["n_docs"].sum()
    if total == 0:
        st.info("No documents found for group share plot.")
        return

    g["share"] = g["n_docs"] / total
    g["group"] = g[gender_col].astype(str) + " – " + g[region_col].astype(str)
    g = g.sort_values("share", ascending=False)

    fig = px.bar(g, x="group", y="share", text="n_docs")
    fig.update_traces(texttemplate="%{text} docs", textposition="outside")
    fig.update_layout(
        xaxis_title="Group (gender – region)",
        yaxis_title="Share of documents",
        xaxis_tickangle=-45,
        yaxis_tickformat=".0%",
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_gender_trend(df: pd.DataFrame):
    """
    Trend: share of documents by gender across years.
    Requires a 'Year' column.
    """
    if "Year" not in df.columns:
        st.info("No 'Year' column found; cannot compute temporal trends.")
        return

    d = df.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d = d.dropna(subset=["Year"])
    if d.empty:
        st.info("Year values are missing or invalid; cannot compute trends.")
        return

    d["Year"] = d["Year"].astype(int)

    g = (
        d.groupby(["Year", "gender"])
         .size()
         .reset_index(name="n_docs")
    )
    g["total_year"] = g.groupby("Year")["n_docs"].transform("sum")
    g["share"] = g["n_docs"] / g["total_year"]

    fig = px.line(
        g,
        x="Year",
        y="share",
        color="gender",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Share of documents",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Streamlit layout
# ------------------------------------------------------------
def main():
    st.title("KRATOS – Knowledge Justice Analytics for Scholarly Data")

    st.markdown(
        """
        This app implements your **KRATOS / KJI framework** to analyse
        epistemic diversity and justice across gender and geography.
        """
    )

    st.sidebar.header("1. Data input")
    use_demo = st.sidebar.checkbox("Use demo_scopus.csv from data/", value=True)
    uploaded = st.sidebar.file_uploader(
        "Upload a Scopus-style CSV file",
        type=["csv"]
    )

    if use_demo and uploaded is None:
        uploaded = load_demo_file()

    if uploaded is None:
        st.info("Upload a CSV file or enable the demo dataset to start.")
        return

    # IMPORTANT: robust CSV reader
    df = read_scopus_csv(uploaded)

    st.subheader("Raw data (first 5 rows)")
    st.dataframe(df.head(), use_container_width=True)

    st.sidebar.header("2. Columns and parameters")
    author_col = st.sidebar.text_input("Author column", "Authors")
    country_col = st.sidebar.text_input("Country column", "Country")
    citations_col = st.sidebar.text_input("Citations / impact column", "Cited by")

    lambda_entropy = st.sidebar.slider(
        "λ (weight of entropy in KCDI)",
        0.0, 1.0, 0.5, 0.05
    )

    # Enrich data
    df = ensure_country_column(df, country_col=country_col)
    df = add_gender_column(df, author_col=author_col, new_col="gender")
    df = add_region_column(df, country_col=country_col, new_col="region")

    st.subheader("Enriched data (author, gender, country, region)")
    cols_show = [c for c in [author_col, "gender", country_col, "region"] if c in df.columns]
    st.dataframe(df[cols_show].head(), use_container_width=True)

tab_kcdi, tab_kji, tab_trends, tab_authors, tab_notes = st.tabs(
    ["KCDI", "KJI / KRATOS", "Trends", "Author / Institution explorer", "Methodological notes"]
)


    with tab_kcdi:
        st.markdown("### KCDI by gender and region")
        kcdi_table = compute_kcdi(
            df,
            group_cols=["gender", "region"],
            weight_col=citations_col if citations_col in df.columns else None,
            lambda_entropy=lambda_entropy
        )
        st.dataframe(kcdi_table, use_container_width=True)
        plot_kcdi_bar(kcdi_table)
                st.markdown("#### Share of documents by gender–region")
        plot_group_share(df, gender_col="gender", region_col="region")


    with tab_kji:
        st.markdown("### KJI – Knowledge Justice Index")
        kratos_table = compute_kratos(
            df,
            group_cols=["gender", "region"],
            weight_col=citations_col if citations_col in df.columns else None,
            lambda_entropy=lambda_entropy
        )
        st.dataframe(kratos_table, use_container_width=True)
        plot_kji_bar(kratos_table)
    with tab_trends:
        st.markdown("### Temporal trends")
        st.markdown("Share of documents by gender across years.")
        plot_gender_trend(df)

    with tab_notes:
        st.markdown(
            """
            **Summary**

            - KCDI combines normalised Shannon entropy and normalised mean
              weight (e.g. citations), controlled by λ.
            - KJI multiplies KCDI by two fairness factors:
              participation (A) and recognition (S).
            - Groups are defined here as intersections of gender and region.
            """
        )


if __name__ == "__main__":
    main()

    with tab_authors:
        st.markdown("### Author / Institution explorer")

        search = st.text_input(
            "Filter by author name, institution or ID (partial match)",
            "",
        )

        # Columns to show if available
        cols_interest = [
            "Authors",
            "Author full names",
            "Author(s) ID",
            "Affiliations",
            "gender",
            "region",
            "Cited by",
            "Year",
            "Source title",
        ]
        cols_available = [c for c in cols_interest if c in df.columns]
        df_show = df[cols_available].copy()

        if search:
            s = search.lower()
            mask = False
            for col in ["Authors", "Author full names", "Affiliations", "Author(s) ID"]:
                if col in df_show.columns:
                    m_col = df_show[col].astype(str).str.lower().str.contains(s)
                    mask = m_col if isinstance(mask, bool) else (mask | m_col)
            if not isinstance(mask, bool):
                df_show = df_show[mask]

        st.dataframe(df_show.head(500), use_container_width=True)
        st.caption("Showing up to 500 rows (filtered).")
