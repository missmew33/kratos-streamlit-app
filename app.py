import io
from pathlib import Path

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
# Helpers
# ------------------------------------------------------------
def load_demo_file() -> io.BytesIO | None:
    """Load demo_scopus.csv from data/ if it exists."""
    demo_path = Path(__file__).parent / "data" / "demo_scopus.csv"
    if demo_path.exists():
        return io.BytesIO(demo_path.read_bytes())
    return None


def _infer_gender(detector: gender.Detector, author: str) -> str:
    """Infer gender from first name, normalising 'andy' → 'unknown'."""
    if pd.isna(author) or not isinstance(author, str) or not author.strip():
        return "unknown"
    first_name = author.split()[0].replace("-", "")
    g = detector.get_gender(first_name)
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
    df[new_col] = df[author_col].astype(str).apply(lambda x: _infer_gender(det, x))
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

