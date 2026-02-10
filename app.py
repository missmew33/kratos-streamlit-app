"""
KRATOS – Knowledge Justice Analytics
======================================
A Streamlit app for computational bibliometric analysis of Scopus CSV exports.

Implements epistemic justice metrics:
- KCDI: Knowledge Contribution Diversity Index (multiplicative)
- KJI/KRATOS: Knowledge Justice Index
- PPI: Pleonasm indicators (concentration diagnostics)

Author: KRATOS Team (M.D.G.BARBADO)(UOC)
License: MIT
Version: 2.1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gender_guesser.detector as gender
import country_converter as coco
from typing import Dict, List, Tuple, Optional
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

# ============================================================================
# PAGE CONFIG - MUST BE FIRST
# ============================================================================

st.set_page_config(
    page_title="KRATOS - Knowledge Justice Analytics",
    page_icon="⚖️",
    layout="wide"
)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

APP_VERSION = "2.1.0"

# Global North countries (ISO3 codes) - heuristic classification
GLOBAL_NORTH = {
    'USA', 'CAN', 'GBR', 'FRA', 'DEU', 'ITA', 'ESP', 'NLD', 'BEL', 'LUX',
    'CHE', 'AUT', 'SWE', 'NOR', 'DNK', 'FIN', 'ISL', 'IRL', 'PRT', 'GRC',
    'AUS', 'NZL', 'JPN', 'KOR', 'ISR', 'CZE', 'POL', 'HUN', 'SVK', 'SVN',
    'EST', 'LVA', 'LTU', 'CYP', 'MLT', 'HRV', 'SGP', 'HKG', 'TWN'
}

# Default column names for auto-detection
DEFAULT_COLUMNS = {
    'author': ['Author full names', 'Authors', 'Author(s)', 'Author Names'],
    'country': ['Country', 'Authors with affiliations'],
    'affiliations': ['Affiliations', 'Authors with affiliations', 'Affiliation'],
    'year': ['Year', 'Publication Year', 'Pub Year'],
    'weight': ['Cited by', 'Citations', 'Times Cited', 'Citation Count'],
    'source': ['Source title', 'Source Title', 'Journal', 'Publication Name']
}

# ============================================================================
# ROBUSTNESS HELPERS (DUPLICATE COLUMNS / SAFE DISPLAY)
# ============================================================================

def _dedupe_list_preserve_order(items: List[Optional[str]]) -> List[str]:
    """De-duplicate a list while preserving order; drops falsy values."""
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df.columns are unique by suffixing duplicates.
    Keeps all columns; required because Streamlit uses pyarrow which errors on duplicates.
    """
    cols = list(df.columns)
    counts: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in cols:
        if c not in counts:
            counts[c] = 0
            new_cols.append(c)
        else:
            counts[c] += 1
            new_cols.append(f"{c}__dup{counts[c]}")
    out = df.copy()
    out.columns = new_cols
    return out

def safe_dataframe_display(df: pd.DataFrame, cols: List[Optional[str]], n: int = 100):
    """
    Safe wrapper for st.dataframe that:
    - de-duplicates selected cols
    - keeps only existing cols
    - uses .loc to avoid pandas edge cases
    """
    cols_clean = _dedupe_list_preserve_order(cols)
    cols_clean = [c for c in cols_clean if c in df.columns]
    if not cols_clean:
        st.warning("No valid columns available for display.")
        return
    st.dataframe(df.loc[:, cols_clean].head(n), use_container_width=True)

def validate_column_mapping_or_stop(mapped: Dict[str, Optional[str]]):
    """
    Prevent mapping the same underlying column to multiple semantic roles.
    This avoids duplicate column selection lists and conceptual confusion.
    """
    vals = [v for v in mapped.values() if v is not None]
    dups = [v for v in set(vals) if vals.count(v) > 1]
    if dups:
        st.error(
            f"❌ Column mapping invalid: the same column was selected multiple times: {dups}. "
            "Please select distinct columns (or set optional fields to <None>)."
        )
        st.stop()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]

def compute_global_north_hash() -> str:
    """Compute hash of GLOBAL_NORTH set for reproducibility."""
    content = '|'.join(sorted(GLOBAL_NORTH))
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def detect_separator(file_content: bytes) -> str:
    """Detect CSV separator (comma or semicolon)."""
    sample = file_content[:5000].decode('utf-8', errors='ignore')
    comma_count = sample.count(',')
    semicolon_count = sample.count(';')
    return ';' if semicolon_count > comma_count else ','

def find_best_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    df_cols_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None

def extract_given_name(full_name: str) -> str:
    """
    Extract given name from author full name field.
    Handles formats like:
    - "Surname, Given Names"
    - "Given Names Surname"
    - Names with hyphens, diacritics, parentheses, IDs
    """
    if pd.isna(full_name) or not isinstance(full_name, str):
        return ""

    # Remove content in parentheses (e.g., author IDs)
    full_name = re.sub(r'\([^)]*\)', '', full_name)

    # Split by semicolon (multiple authors) and take first
    full_name = full_name.split(';')[0].strip()

    # Handle "Surname, Given Names" format
    if ',' in full_name:
        parts = full_name.split(',')
        if len(parts) >= 2:
            given_names = parts[1].strip()
        else:
            given_names = parts[0].strip()
    else:
        # Assume "Given Names Surname" format - take first word
        parts = full_name.split()
        given_names = parts[0] if parts else ""

    # Take first given name if multiple
    given_names = given_names.split()[0] if given_names else ""

    # Remove hyphens for better matching
    given_names = given_names.replace('-', ' ').split()[0] if given_names else ""

    # Remove non-alphabetic characters except basic diacritics
    given_names = re.sub(
        r'[^a-zA-ZáéíóúñÁÉÍÓÚÑäöüÄÖÜàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛ]',
        '',
        given_names
    )

    return given_names.strip()

@st.cache_data
def infer_gender_cached(names: List[str]) -> Dict[str, str]:
    """
    Cached gender inference to avoid recomputing for repeated names.
    Returns dict mapping name -> gender category.
    """
    detector = gender.Detector(case_sensitive=False)
    gender_map: Dict[str, str] = {}

    for name in set(names):  # Use set to avoid duplicates
        if not name:
            gender_map[name] = 'unknown'
            continue

        result = detector.get_gender(name)

        # Collapse mostly_* to binary
        if result in ['mostly_male', 'male']:
            gender_map[name] = 'male'
        elif result in ['mostly_female', 'female']:
            gender_map[name] = 'female'
        else:  # 'andy', 'unknown', or None
            gender_map[name] = 'unknown'

    return gender_map

def parse_weight_safe(value) -> float:
    """
    Safely parse weight/citation values to numeric.
    Handles: strings, NaN, "-", empty values, "None"
    """
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value in ['', '-', 'None', 'nan', 'NaN']:
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0

def parse_year_safe(value) -> Optional[int]:
    """Safely parse year to integer."""
    if pd.isna(value):
        return None
    try:
        year = int(float(value))
        if 1900 <= year <= 2100:
            return year
    except (ValueError, TypeError):
        pass
    return None

def ensure_country_column(
    df: pd.DataFrame,
    country_col: Optional[str],
    affiliations_col: Optional[str]
) -> pd.Series:
    """
    Ensure country column exists. Priority:
    1. Use country_col if provided and exists
    2. Extract from affiliations_col if exists (last comma token)
    3. Return None series

    Returns a Series with country names (may contain NaN).
    """
    if country_col and country_col in df.columns:
        return df[country_col]

    if affiliations_col and affiliations_col in df.columns:
        def extract_country(affiliation):
            if pd.isna(affiliation) or not isinstance(affiliation, str):
                return None
            tokens = affiliation.split(',')
            if tokens:
                return tokens[-1].strip()
            return None

        return df[affiliations_col].apply(extract_country)

    return pd.Series([None] * len(df))

def classify_region_safe(country_name) -> str:
    """
    Classify country into Global North/South/Unknown.
    Uses country_converter to get ISO3 code, then checks against GLOBAL_NORTH set.
    """
    if pd.isna(country_name) or not country_name:
        return 'Unknown'

    if not isinstance(country_name, str):
        country_name = str(country_name)

    cc = coco.CountryConverter()
    try:
        iso3 = cc.convert(names=country_name.strip(), to='ISO3', not_found=None)
        if iso3 is None or iso3 == country_name.strip() or iso3 == 'not found':
            return 'Unknown'
        return 'Global North' if iso3 in GLOBAL_NORTH else 'Global South'
    except Exception:
        return 'Unknown'

# ============================================================================
# THESIS-ALIGNED METRICS (MULTIPLICATIVE FORMULATION)
# ============================================================================

def compute_shannon_entropy_normalized(counts: pd.Series) -> float:
    """
    Compute normalized Shannon entropy H' ∈ [0,1].
    H' = H / log(k) where k is number of categories.
    Returns 0 if only one category or total is 0.
    """
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    H = -np.sum(proportions * np.log(proportions))
    H_max = np.log(len(counts))
    return H / H_max if H_max > 0 else 0.0

def compute_weight_normalization(weights: pd.Series) -> float:
    """
    Compute normalized weight intensity W_norm ∈ [0,1].
    W_norm = (W_mean - W_min) / (W_max - W_min)
    If all weights are equal, returns 1.0 by convention.
    """
    if len(weights) == 0:
        return 0.0
    w_min = weights.min()
    w_max = weights.max()
    w_mean = weights.mean()
    if w_max == w_min:
        return 1.0
    return (w_mean - w_min) / (w_max - w_min)

def compute_kcdi_corpus(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str,
    lambda_param: float = 0.5
) -> Tuple[float, Dict]:
    """
    Compute KCDI at CORPUS LEVEL (not within groups).

    THESIS-CONSISTENT FORMULA:
    KCDI = (H')^λ × (W_norm)^(1-λ)
    """
    if len(df) == 0:
        return 0.0, {'H_prime': 0.0, 'W_norm': 0.0, 'n_groups': 0}

    counts = df[group_col].value_counts()
    H_prime = compute_shannon_entropy_normalized(counts)
    group_weights = df.groupby(group_col)[weight_col].sum()
    W_norm = compute_weight_normalization(group_weights)

    kcdi = (H_prime ** lambda_param) * (W_norm ** (1 - lambda_param))

    details = {
        'H_prime': H_prime,
        'W_norm': W_norm,
        'n_groups': len(counts),
        'lambda': lambda_param
    }
    return kcdi, details

def compute_group_justice_metrics(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str,
    kcdi_corpus: float
) -> pd.DataFrame:
    """
    Parity-ideal A and S for each group, plus group-level KJI(u).
    """
    if len(df) == 0:
        return pd.DataFrame()

    N = len(df)
    C = df[weight_col].sum()

    group_counts = df[group_col].value_counts()
    G = len(group_counts)
    if G == 0:
        return pd.DataFrame()

    p_star = 1.0 / G

    group_stats = df.groupby(group_col).agg({
        weight_col: 'sum',
        group_col: 'count'
    }).rename(columns={group_col: 'n_docs', weight_col: 'total_weight'})

    group_stats['doc_share'] = group_stats['n_docs'] / N
    group_stats['weight_share'] = group_stats['total_weight'] / C if C > 0 else 0.0
    group_stats['signed_gap'] = group_stats['weight_share'] - group_stats['doc_share']
    group_stats['abs_gap'] = np.abs(group_stats['signed_gap'])

    group_stats['A_factor'] = group_stats['doc_share'].apply(
        lambda p_u: max(0.0, 1.0 - abs(p_u - p_star) / p_star)
    )

    def compute_S(row):
        p_u = row['doc_share']
        s_u = row['weight_share']
        if p_u == 0:
            return 0.0
        ratio = s_u / p_u
        return max(0.0, 1.0 - abs(ratio - 1.0))

    group_stats['S_factor'] = group_stats.apply(compute_S, axis=1)
    group_stats['KJI_group'] = kcdi_corpus * group_stats['A_factor'] * group_stats['S_factor']

    group_stats = group_stats.reset_index().rename(columns={group_col: 'Group'})
    return group_stats

def compute_corpus_kji(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str,
    lambda_param: float = 0.5
) -> Tuple[float, Dict]:
    """
    Compute overall corpus-level KJI as mean of group-level KJIs.
    """
    kcdi_corpus, kcdi_details = compute_kcdi_corpus(df, group_col, weight_col, lambda_param)
    group_metrics = compute_group_justice_metrics(df, group_col, weight_col, kcdi_corpus)

    if len(group_metrics) == 0:
        return 0.0, {**kcdi_details, 'KJI_mean': 0.0, 'A_mean': 0.0, 'S_mean': 0.0}

    KJI_mean = group_metrics['KJI_group'].mean()
    A_mean = group_metrics['A_factor'].mean()
    S_mean = group_metrics['S_factor'].mean()

    details = {
        **kcdi_details,
        'KJI_mean': KJI_mean,
        'A_mean': A_mean,
        'S_mean': S_mean,
        'KCDI_corpus': kcdi_corpus
    }
    return KJI_mean, details

def compute_ppi(df: pd.DataFrame, weight_col: str) -> Dict:
    """
    PPI indicators (SEPARATE):
    - HHI
    - Top10_share
    """
    if len(df) == 0:
        return {'HHI': 0.0, 'Top10_share': 0.0}

    weights = df[weight_col].copy()
    total_weight = weights.sum()
    if total_weight == 0:
        return {'HHI': 0.0, 'Top10_share': 0.0}

    shares = weights / total_weight
    HHI = float(np.sum(shares ** 2))

    weights_sorted = weights.sort_values(ascending=False)
    n_top = max(1, int(np.ceil(0.10 * len(weights))))
    top_weight = weights_sorted.head(n_top).sum()
    top10_share = float(top_weight / total_weight) if total_weight > 0 else 0.0

    return {
        'HHI': HHI,
        'Top10_share': top10_share,
        'n_items': int(len(weights)),
        'n_top': int(n_top)
    }

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_enrich_data(
    file_bytes: bytes,
    file_name: str,
    author_col: str,
    country_col: Optional[str],
    affiliations_col: Optional[str],
    year_col: str,
    weight_col: str,
    source_col: Optional[str]
) -> pd.DataFrame:
    """
    Load CSV and enrich with gender and region classifications.
    Cached based on file content and column selections.

    Raises ValueError if required columns missing (to be caught by main).
    """
    separator = detect_separator(file_bytes)

    from io import BytesIO
    df = pd.read_csv(
        BytesIO(file_bytes),
        sep=separator,
        encoding='utf-8',
        on_bad_lines='skip',
        dtype=str
    )

    # Critical robustness: Streamlit/pyarrow cannot handle duplicate column names.
    if df.columns.duplicated().any():
        df = _make_unique_columns(df)

    required_cols = [author_col, year_col, weight_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    df['_weight_numeric'] = df[weight_col].apply(parse_weight_safe)
    df['_year_numeric'] = df[year_col].apply(parse_year_safe)

    df['_given_name'] = df[author_col].apply(extract_given_name)
    all_names = df['_given_name'].tolist()
    gender_map = infer_gender_cached(all_names)
    df['Gender'] = df['_given_name'].map(gender_map)

    country_series = ensure_country_column(df, country_col, affiliations_col)
    df['Region'] = country_series.apply(classify_region_safe)

    df['Gender_Region'] = df['Gender'].astype(str) + ' × ' + df['Region'].astype(str)

    if source_col and source_col in df.columns:
        df['_source_title'] = df[source_col].astype(str)
    else:
        df['_source_title'] = 'Unknown'

    return df

def generate_gender_report(df: pd.DataFrame) -> pd.DataFrame:
    gender_counts = df['Gender'].value_counts()
    total = len(df)
    return pd.DataFrame({
        'Category': gender_counts.index,
        'Count': gender_counts.values,
        'Percentage': (gender_counts.values / total * 100).round(2)
    })

def generate_region_report(df: pd.DataFrame) -> pd.DataFrame:
    region_counts = df['Region'].value_counts()
    total = len(df)
    return pd.DataFrame({
        'Region': region_counts.index,
        'Count': region_counts.values,
        'Percentage': (region_counts.values / total * 100).round(2)
    })

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_group_justice_table(group_metrics: pd.DataFrame):
    if len(group_metrics) == 0:
        st.warning("No group data to display")
        return

    display_df = group_metrics[[
        'Group', 'n_docs', 'doc_share', 'total_weight', 'weight_share',
        'signed_gap', 'abs_gap', 'A_factor', 'S_factor', 'KJI_group'
    ]].copy()

    numeric_cols = ['doc_share', 'weight_share', 'signed_gap', 'abs_gap', 'A_factor', 'S_factor', 'KJI_group']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

def plot_kji_bars(group_metrics: pd.DataFrame):
    if len(group_metrics) == 0:
        st.warning("No data to plot")
        return None

    fig = px.bar(
        group_metrics,
        x='Group',
        y='KJI_group',
        title='Knowledge Justice Index (KJI) by Group',
        labels={'KJI_group': 'KJI', 'Group': 'Gender × Region'},
        color='KJI_group',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        margin=dict(b=120),
        showlegend=False
    )
    return fig

def plot_temporal_trends(df: pd.DataFrame):
    df_years = df[df['_year_numeric'].notna()].copy()
    if len(df_years) == 0:
        st.warning("No valid year data for temporal analysis")
        return None

    yearly = df_years.groupby(['_year_numeric', 'Gender']).agg({
        '_weight_numeric': 'sum',
        'Gender': 'count'
    }).rename(columns={'Gender': 'n_docs'}).reset_index()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Documents per Year', 'Citations per Year'),
        vertical_spacing=0.12
    )

    for g in yearly['Gender'].unique():
        subset = yearly[yearly['Gender'] == g]
        fig.add_trace(
            go.Scatter(
                x=subset['_year_numeric'],
                y=subset['n_docs'],
                name=f'{g} (docs)',
                mode='lines+markers',
                legendgroup=g
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=subset['_year_numeric'],
                y=subset['_weight_numeric'],
                name=f'{g} (cites)',
                mode='lines+markers',
                legendgroup=g,
                showlegend=False
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Number of Documents", row=1, col=1)
    fig.update_yaxes(title_text="Total Citations", row=2, col=1)
    fig.update_layout(height=700, title_text="Temporal Trends by Gender")
    return fig

def plot_ppi_diagnostic(df: pd.DataFrame, S_mean: float):
    ppi = compute_ppi(df, '_weight_numeric')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['HHI', 'Top 10% Share'],
        y=[ppi['HHI'], ppi['Top10_share']],
        text=[f"{ppi['HHI']:.3f}", f"{ppi['Top10_share']:.1%}"],
        textposition='auto'
    ))
    fig.add_hline(
        y=S_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"S̄ = {S_mean:.3f}",
        annotation_position="right"
    )
    fig.update_layout(
        title="Concentration Indicators (PPI) vs Recognition Fairness",
        yaxis_title="Value",
        height=400,
        yaxis_range=[0, 1]
    )
    return fig

def plot_concentration_scatter(corpus_level_df: pd.DataFrame):
    if len(corpus_level_df) < 3:
        return None

    fig = px.scatter(
        corpus_level_df,
        x='HHI',
        y='S_mean',
        color='Corpus',
        size='n_docs_total',
        hover_data=['n_docs_total', 'KJI_mean'],
        title='Concentration (HHI) vs Recognition Fairness (S̄) Across Corpora',
        labels={'HHI': 'HHI (Concentration)', 'S_mean': 'S̄ (Mean Recognition Fairness)'}
    )
    fig.update_layout(height=500)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig

# ============================================================================
# MULTI-CORPUS COMPARISON
# ============================================================================

def compare_corpora(corpus_data: Dict[str, pd.DataFrame], lambda_param: float) -> Tuple[pd.DataFrame, Dict]:
    comparison_rows = []
    corpus_summary = {}

    for corpus_name, df in corpus_data.items():
        kcdi_corpus, _ = compute_kcdi_corpus(df, 'Gender_Region', '_weight_numeric', lambda_param)
        kji_mean, kji_details = compute_corpus_kji(df, 'Gender_Region', '_weight_numeric', lambda_param)
        ppi = compute_ppi(df, '_weight_numeric')

        corpus_summary[corpus_name] = {
            'KCDI': kcdi_corpus,
            'KJI_mean': kji_mean,
            'A_mean': kji_details['A_mean'],
            'S_mean': kji_details['S_mean'],
            'HHI': ppi['HHI'],
            'Top10_share': ppi['Top10_share'],
            'n_docs_total': len(df)
        }

        group_metrics = compute_group_justice_metrics(df, 'Gender_Region', '_weight_numeric', kcdi_corpus)
        for _, row in group_metrics.iterrows():
            comparison_rows.append({
                'Corpus': corpus_name,
                'Group': row['Group'],
                'n_docs': row['n_docs'],
                'doc_share': row['doc_share'],
                'total_weight': row['total_weight'],
                'weight_share': row['weight_share'],
                'A_factor': row['A_factor'],
                'S_factor': row['S_factor'],
                'KJI_group': row['KJI_group']
            })

    comparison_df = pd.DataFrame(comparison_rows)

    for corpus_name, summary in corpus_summary.items():
        mask = comparison_df['Corpus'] == corpus_name
        comparison_df.loc[mask, 'KCDI_corpus'] = summary['KCDI']
        comparison_df.loc[mask, 'KJI_mean'] = summary['KJI_mean']
        comparison_df.loc[mask, 'A_mean'] = summary['A_mean']
        comparison_df.loc[mask, 'S_mean'] = summary['S_mean']
        comparison_df.loc[mask, 'HHI'] = summary['HHI']
        comparison_df.loc[mask, 'Top10_share'] = summary['Top10_share']
        comparison_df.loc[mask, 'n_docs_total'] = summary['n_docs_total']

    return comparison_df, corpus_summary

def plot_corpus_comparison(comparison_df: pd.DataFrame, metric: str = 'KJI_group'):
    if len(comparison_df) == 0:
        return None
    if metric not in comparison_df.columns:
        st.warning(f"Metric '{metric}' not found in comparison data")
        return None

    fig = px.bar(
        comparison_df,
        x='Group',
        y=metric,
        color='Corpus',
        barmode='group',
        title=f'{metric} Comparison Across Corpora',
        labels={metric: metric, 'Group': 'Gender × Region'}
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        margin=dict(b=150)
    )
    return fig

def generate_snapshot_export(lambda_param: float, column_mappings: Dict, corpus_info: Dict[str, Dict]) -> str:
    snapshot = {
        'app_version': APP_VERSION,
        'timestamp': datetime.now().isoformat(),
        'lambda': lambda_param,
        'column_mappings': column_mappings,
        'global_north_hash': compute_global_north_hash(),
        'corpora': corpus_info
    }
    return json.dumps(snapshot, indent=2)

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Hero image with robust loading
    hero_path = Path(__file__).parent / "assets" / "kratos_front.png"
    if hero_path.exists():
        try:
            st.image(str(hero_path), use_container_width=True)
        except Exception:
            st.markdown("---")
    else:
        st.markdown("---")

    st.title("⚖️ KRATOS – Knowledge Justice Analytics")
    st.markdown(f"*Computational bibliometric analysis for epistemic justice* • v{APP_VERSION}")

    # ========================================================================
    # SIDEBAR - Configuration
    # ========================================================================

    with st.sidebar:
        st.header("⚙️ Configuration")

        mode = st.radio(
            "Analysis Mode",
            ["Single Corpus", "Comparative Corpora"],
            help="Choose whether to analyze one corpus or compare multiple"
        )

        st.markdown("---")

        if mode == "Single Corpus":
            uploaded_file = st.file_uploader(
                "📁 Upload Scopus CSV",
                type=['csv'],
                help="Upload a CSV export from Scopus"
            )
            corpus_data = {}
            if uploaded_file:
                corpus_data['Main Corpus'] = uploaded_file
        else:
            st.markdown("**Upload 2-5 corpora for comparison**")
            n_corpora = st.number_input("Number of corpora", min_value=2, max_value=5, value=2)
            corpus_data = {}
            for i in range(n_corpora):
                col1, col2 = st.columns([2, 1])
                with col1:
                    uploaded = st.file_uploader(
                        f"Corpus {i+1}",
                        type=['csv'],
                        key=f'corpus_{i}'
                    )
                with col2:
                    label = st.text_input(
                        "Label",
                        value=f"Corpus {i+1}",
                        key=f'label_{i}'
                    )
                if uploaded:
                    corpus_data[label] = uploaded

        if not corpus_data:
            st.info("👆 Please upload at least one CSV file to begin")
            st.stop()

        st.markdown("---")

        st.subheader("📊 Column Mapping")

        first_file = list(corpus_data.values())[0]
        file_bytes = first_file.read()
        first_file.seek(0)

        separator = detect_separator(file_bytes)
        from io import BytesIO
        sample_df = pd.read_csv(BytesIO(file_bytes), sep=separator, nrows=0)

        # If header itself contains duplicates, make unique so selection is stable.
        if sample_df.columns.duplicated().any():
            sample_df = _make_unique_columns(sample_df)

        available_cols = list(sample_df.columns)

        default_author = find_best_column(sample_df, DEFAULT_COLUMNS['author'])
        default_country = find_best_column(sample_df, DEFAULT_COLUMNS['country'])
        default_affiliations = find_best_column(sample_df, DEFAULT_COLUMNS['affiliations'])
        default_year = find_best_column(sample_df, DEFAULT_COLUMNS['year'])
        default_weight = find_best_column(sample_df, DEFAULT_COLUMNS['weight'])
        default_source = find_best_column(sample_df, DEFAULT_COLUMNS['source'])

        author_col = st.selectbox(
            "Author names *",
            options=available_cols,
            index=available_cols.index(default_author) if default_author in available_cols else 0,
            help="Column containing author full names"
        )

        year_col = st.selectbox(
            "Year *",
            options=available_cols,
            index=available_cols.index(default_year) if default_year in available_cols else 0,
            help="Publication year"
        )

        weight_col = st.selectbox(
            "Weight/Citations *",
            options=available_cols,
            index=available_cols.index(default_weight) if default_weight in available_cols else 0,
            help="Citation count or impact metric"
        )

        country_col = st.selectbox(
            "Country (optional)",
            options=['<None>'] + available_cols,
            index=(available_cols.index(default_country) + 1) if (default_country in available_cols) else 0,
            help="Country column (if not available, will try to extract from affiliations)"
        )
        country_col = None if country_col == '<None>' else country_col

        affiliations_col = st.selectbox(
            "Affiliations (optional)",
            options=['<None>'] + available_cols,
            index=(available_cols.index(default_affiliations) + 1) if (default_affiliations in available_cols) else 0,
            help="Affiliation column (used to infer country if Country not available)"
        )
        affiliations_col = None if affiliations_col == '<None>' else affiliations_col

        source_col = st.selectbox(
            "Source title (optional)",
            options=['<None>'] + available_cols,
            index=(available_cols.index(default_source) + 1) if (default_source in available_cols) else 0,
            help="Journal or source title"
        )
        source_col = None if source_col == '<None>' else source_col

        # Prevent user-induced duplicates in selections
        validate_column_mapping_or_stop({
            'author': author_col,
            'year': year_col,
            'weight': weight_col,
            'country': country_col,
            'affiliations': affiliations_col,
            'source': source_col
        })

        st.markdown("---")

        st.subheader("🎛️ Parameters")

        lambda_param = st.slider(
            "λ (diversity-recognition balance)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="λ=0: pure recognition, λ=1: pure diversity (MULTIPLICATIVE formulation)"
        )

        display_limit = st.slider(
            "Display rows limit",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Limit number of rows shown in tables"
        )

        st.markdown("---")
        st.caption(f"KRATOS v{APP_VERSION}")

    # ========================================================================
    # MAIN CONTENT - Data Loading
    # ========================================================================

    enriched_data: Dict[str, pd.DataFrame] = {}
    corpus_info: Dict[str, Dict] = {}

    with st.spinner("🔄 Loading and enriching data..."):
        for corpus_name, uploaded_file in corpus_data.items():
            fb = uploaded_file.read()
            uploaded_file.seek(0)
            file_hash = compute_file_hash(fb)

            try:
                df_enriched = load_and_enrich_data(
                    fb,
                    uploaded_file.name,
                    author_col,
                    country_col,
                    affiliations_col,
                    year_col,
                    weight_col,
                    source_col
                )

                # Defensive: ensure uniqueness even if upstream changes occur
                if df_enriched.columns.duplicated().any():
                    df_enriched = _make_unique_columns(df_enriched)

                enriched_data[corpus_name] = df_enriched
                corpus_info[corpus_name] = {
                    'file_hash': file_hash,
                    'n_rows': len(df_enriched),
                    'file_name': uploaded_file.name
                }

            except ValueError as e:
                st.error(f"❌ Error loading {corpus_name}: {str(e)}")
                st.stop()

    st.success(f"✅ Loaded and enriched {len(enriched_data)} corpus/corpora")

    column_mappings = {
        'author': author_col,
        'year': year_col,
        'weight': weight_col,
        'country': country_col,
        'affiliations': affiliations_col,
        'source': source_col
    }
    snapshot_json = generate_snapshot_export(lambda_param, column_mappings, corpus_info)

    st.download_button(
        label="📸 Download Analysis Snapshot (JSON)",
        data=snapshot_json,
        file_name=f"kratos_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    # ========================================================================
    # TABS - Analysis and Visualization
    # ========================================================================

    if mode == "Single Corpus":
        df = enriched_data['Main Corpus']

        tabs = st.tabs([
            "📊 Overview",
            "🎯 KCDI Analysis",
            "⚖️ Group Justice (KJI)",
            "📈 Temporal Trends",
            "🔍 Explorer",
            "📉 PPI Diagnostic",
            "📖 Methodology"
        ])

        # --- TAB 1: Overview ---
        with tabs[0]:
            st.header("Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", f"{len(df):,}")
            with col2:
                st.metric("Total Citations", f"{df['_weight_numeric'].sum():,.0f}")
            with col3:
                valid_years = df[df['_year_numeric'].notna()]['_year_numeric']
                st.metric("Year Range", f"{int(valid_years.min())} - {int(valid_years.max())}" if len(valid_years) > 0 else "N/A")
            with col4:
                st.metric("Mean Citations/Doc", f"{df['_weight_numeric'].mean():.1f}")

            st.subheader("Gender Inference Report")
            gender_report = generate_gender_report(df)
            st.dataframe(gender_report, use_container_width=True, hide_index=True)

            st.subheader("Region Classification Report")
            region_report = generate_region_report(df)
            st.dataframe(region_report, use_container_width=True, hide_index=True)

            st.subheader("Sample Data (enriched)")
            display_cols = [author_col, year_col, weight_col, 'Gender', 'Region', 'Gender_Region']
            safe_dataframe_display(df, display_cols, n=display_limit)

        # --- TAB 2: KCDI ---
        with tabs[1]:
            st.header("KCDI Analysis (Corpus Level)")

            st.markdown(f"""
            **KCDI** (Knowledge Contribution Diversity Index) at corpus level:

            $$
            KCDI = (H')^\\lambda \\times (W_{{norm}})^{{1-\\lambda}}
            $$

            Where:
            - **H'**: Shannon entropy (diversity of document distribution across groups)
            - **W_norm**: Normalized recognition intensity (mean group weights)
            - **λ = {lambda_param}**: Balance parameter (MULTIPLICATIVE formulation)
            """)

            kcdi, kcdi_details = compute_kcdi_corpus(df, 'Gender_Region', '_weight_numeric', lambda_param)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("KCDI (corpus)", f"{kcdi:.4f}")
            with col2:
                st.metric("H' (entropy)", f"{kcdi_details['H_prime']:.4f}")
            with col3:
                st.metric("W_norm", f"{kcdi_details['W_norm']:.4f}")
            with col4:
                st.metric("Groups", kcdi_details['n_groups'])

            st.info(
                "KCDI is computed at the corpus level over Gender×Region groups, NOT within individual groups."
            )

            with st.expander("📚 Optional: Within-Group Source Diversity", expanded=False):
                st.markdown(
                    "Source-level diversity within each Gender×Region group (distinct from corpus-level KCDI)."
                )

                source_diversity = []
                for grp in df['Gender_Region'].unique():
                    subset = df[df['Gender_Region'] == grp]
                    if len(subset) > 0:
                        source_counts = subset['_source_title'].value_counts()
                        H_source = compute_shannon_entropy_normalized(source_counts)
                        source_diversity.append({
                            'Group': grp,
                            'n_docs': int(len(subset)),
                            'n_sources': int(len(source_counts)),
                            'Source_H_prime': float(H_source)
                        })

                source_div_df = pd.DataFrame(source_diversity)
                st.dataframe(source_div_df, use_container_width=True, hide_index=True)

        # --- TAB 3: Group Justice (KJI) ---
        with tabs[2]:
            st.header("Group Justice Analysis (KJI)")

            st.markdown("""
            **KJI** (Knowledge Justice Index) measures fairness for each group:

            $$
            KJI(u) = KCDI_{corpus} \\times A(u) \\times S(u)
            $$

            **Parity-Ideal Formulation:**
            - **A(u)**: $A(u) = \\max(0, 1 - |p_u - p^*| / p^*)$ where $p^* = 1/G$
            - **S(u)**: $S(u) = \\max(0, 1 - |(s_u/p_u) - 1|)$
            """)

            kcdi_corpus, _ = compute_kcdi_corpus(df, 'Gender_Region', '_weight_numeric', lambda_param)
            kji_mean, kji_details = compute_corpus_kji(df, 'Gender_Region', '_weight_numeric', lambda_param)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("KJI (mean)", f"{kji_mean:.4f}")
            with col2:
                st.metric("Ā (participation)", f"{kji_details['A_mean']:.4f}")
            with col3:
                st.metric("S̄ (recognition)", f"{kji_details['S_mean']:.4f}")
            with col4:
                st.metric("KCDI (corpus)", f"{kcdi_corpus:.4f}")

            st.subheader("Group Justice Table")
            group_metrics = compute_group_justice_metrics(df, 'Gender_Region', '_weight_numeric', kcdi_corpus)
            plot_group_justice_table(group_metrics)

            csv = group_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Group Metrics CSV",
                data=csv,
                file_name="kratos_group_justice.csv",
                mime="text/csv"
            )

            st.subheader("KJI by Group")
            fig_kji = plot_kji_bars(group_metrics)
            if fig_kji is not None:
                st.plotly_chart(fig_kji, use_container_width=True)

            st.markdown("""
            **Interpretation Guide:**
            - **A(u) close to 1**: document share near parity ideal (1/G)
            - **S(u) close to 1**: citation share aligns with document share
            - **KJI(u) high**: higher epistemic justice for that group
            - **Signed gap > 0**: over-recognized (citations > docs)
            - **Signed gap < 0**: under-recognized (citations < docs)
            """)

        # --- TAB 4: Temporal ---
        with tabs[3]:
            st.header("Temporal Trends")

            fig_temporal = plot_temporal_trends(df)
            if fig_temporal is not None:
                st.plotly_chart(fig_temporal, use_container_width=True)

            st.subheader("KJI Evolution Over Time")
            valid_years = df[df['_year_numeric'].notna()]['_year_numeric'].unique()
            valid_years = sorted(valid_years)

            if len(valid_years) > 1:
                kji_temporal = []
                for y in valid_years:
                    year_df = df[df['_year_numeric'] == y]
                    if len(year_df) >= 5:
                        year_kji, _ = compute_corpus_kji(year_df, 'Gender_Region', '_weight_numeric', lambda_param)
                        kji_temporal.append({
                            'Year': int(y),
                            'KJI': float(year_kji),
                            'n_docs': int(len(year_df))
                        })

                if kji_temporal:
                    kji_temp_df = pd.DataFrame(kji_temporal)
                    fig_kji_time = px.line(
                        kji_temp_df,
                        x='Year',
                        y='KJI',
                        markers=True,
                        title='KJI Evolution (Mean Across Groups)',
                        labels={'KJI': 'KJI Mean', 'Year': 'Publication Year'}
                    )
                    st.plotly_chart(fig_kji_time, use_container_width=True)
                    st.dataframe(kji_temp_df, use_container_width=True, hide_index=True)

        # --- TAB 5: Explorer ---
        with tabs[4]:
            st.header("Data Explorer")
            st.subheader("Filter and Explore")

            col1, col2 = st.columns(2)
            with col1:
                selected_gender = st.multiselect(
                    "Filter by Gender",
                    options=list(df['Gender'].unique()),
                    default=list(df['Gender'].unique())
                )
            with col2:
                selected_region = st.multiselect(
                    "Filter by Region",
                    options=list(df['Region'].unique()),
                    default=list(df['Region'].unique())
                )

            filtered_df = df[(df['Gender'].isin(selected_gender)) & (df['Region'].isin(selected_region))]
            st.metric("Filtered Documents", f"{len(filtered_df):,}")

            st.subheader("Top Cited Documents")
            top_cited = filtered_df.nlargest(20, '_weight_numeric')

            display_cols_explorer = [author_col, year_col, '_source_title', '_weight_numeric', 'Gender', 'Region']
            display_cols_explorer = _dedupe_list_preserve_order(display_cols_explorer)
            display_cols_explorer = [c for c in display_cols_explorer if c in top_cited.columns]

            display_df = top_cited.loc[:, display_cols_explorer].copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str)

            st.dataframe(display_df.head(display_limit), use_container_width=True, hide_index=True)

            st.subheader("Top Sources")
            if len(filtered_df) > 0:
                source_counts = filtered_df['_source_title'].value_counts().head(20)
                fig_sources = px.bar(
                    x=source_counts.index.astype(str),
                    y=source_counts.values,
                    labels={'x': 'Source', 'y': 'Number of Documents'},
                    title='Top 20 Sources'
                )
                fig_sources.update_layout(xaxis_tickangle=-45, margin=dict(b=150))
                st.plotly_chart(fig_sources, use_container_width=True)

        # --- TAB 6: PPI ---
        with tabs[5]:
            st.header("PPI Diagnostic (Pleonasm/Concentration)")

            # ensure kji_details exists even if user jumps tabs early
            _, kji_details_local = compute_corpus_kji(df, 'Gender_Region', '_weight_numeric', lambda_param)
            ppi = compute_ppi(df, '_weight_numeric')
            S_mean = kji_details_local['S_mean']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("HHI", f"{ppi['HHI']:.4f}")
            with col2:
                st.metric("Top 10% Share", f"{ppi['Top10_share']:.1%}")
            with col3:
                st.metric("S̄ (recognition)", f"{S_mean:.4f}")

            st.markdown("""
            **PPI Indicators** (separate, not summed):
            - **HHI**: concentration of citations (0→low concentration; 1→monopoly)
            - **Top 10% Share**: fraction of citations held by the top 10% of documents

            **Expected relationship:** higher concentration tends to depress recognition fairness (S̄).
            """)

            fig_ppi = plot_ppi_diagnostic(df, S_mean)
            st.plotly_chart(fig_ppi, use_container_width=True)

        # --- TAB 7: Methodology ---
        with tabs[6]:
            st.header("Methodological Notes")
            st.markdown(f"""
            ## KRATOS Metrics: Mathematical Definitions

            ### 1. KCDI (Knowledge Contribution Diversity Index)
            $$
            KCDI = (H')^\\lambda \\times (W_{{norm}})^{{1-\\lambda}}
            $$

            **H' (Normalized Shannon Entropy):**
            $$
            H' = \\frac{{-\sum_{{i=1}}^{{k}} p_i \\log p_i}}{{\\log k}}
            $$

            **W_norm (Normalized Weight Intensity):**
            $$
            W_{{norm}} = \\frac{{\\bar{{W}} - W_{{min}}}}{{W_{{max}} - W_{{min}}}}
            $$

            ### 2. KJI/KRATOS (Knowledge Justice Index)
            $$
            KJI(u) = KCDI_{{corpus}} \\times A(u) \\times S(u)
            $$

            **A(u) (Participation Fairness):**
            $$
            A(u) = \\max\\left(0, 1 - \\frac{{|p_u - p^*|}}{{p^*}}\\right), \\quad p^* = 1/G
            $$

            **S(u) (Recognition Fairness):**
            $$
            S(u) = \\max\\left(0, 1 - \\left|\\frac{{s_u}}{{p_u}} - 1\\right|\\right)
            $$

            ### 3. PPI (Pleonasm/Concentration Indicators)
            **HHI:**
            $$
            HHI = \\sum_{{i=1}}^n s_i^2
            $$

            **Top 10% Share:**
            $$
            T_{{10}} = \\frac{{\\sum_{{i \\in Top10\\%}} c_i}}{{C}}
            $$

            **GLOBAL_NORTH hash:** `{compute_global_north_hash()}`  
            **Version:** {APP_VERSION}
            """)

    else:
        tabs = st.tabs([
            "📊 Overview",
            "🔄 Comparison",
            "📈 Visualizations",
            "📉 Concentration Analysis",
            "📖 Methodology"
        ])

        comparison_df, corpus_summary = compare_corpora(enriched_data, lambda_param)

        # --- TAB 1: Overview ---
        with tabs[0]:
            st.header("Comparative Corpora Overview")

            for corpus_name, dfc in enriched_data.items():
                with st.expander(f"📁 {corpus_name}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Documents", f"{len(dfc):,}")
                    with col2:
                        st.metric("Citations", f"{dfc['_weight_numeric'].sum():,.0f}")
                    with col3:
                        st.metric("Avg Citations", f"{dfc['_weight_numeric'].mean():.1f}")
                    with col4:
                        st.metric("KJI", f"{corpus_summary[corpus_name]['KJI_mean']:.4f}")

                    gender_dist = dfc['Gender'].value_counts()
                    fig_gender = px.pie(
                        values=gender_dist.values,
                        names=gender_dist.index,
                        title=f"Gender Distribution - {corpus_name}"
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)

        # --- TAB 2: Comparison ---
        with tabs[1]:
            st.header("Metric Comparison Across Corpora")

            st.subheader("Full Comparison Table")
            display_comparison = comparison_df.copy()
            numeric_cols = ['doc_share', 'weight_share', 'A_factor', 'S_factor', 'KJI_group',
                            'KCDI_corpus', 'KJI_mean', 'A_mean', 'S_mean', 'HHI', 'Top10_share']
            for col in numeric_cols:
                if col in display_comparison.columns:
                    display_comparison[col] = display_comparison[col].round(4)
            st.dataframe(display_comparison, use_container_width=True, hide_index=True)

            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison CSV",
                data=csv,
                file_name="kratos_comparison.csv",
                mime="text/csv"
            )

            st.subheader("Summary by Corpus")
            summary_data = []
            for corpus_name, summary in corpus_summary.items():
                summary_data.append({
                    'Corpus': corpus_name,
                    'n_docs': summary['n_docs_total'],
                    'KCDI': summary['KCDI'],
                    'KJI_mean': summary['KJI_mean'],
                    'A_mean': summary['A_mean'],
                    'S_mean': summary['S_mean'],
                    'HHI': summary['HHI'],
                    'Top10_share': summary['Top10_share']
                })
            summary_df = pd.DataFrame(summary_data)
            for col in ['KCDI', 'KJI_mean', 'A_mean', 'S_mean', 'HHI', 'Top10_share']:
                summary_df[col] = summary_df[col].round(4)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.subheader("Delta Analysis")
            baseline_corpus = st.selectbox("Select baseline corpus", options=list(enriched_data.keys()))
            baseline_data = comparison_df[comparison_df['Corpus'] == baseline_corpus]

            delta_rows = []
            for corpus_name in enriched_data.keys():
                if corpus_name == baseline_corpus:
                    continue
                corpus_part = comparison_df[comparison_df['Corpus'] == corpus_name]
                merged = corpus_part.merge(
                    baseline_data[['Group', 'KJI_group']],
                    on='Group',
                    suffixes=('', '_baseline'),
                    how='outer'
                )
                merged['KJI_delta'] = merged['KJI_group'] - merged['KJI_group_baseline']
                merged['Corpus'] = corpus_name
                delta_rows.append(merged[['Corpus', 'Group', 'KJI_group', 'KJI_group_baseline', 'KJI_delta']])

            if delta_rows:
                delta_df = pd.concat(delta_rows, ignore_index=True)
                for col in ['KJI_group', 'KJI_group_baseline', 'KJI_delta']:
                    if col in delta_df.columns:
                        delta_df[col] = delta_df[col].round(4)
                st.dataframe(delta_df, use_container_width=True, hide_index=True)

        # --- TAB 3: Visualizations ---
        with tabs[2]:
            st.header("Comparative Visualizations")

            st.subheader("KJI Comparison by Group")
            fig_kji_comp = plot_corpus_comparison(comparison_df, 'KJI_group')
            if fig_kji_comp is not None:
                st.plotly_chart(fig_kji_comp, use_container_width=True)

            st.subheader("Participation Fairness (A) Comparison")
            fig_a_comp = plot_corpus_comparison(comparison_df, 'A_factor')
            if fig_a_comp is not None:
                st.plotly_chart(fig_a_comp, use_container_width=True)

            st.subheader("Recognition Fairness (S) Comparison")
            fig_s_comp = plot_corpus_comparison(comparison_df, 'S_factor')
            if fig_s_comp is not None:
                st.plotly_chart(fig_s_comp, use_container_width=True)

            st.subheader("KCDI Comparison (Corpus Level)")
            kcdi_data = pd.DataFrame([
                {'Corpus': name, 'KCDI': summary['KCDI']}
                for name, summary in corpus_summary.items()
            ])
            fig_kcdi = px.bar(
                kcdi_data,
                x='Corpus',
                y='KCDI',
                title='KCDI Across Corpora',
                labels={'KCDI': 'KCDI (Corpus Level)'}
            )
            st.plotly_chart(fig_kcdi, use_container_width=True)

        # --- TAB 4: Concentration Analysis ---
        with tabs[3]:
            st.header("Concentration Analysis (PPI)")

            hhi_data = pd.DataFrame([
                {'Corpus': name, 'HHI': summary['HHI'], 'Top10_share': summary['Top10_share'],
                 'S_mean': summary['S_mean'], 'KJI_mean': summary['KJI_mean'], 'n_docs_total': summary['n_docs_total']}
                for name, summary in corpus_summary.items()
            ])

            st.subheader("HHI Comparison")
            fig_hhi = px.bar(
                hhi_data,
                x='Corpus',
                y='HHI',
                title='HHI Across Corpora',
                labels={'HHI': 'Herfindahl-Hirschman Index'}
            )
            st.plotly_chart(fig_hhi, use_container_width=True)

            st.subheader("Top 10% Share Comparison")
            fig_top10 = px.bar(
                hhi_data,
                x='Corpus',
                y='Top10_share',
                title='Top 10% Citation Share Across Corpora',
                labels={'Top10_share': 'Top 10% Share'}
            )
            st.plotly_chart(fig_top10, use_container_width=True)

            if len(corpus_summary) >= 3:
                st.subheader("Concentration vs Recognition Fairness")

                fig_scatter = plot_concentration_scatter(
                    hhi_data[['Corpus', 'HHI', 'S_mean', 'KJI_mean', 'n_docs_total']].copy()
                )
                if fig_scatter is not None:
                    st.plotly_chart(fig_scatter, use_container_width=True)

                HHI_values = hhi_data['HHI'].values
                S_values = hhi_data['S_mean'].values
                if len(HHI_values) >= 3 and len(S_values) >= 3:
                    corr, p_value = spearmanr(HHI_values, S_values)
                    st.metric("Spearman Correlation (HHI vs S̄)", f"{corr:.3f}")
                    st.caption(f"p-value: {p_value:.4f}")

        # --- TAB 5: Methodology ---
        with tabs[4]:
            st.header("Methodological Notes - Comparative Mode")
            st.markdown("""
            In comparative mode, KRATOS computes metrics independently for each corpus using the same:
            - column mappings
            - λ parameter
            - grouping definition (Gender × Region)

            With ≥3 corpora, Spearman rank correlation between concentration (HHI) and recognition fairness (S̄)
            is reported to test the expected negative association (higher concentration → lower fairness).
            """)

    st.markdown("---")
    st.caption(
        f"KRATOS – Knowledge Justice Analytics v{APP_VERSION} • Computational bibliometric analysis for epistemic justice research"
    )

if __name__ == "__main__":
    main()
