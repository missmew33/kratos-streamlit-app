# KCDI Streamlit App — Knowledge Justice Analytics for Scopus Data

An interactive web application that turns Scopus-exported metadata into **epistemic diversity and justice diagnostics** using the **Knowledge Contribution Diversity Index (KCDI)**.  
Built in **Python** and delivered as a **Streamlit** web app so the full analytical pipeline can be executed, inspected, and replicated without running scripts manually.

---

## What this app is (as a research instrument)

This app is not only a calculator. It is a **computational research instrument** that operationalises KCDI as a structured pipeline:

1. **Data input** (Scopus export upload)
2. **Column selection and parameterisation** (explicit analytical choices)
3. **Data enrichment** (gender-coded proxy + country/region classification)
4. **Metric computation** (KCDI and supporting components)
5. **Diagnostic visual analytics** (tables + plots for interpretability)
6. **Export for audit and reproducibility**

The interface is designed so that key assumptions (e.g., grouping variables, weights, proxies, and parameters) are **visible**, which is essential for interpretability and for comparing results across different corpora.

---

## What the app does

### 1) Data input
- Upload a Scopus-style dataset in `.csv` or `.xlsx`
- Preview the dataset immediately to verify correct parsing (first rows)

### 2) Columns and parameters
- Select or confirm which columns correspond to:
  - authors
  - country (or affiliation-derived country)
  - year (optional but recommended)
  - citations / impact proxy (optional; used for weighting if available)
- Set key parameters (when enabled in the current version), such as the blending parameter used in the KCDI formulation.

### 3) Enriched dataset (auditable)
The app generates an enriched analytical dataset including:
- author name extraction (for proxy coding)
- **gender-coded labels** (proxy inference when not provided by the source)
- country normalisation and mapping
- **region classification** (e.g., Global North / Global South proxy)

A preview of the enriched table is shown to support auditing before interpreting results.

### 4) KCDI metrics
The app computes:
- **KCDI** by selected grouping variables (e.g., gender, region, or intersection)
- supporting components required for interpretability (e.g., diversity and weighting terms, group sizes)

### 5) Visual diagnostics
Interactive plots help interpret structural patterns and compare groups, including:
- group-level KCDI comparisons
- intersectional comparisons (e.g., gender × region)
- (when available) time-based summaries

### 6) Export
- Download the enriched dataset and/or summary outputs for independent verification or extended analysis.

---

## Input requirements

Minimum required fields:
- `Authors` (or equivalent author-name field)
- `Country` (or a field from which country can be derived)

Recommended fields:
- `Year` (enables temporal diagnostics)
- `Cited by` (enables citation-weighted variants where applicable)

---

## How to use

1. Upload a Scopus-exported `.csv` or `.xlsx` file.
2. Confirm/select the correct columns under **Columns and parameters**.
3. Inspect the **Raw data** preview (parsing check).
4. Inspect the **Enriched data** preview (enrichment check).
5. Read KCDI outputs and plots; export results if needed.

---

## Deployment

The app runs on **Streamlit Community Cloud** and can also be executed locally.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py

