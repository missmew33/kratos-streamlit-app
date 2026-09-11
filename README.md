# KRATOS v2.3 — Bibliometric Recognition Diagnostic

KRATOS is an auditable Python/Streamlit research instrument for examining whether bibliometric recognition signals are distributed comparably across predefined knowledge-producing positions.

The production application is `app.py`. The computational definitions live in `kratos_core.py`. Robustness and validation diagnostics live in `kratos_diagnostics.py` and are exposed interactively through `pages/2_Robustness_diagnostics.py`.

## Current measurement regime

The substantive analytical universe is fixed at **G=4**:

- female × Global North
- female × Global South
- male × Global North
- male × Global South

Gender is a **metadata-derived proxy**, not self-identified gender. Unresolved gender or geography remains an `unknown` audit state and is excluded from the primary G=4 parity universe. Demographic coverage is reported explicitly.

One document is attributed to the **first author**. For raw Scopus data, geography is resolved from the first listed affiliation of that first author using conservative exact-country matching.

## KRATOS metrics

For document shares `p_u` and citation shares `s_u` over the fixed four-cell universe:

```text
H_D_prime = normalised Shannon entropy of document shares
H_C_prime = normalised Shannon entropy of citation shares
KCDI      = H_D_prime^lambda * H_C_prime^(1-lambda)
A(u)      = max(0, 1 - |p_u - 1/4| / (1/4))
S(u)      = max(0, 1 - |s_u/p_u - 1|)
P         = mean[A(u) * S(u)] over G=4
R         = 1 - P
KJI       = KCDI * P
```

The primary specification uses `lambda = 0.5`.

`KJI <= KCDI` and `KCDI - KJI = KCDI(1-P)` are architectural identities. They are **not** empirical tests of canonical closure, epistemic stigma, or epistemic injustice.

### Why `W_norm` was removed

The former range-normalised component

```text
(mean - min) / (max - min)
```

did not behave as a valid measure of citation evenness: near-equal citation totals could receive low values, more unequal configurations could receive higher values, and the equality convention introduced a discontinuity. KRATOS v2.2+ therefore uses `H_C_prime`, normalised Shannon entropy of citation shares over the same fixed G=4 universe as `H_D_prime`.

## Robustness diagnostics added in v2.3

### Resolved-set permutation null calibration

The null calibration freezes the exact resolved analytical set `D_R` and preserves:

- `N_R`;
- the four G=4 group labels and group sizes;
- the empirical citation-count distribution among resolved records.

Citation counts are then permuted across those same resolved records. This removes only the association between analytical group and citation outcome. The procedure is equivalent to permuting group labels while preserving group sizes, but makes the frozen resolved set explicit.

For `H_C_prime`, `KCDI`, `P`, and `KJI`, KRATOS reports:

- observed value;
- null median;
- empirical 2.5th and 97.5th percentiles;
- observed-minus-null-median;
- Monte Carlo corrected lower- and upper-tail probabilities;
- whether the observed value lies below, within, or above the 95% null range.

These quantities are **conditional diagnostic baselines**, not population-level confidence intervals and not evidence of causal demographic effects.

### Composition diagnostics

Two supplementary quantities are reported to make the architecture of `P` more transparent:

```text
A_bar = mean[A(u)]
P_S   = sum[p_u * S(u)]
```

`A_bar` is the maximum attainable value of `P` if `S(u)=1` for all four groups under the fixed `p*=1/4` reference. `P_S` summarises proportional recognition without using the fixed parity reference. It remains composition-weighted and is therefore a supplementary diagnostic, not a replacement for `P` or `KJI`.

## Citation concentration

Document-level citation concentration is reported separately using:

- Gini coefficient
- Herfindahl–Hirschman Index (HHI)
- Top 10% citation share

These are full-corpus descriptive diagnostics and are not components of KCDI, P, or KJI.

## Inputs

The production Streamlit application accepts CSV files with a citation-count column plus one of the following:

1. raw Scopus `Authors with affiliations` data;
2. an explicit first-author country field; or
3. precomputed KRATOS demographic audit fields (`group`, `gender_category`, `region`).

For manuscript-grade permutation calibration, use the **exact frozen record-level audit CSV** used to generate the primary KRATOS estimates. Licensed Scopus source records are not committed to this public repository.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The main app now exposes a separate **Robustness diagnostics** page automatically through Streamlit's `pages/` directory.

The compatibility entrypoint also remains valid:

```bash
streamlit run app_v2_2.py
```

## Reproducibility utilities

- `scripts/kratos_analyze_csv.py`: record-level demographic audit, G=4 group metrics, corpus snapshot, and full-corpus citation concentration.
- `scripts/kratos_g4_sensitivity.py`: unresolved-gender sensitivity and matched-size resampling.
- `scripts/kratos_null_baseline.py`: exact resolved-set permutation calibration plus composition diagnostics.
- `kratos_diagnostics.py`: reusable implementation of null calibration, Monte Carlo summaries, and composition diagnostics.
- `pages/2_Robustness_diagnostics.py`: interactive multi-corpus robustness page with downloadable null summaries, permutation draws, and reproducibility snapshot.
- `tests/`: regression tests for geography resolution, fixed G=4 treatment, citation entropy, KCDI boundary behaviour, KJI architecture, resolved-set permutation calibration, and production-app startup.

The default stochastic seed is `20260831`; manuscript analyses should record the exact seed, draw count, input hashes, and software revision used for the frozen results. The permutation calibration defaults to `B=5000` draws.

## Interpretation

KRATOS is a **recognition-comparability diagnostic**. Corpus-level values should not be read as direct rankings of epistemic justice. Cross-corpus interpretation should consider the accompanying corpus-conditional null calibration, common-window, matched-size, parameter, and unresolved-metadata sensitivity analyses.
