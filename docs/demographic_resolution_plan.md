# KRATOS demographic resolution v2

## Purpose

This branch replaces the legacy `gender-guesser` enrichment path with an auditable first-author demographic-resolution layer and aligns the executable code with the revised Scientometrics measurement regime.

## Measurement rules

1. **Unit of attribution:** one document is assigned to one analytical cell using the first author.
2. **Geography:** the country is taken from the **first listed affiliation of the first author** in Scopus `Authors with affiliations`. A country attached to a later coauthor must never determine the record's region.
3. **Region:** country is standardised to ISO3 and mapped to the canonical KRATOS Global North set. The executable set contains **39 ISO3 codes** and reproduces taxonomy hash `39dcca84388a6762`.
4. **Exact geography rule:** only exact country-name tokens are accepted. Fuzzy `country-converter` regex matches are not used on raw affiliation strings because place names such as `Bengbu`, `Ningbo`, or `Central Macedonia` can otherwise generate false country assignments.
5. **Name-based gender category:** the open-source `genderComputer` package is used through a KRATOS adapter and receives name plus first-author country where available.
6. **Conservative gender rule:** only exact `female` or `male` outputs are accepted. `unisex`, unresolved values, missing names, and backend errors become `unknown`; classifications are never probabilistically forced in the primary analysis.
7. **Interpretation:** `unknown` is a metadata-resolution state, not a substantive demographic category and not self-identified gender.
8. **Primary analytical universe:** fixed **G=4** = female/male × Global North/Global South. Empty substantive cells remain.
9. **Parity reference:** `p*=1/4`.
10. **Coverage rule:** records with unresolved gender or geography remain in the audit trail but do not enter the G=4 parity calculation. Demographic coverage is reported for every corpus.
11. **Distributional components:** `H_D_prime` is normalised Shannon entropy of resolved document shares; `H_C_prime` is normalised Shannon entropy of resolved citation shares over the same fixed G=4 universe.
12. **KCDI:** `KCDI = H_D_prime^lambda × H_C_prime^(1-lambda)`.
13. **KJI architecture:** `KJI=KCDI×P`, with `P=mean[A(u)S(u)]` over the four substantive cells, so `KJI<=KCDI` is architectural.
14. **Sensitivity:** unresolved gender is evaluated separately through explicit sensitivity analyses; it is not granted its own parity target.

## Mathematical audit revision

The former `W_norm=(mean-min)/(max-min)` component was removed after mathematical audit. It did not behave as a valid citation-evenness measure: near-equal citation totals could receive low scores, more unequal configurations could receive higher scores, and the equality convention introduced a discontinuity. `H_C_prime` replaces it with a bounded, continuous, scale-invariant measure using the same fixed-G entropy logic as `H_D_prime`.

## Geography validation

The exact-country rule was regression-checked against the manually audited canonical Trade Fairs/MICE corpus (`N=101`), reproducing its validated geography distribution: Global North=62, Global South=38, unknown=1. Additional regression cases ensure that affiliation/location fragments are not converted into countries.

## Primary analysis and unresolved-metadata sensitivity

The primary KRATOS comparison is complete-case with respect to the four substantive demographic cells. Every corpus must therefore report both `n_docs_input` and `demographic_coverage` alongside `H_D_prime`, `H_C_prime`, KCDI, P, R, and KJI.

Unresolved gender is assessed through **stratified stochastic sensitivity analysis**. For records with known Global North/Global South geography but unresolved gender, female/male assignments are drawn according to the observed resolved-gender distribution within the same region and corpus. Geography-unknown records remain unresolved. The default sensitivity analysis uses `B=1000` draws and reports the median and empirical 2.5th/97.5th percentiles.

This procedure is not interpreted as recovering an author's true or self-identified gender. It is a measurement-sensitivity scenario used to test whether substantive cross-corpus comparisons depend on unresolved metadata.

**Interpretation rule:** cross-corpus ordering is not treated as robust when it changes materially between the complete-case analysis and reasonable unresolved-gender sensitivity scenarios. In that case, the result is reported as measurement-sensitive rather than converted into a categorical ranking or epistemic-regime classification.

Matched-size robustness samples documents before demographic filtering and then recomputes G=4 KRATOS, so demographic coverage remains part of each resampled diagnostic. The harmonised common-window anchor is `n=92` for 2010--2025; the full-window anchor is `n=101` for 2006--2025. The default number of draws is `B=1000`.

## Implementation

`kratos_core.py` provides first-author parsing; exact first-listed affiliation country resolution; ISO3/region classification; replaceable `GenderResolver`; conservative `GenderComputerResolver`; record-level audit metadata; substantive fixed-G=4 KCDI/KJI; demographic coverage; and document-level concentration metrics.

`app.py` is the validated production Streamlit interface and uses the same `kratos_core.py` definitions. It no longer displays or computes the deprecated `W_norm` formulation. `app_v2_2.py` is retained only as a compatibility entrypoint and delegates to `app.py`.

`scripts/kratos_g4_sensitivity.py` implements the unresolved-gender and matched-size sensitivity procedures without requiring licensed source records to be stored in the public repository.

Citation concentration remains a full-corpus document-level diagnostic because it does not require demographic attribution.

## Third-party dependency

`genderComputer` is pinned to commit `f6267615517913e53cb0b882b248f1c2e11b8bbc`; upstream Python code is LGPLv3. KRATOS does not copy its underlying name datasets.

## Validation status

- [x] pure computational core
- [x] replace `gender-guesser`
- [x] exact-country geography safeguard
- [x] substantive G=4 measurement regime
- [x] unknown separated as audit/coverage state
- [x] replace `W_norm` with `H_C_prime` after mathematical audit
- [x] regression tests for entropy boundaries, scale invariance, and KJI architecture
- [x] canonical MICE geography validation
- [x] four-corpus G=4 execution and coverage audit
- [x] common-window and matched-size sensitivity outputs
- [x] unresolved-gender sensitivity procedure and interpretation rule
- [x] production `app.py` aligned with `H_D_prime`/`H_C_prime`
- [x] Streamlit startup smoke test in CI
- [ ] freeze manuscript snapshot/version/hash after the final manuscript equations and result tables are locked
