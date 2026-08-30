# KRATOS demographic resolution v2

## Purpose

This branch replaces the legacy `gender-guesser` enrichment path with an auditable first-author demographic-resolution layer and aligns the executable code with the revised Scientometrics measurement regime.

## Measurement rules

1. **Unit of attribution:** one document is assigned to one analytical cell using the first author.
2. **Geography:** the country is taken from the **first listed affiliation of the first author** in Scopus `Authors with affiliations`. A country attached to a later coauthor must never determine the record's region.
3. **Region:** country is standardised to ISO3 and mapped to the canonical KRATOS Global North set. The executable set contains **39 ISO3 codes** and reproduces taxonomy hash `39dcca84388a6762`.
4. **Exact geography rule:** only exact country-name tokens are accepted. Fuzzy `country-converter` regex matches are not used because place names such as `Bengbu`, `Ningbo`, or `Central Macedonia` can otherwise generate false country assignments.
5. **Name-based gender category:** the open-source `genderComputer` package is used through a KRATOS adapter and receives name plus first-author country where available.
6. **Conservative gender rule:** only exact `female` or `male` outputs are accepted. `unisex`, unresolved values, missing names, and backend errors become `unknown`; classifications are never probabilistically forced in the primary analysis.
7. **Interpretation:** `unknown` is a metadata-resolution state, not a substantive demographic category and not self-identified gender.
8. **Primary analytical universe:** fixed **G=4** = female/male × Global North/Global South. Empty substantive cells remain.
9. **Parity reference:** `p*=1/4`.
10. **Coverage rule:** records with unresolved gender or geography remain in the audit trail but do not enter the G=4 parity calculation. Demographic coverage is reported for every corpus.
11. **KJI architecture:** `KJI=KCDI×P`, with `P=mean[A(u)S(u)]` over the four substantive cells, so `KJI<=KCDI` is architectural.
12. **Sensitivity:** unresolved gender is evaluated separately through explicit sensitivity analyses; it is not granted its own parity target.

## Geography validation

The exact-country rule was regression-checked against the manually audited canonical Trade Fairs/MICE corpus (`N=101`), reproducing its validated geography distribution: Global North=62, Global South=38, unknown=1. Additional regression cases ensure that affiliation/location fragments are not converted into countries.

## Implementation

`kratos_core.py` provides first-author parsing; exact first-listed affiliation country resolution; ISO3/region classification; replaceable `GenderResolver`; conservative `GenderComputerResolver`; record-level audit metadata; substantive fixed-G=4 KCDI/KJI; demographic coverage; and document-level concentration metrics.

The primary KRATOS calculation uses complete demographic cases for the four substantive cells. Citation concentration remains a full-corpus document-level diagnostic because it does not require demographic attribution.

## Third-party dependency

`genderComputer` is pinned to commit `f6267615517913e53cb0b882b248f1c2e11b8bbc`; upstream Python code is LGPLv3. KRATOS does not copy its underlying name datasets.

## Migration status

- [x] pure core
- [x] replace `gender-guesser` on branch
- [x] exact-country geography safeguard
- [x] substantive G=4 measurement regime
- [x] unknown separated as audit/coverage state
- [x] regression tests
- [x] canonical MICE geography validation
- [x] four-corpus provisional execution and coverage audit
- [ ] integrate into production `app.py`
- [ ] freeze final four-corpus results after sensitivity analysis
- [ ] freeze manuscript snapshot/version/hash
