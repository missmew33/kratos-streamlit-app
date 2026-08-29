# KRATOS demographic resolution v2

## Purpose

This branch replaces the legacy `gender-guesser` enrichment path with an auditable first-author demographic-resolution layer and aligns the executable code with the revised Scientometrics measurement regime.

## Measurement rules

1. **Unit of attribution:** one document is assigned to one analytical cell using the first author.
2. **Geography:** the country is taken from the **first listed affiliation of the first author** in Scopus `Authors with affiliations`. A country attached to a later coauthor must never determine the record's region.
3. **Region:** country is standardised to ISO3 with `country-converter` and mapped to the canonical KRATOS Global North set. The executable set contains **39 ISO3 codes** and reproduces the existing taxonomy hash `39dcca84388a6762`.
4. **Name-based gender category:** the open-source `genderComputer` package is used through a KRATOS adapter and receives name plus first-author country where available.
5. **Conservative decision rule:** only exact `female` or `male` outputs are accepted. `unisex`, unresolved values, missing names, and backend errors become `unknown`; classifications are never probabilistically forced.
6. **Interpretation:** `female`, `male`, and `unknown` are metadata-derived analytical states. They do not measure self-identified gender.
7. **Analytical universe:** the cross-classification is fixed at `G=9` (`female/male/unknown` × `Global North/Global South/unknown`) and empty cells remain in the reference space.
8. **Parity reference:** `p*=1/9` in every corpus. The denominator does not shrink when one or more cells are empty.
9. **KJI architecture:** `KJI = KCDI × P`, where `P = mean[A(u)S(u)]` over all nine cells. Therefore `KJI <= KCDI` is an architectural property and must not be interpreted as an empirical test.

## Implementation

`kratos_core.py` contains the pure, UI-independent implementation:

- first-author parsing;
- first-listed affiliation country resolution;
- ISO3 and region classification;
- replaceable `GenderResolver` protocol;
- conservative `GenderComputerResolver` adapter;
- record-level audit fields;
- fixed-nine-cell KCDI/KJI computation;
- citation concentration metrics.

The Streamlit UI should call these functions rather than reimplementing demographic resolution or metric formulas inside `app.py`.

## Third-party dependency

`genderComputer` is pinned to commit `f6267615517913e53cb0b882b248f1c2e11b8bbc` for reproducibility. The upstream Python code is LGPLv3; its name data have separate open-data terms described by the upstream project. KRATOS does not copy those datasets into this repository.

## Migration status

- [x] Create pure computational core.
- [x] Replace `gender-guesser` in branch dependencies.
- [x] Add tests for first-author geography and fixed `G=9` behaviour.
- [ ] Integrate `kratos_core.py` into `app.py`.
- [ ] Run the canonical MICE corpus through the branch implementation.
- [ ] Export record-level demographic audit table.
- [ ] Freeze the manuscript analysis snapshot and version/hash.
