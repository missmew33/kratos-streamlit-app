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
7. **Analytical universe:** cross-classification fixed G=9 and empty cells remain.
8. **Parity reference:** p*=1/9.
9. **KJI architecture:** KJI=KCDI×P, P=mean[A(u)S(u)] over 9, so KJI<=KCDI architectural.

## Geography resolution safeguard

Scopus author-affiliation blocks begin with the author name, e.g. `Surname, Given names, ...`. Country resolution therefore skips the author-name components before scanning affiliation tokens and accepts only explicit country-name tokens. Short alphabetic tokens and country-code-like abbreviations are rejected because they can be affiliation or name fragments (`Lu`, `Li`, `Pan`, `CH`) that `country-converter` may otherwise reinterpret as countries. If no explicit country is resolved, the record remains `unknown`.

This rule was regression-checked against the manually audited canonical Trade Fairs/MICE corpus (N=101), reproducing its validated geography distribution: Global North=62, Global South=38, unknown=1.

## Implementation

`kratos_core.py`: first-author parsing; first-listed affiliation country resolution; ISO3/region; replaceable `GenderResolver`; conservative `GenderComputerResolver`; audit metadata; fixed-nine KCDI/KJI; concentration metrics.

## Third-party dependency

`genderComputer` is pinned to commit `f6267615517913e53cb0b882b248f1c2e11b8bbc`; upstream Python code is LGPLv3. KRATOS does not copy its underlying name datasets.

## Migration status

- [x] pure core
- [x] replace `gender-guesser` on branch
- [x] tests first-author geography/fixed G
- [x] restore demographic-resolution methodology plan
- [ ] integrate into `app.py`
- [ ] run canonical MICE
- [ ] export audit
- [ ] freeze manuscript snapshot/version/hash
