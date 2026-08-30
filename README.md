# KRATOS — Bibliometric Recognition Diagnostics

KRATOS is an auditable Python/Streamlit research instrument for examining whether bibliometric recognition signals are distributed comparably across predefined knowledge-producing groups.

The current methodological branch implements the measurement regime used in the Scientometrics revision. It should not be interpreted as a ground-truth measure of epistemic justice or epistemic change.

## Current measurement regime

The substantive analytical universe is fixed at **G=4**:

- female × Global North
- female × Global South
- male × Global North
- male × Global South

Gender is a metadata-derived name-based proxy, not self-identified gender. Unresolved gender or geography remains an **audit/coverage state** and is not assigned a parity target.

One document is attributed to the first author. Geography is resolved from the first listed affiliation of that author using conservative exact-country matching.

## KCDI architecture

KRATOS separates marginal distribution from participation–recognition alignment.

### Document-distribution entropy

`H_D_prime` is normalised Shannon entropy over document shares in the fixed four-cell universe:

```text
H_D_prime = -sum(p_u * ln(p_u)) / ln(4)
```

### Citation-distribution entropy

`H_C_prime` applies the same normalised entropy to group citation shares:

```text
H_C_prime = -sum(s_u * ln(s_u)) / ln(4)
```

The earlier range-normalised citation component `W_norm` has been removed from the computational core. Its min/mean/max construction did not behave monotonically as a citation-balance measure and was therefore replaced after mathematical audit.

### KCDI

```text
KCDI = H_D_prime^lambda * H_C_prime^(1-lambda)
```

The primary specification uses `lambda = 0.5`; sensitivity analyses evaluate alternative values.

## Participation–recognition parity

For each substantive group `u`, KRATOS computes:

```text
A(u) = max(0, 1 - abs(p_u - 0.25) / 0.25)
S(u) = max(0, 1 - abs(s_u / p_u - 1))
```

with `S(u)=0` when `p_u=0`.

The corpus-level parity factor is:

```text
P = mean(A(u) * S(u))
R = 1 - P
KJI = KCDI * P
```

`KJI <= KCDI` is an architectural identity, not empirical evidence of canonical closure, epistemic stigma, discrimination, or epistemic injustice.

## Citation concentration

Gini, HHI and Top-10% citation share are computed separately on the complete canonical corpus. They are descriptive concentration diagnostics and are not components of KCDI or KJI.

## Demographic resolution

The current branch uses the open-source `genderComputer` backend pinned to revision `f6267615517913e53cb0b882b248f1c2e11b8bbc`.

Only exact `female` or `male` outputs are accepted. `unisex`, missing, unresolved and error states remain `unknown`.

Country resolution uses exact country-name tokens from the first author's first listed affiliation. Fuzzy location matching is deliberately avoided.

## Reproducibility tools

Core computation:

```text
kratos_core.py
```

Run a canonical Scopus-derived CSV through the auditable pipeline:

```bash
python scripts/kratos_analyze_csv.py input.csv --output-dir kratos_output
```

Run unresolved-gender and matched-size sensitivity analyses:

```bash
python scripts/kratos_g4_sensitivity.py kratos_output/demographic_audit.csv \
  --matched-n 101 --B 1000 --seed 20260831
```

Licensed Scopus source records are not stored in this public repository.

## Streamlit status

`app_v2_2.py` is the release-candidate wrapper around the legacy Streamlit interface and uses the revised `kratos_core.py` calculations. The production `app.py` still contains legacy UI text and is intentionally not yet treated as the frozen manuscript implementation. The release-candidate wrapper exposes temporary legacy renderer aliases only to avoid breaking that UI; reproducibility exports use `H_D_prime` and `H_C_prime`.

## Tests

```bash
pytest -q
```

Regression tests cover the fixed G=4 universe, treatment of unresolved metadata, conservative geography parsing, citation-entropy boundary behaviour, scale invariance, and the architectural constraint `KJI <= KCDI`.

## Licence

MIT.
