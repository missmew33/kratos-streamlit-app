# Third-party notices

## genderComputer

KRATOS demographic resolution v2 uses the open-source `genderComputer` Python package as an external dependency, pinned to upstream commit:

`f6267615517913e53cb0b882b248f1c2e11b8bbc`

Upstream repository: `tue-mdse/genderComputer`.

The upstream project states that its Python code is licensed under LGPLv3 and that its underlying name datasets carry separate open-data/database terms. KRATOS does **not** copy those upstream name datasets into this repository; the package is installed as a pinned dependency.

The KRATOS adapter applies an additional conservative decision rule: only exact `female` or `male` results are accepted; ambiguous, unisex, unresolved, missing, or failed classifications are retained as `unknown`.

Name-based categories are metadata-derived analytical proxies and are not interpreted as self-identified gender.
