"""Pure computational core for KRATOS.

This module separates demographic metadata resolution from the Streamlit UI and
implements the substantive four-cell Gender x Region measurement regime used by
the Scientometrics revision.

Design principles
-----------------
* one document is attributed to one first-author cell;
* geography is resolved from the first listed affiliation of the first author;
* name-based gender is a metadata proxy, never self-identified gender;
* ambiguous/unresolved classifications remain ``unknown`` audit states;
* the substantive analytical universe is G=4: female/male x Global North/South;
* unresolved metadata are excluded from parity expectations and reported as
  measurement coverage rather than treated as substantive demographic groups;
* KCDI combines document-distribution entropy and citation-distribution entropy;
* KJI = KCDI * P, so KJI <= KCDI is architectural rather than empirical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import country_converter as coco
import numpy as np
import pandas as pd


# Audit states preserved in record-level outputs.
GENDER_STATES: Tuple[str, ...] = ("female", "male", "unknown")
REGION_STATES: Tuple[str, ...] = ("Global North", "Global South", "unknown")
AUDIT_GROUPS: Tuple[str, ...] = tuple(
    f"{gender} x {region}" for gender in GENDER_STATES for region in REGION_STATES
)

# Substantive measurement universe. Unknown is measurement uncertainty, not a
# demographic category with its own parity expectation.
SUBSTANTIVE_GENDER_STATES: Tuple[str, ...] = ("female", "male")
SUBSTANTIVE_REGION_STATES: Tuple[str, ...] = ("Global North", "Global South")
ALL_GROUPS: Tuple[str, ...] = tuple(
    f"{gender} x {region}"
    for gender in SUBSTANTIVE_GENDER_STATES
    for region in SUBSTANTIVE_REGION_STATES
)
G = len(ALL_GROUPS)

# Canonical KRATOS set. The executable constant contains 39 ISO3 codes.
GLOBAL_NORTH_ISO3 = frozenset(
    {
        "USA", "CAN", "GBR", "FRA", "DEU", "ITA", "ESP", "NLD", "BEL", "LUX",
        "CHE", "AUT", "SWE", "NOR", "DNK", "FIN", "ISL", "IRL", "PRT", "GRC",
        "AUS", "NZL", "JPN", "KOR", "ISR", "CZE", "POL", "HUN", "SVK", "SVN",
        "EST", "LVA", "LTU", "CYP", "MLT", "HRV", "SGP", "HKG", "TWN",
    }
)

_CC = coco.CountryConverter()


def _normalise_country_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip(" .")
    return text.casefold()


def _build_exact_country_lookup() -> Dict[str, str]:
    """Build an exact country-name lookup from country-converter metadata.

    Only exact short/official names are admitted. The library's permissive regex
    aliases are deliberately not used because they can map place names such as
    ``Bengbu`` or ``Ningbo`` to unrelated countries.
    """
    lookup: Dict[str, str] = {}
    for _, row in _CC.data.iterrows():
        iso3 = str(row.get("ISO3", "") or "").strip()
        if not re.fullmatch(r"[A-Z]{3}", iso3):
            continue
        for column in ("name_short", "name_official"):
            key = _normalise_country_name(row.get(column))
            if key:
                lookup[key] = iso3

    lookup.update(
        {
            "macao": "MAC",
            "macao sar": "MAC",
            "viet nam": "VNM",
            "türkiye": "TUR",
            "turkey": "TUR",
        }
    )
    return lookup


_EXACT_COUNTRY_TO_ISO3 = _build_exact_country_lookup()


class GenderResolver(Protocol):
    """Interface for replaceable, auditable name-based resolvers."""

    method_name: str

    def resolve(self, full_name: str, given_name: str, country: Optional[str]) -> Dict[str, str]:
        """Return category/raw/status/method fields for one first author."""


@dataclass(frozen=True)
class DemographicResolution:
    first_author: str
    given_name: str
    first_author_affiliation: str
    country: str
    country_iso3: str
    region: str
    region_method: str
    gender_category: str
    gender_raw_result: str
    gender_method: str
    gender_resolution_status: str
    group: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def extract_first_author(full_names: object) -> str:
    """Return the first author string from a Scopus author field."""
    if not isinstance(full_names, str) or not full_names.strip():
        return ""
    first = full_names.split(";")[0].strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", first).strip()


def extract_given_name(full_names: object) -> str:
    """Extract the first given name from the first Scopus author."""
    first = extract_first_author(full_names)
    if not first:
        return ""
    if "," in first:
        parts = first.split(",", 1)
        given_part = parts[1].strip()
    else:
        given_part = first.strip()
    given = given_part.split()[0] if given_part else ""
    given = given.replace("-", " ").split()[0] if given else ""
    return re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]", "", given).strip()


def extract_first_author_affiliation(authors_with_affiliations: object) -> str:
    """Return the first author's complete affiliation block.

    Scopus separates author blocks with semicolons. If a first author has more
    than one affiliation, all of those affiliations may occur inside this block;
    country resolution therefore takes the first exact country-name token from
    left to right.
    """
    if not isinstance(authors_with_affiliations, str) or not authors_with_affiliations.strip():
        return ""
    return authors_with_affiliations.split(";")[0].strip()


def _country_to_iso3(candidate: str) -> Optional[str]:
    """Resolve only exact country names; never fuzzy location/name fragments."""
    key = _normalise_country_name(candidate)
    return _EXACT_COUNTRY_TO_ISO3.get(key)


def _explicit_country_tokens(first_author_block: str) -> Sequence[str]:
    """Return conservative country candidates from a Scopus author block."""
    parts = [part.strip() for part in first_author_block.split(",") if part.strip()]
    affiliation_parts = parts[2:] if len(parts) >= 3 else []

    candidates = []
    for token in affiliation_parts:
        cleaned = re.sub(r"\s+", " ", token).strip(" .")
        letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]", "", cleaned)
        if len(letters) < 4:
            continue
        if re.fullmatch(r"[A-Z]{2,3}", cleaned):
            continue
        candidates.append(cleaned)
    return candidates


def resolve_first_author_country(authors_with_affiliations: object) -> Tuple[str, str, str]:
    """Resolve first-listed first-author country to (country, ISO3, method)."""
    block = extract_first_author_affiliation(authors_with_affiliations)
    if not block:
        return "unknown", "unknown", "first_author_affiliation_unresolved"

    for token in _explicit_country_tokens(block):
        iso3 = _country_to_iso3(token)
        if iso3:
            canonical = _CC.convert(names=iso3, to="name_short", not_found=token)
            return str(canonical), iso3, "first_author_first_listed_affiliation_exact_country"

    return "unknown", "unknown", "first_author_affiliation_unresolved"


def classify_region(iso3: object) -> str:
    if not isinstance(iso3, str) or not re.fullmatch(r"[A-Z]{3}", iso3):
        return "unknown"
    return "Global North" if iso3 in GLOBAL_NORTH_ISO3 else "Global South"


class GenderComputerResolver:
    """Conservative adapter around the open-source ``genderComputer`` package.

    Only exact ``male`` or ``female`` outputs are accepted. ``unisex``, missing,
    errors and any other output become ``unknown``. No probability is fabricated.
    Country is supplied when available; the raw result and resolution status are
    retained for record-level auditing.
    """

    method_name = "genderComputer@f626761"

    def __init__(self) -> None:
        try:
            from genderComputer import GenderComputer  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "genderComputer is not installed. Install the pinned dependency from requirements.txt."
            ) from exc
        self._resolver = GenderComputer()

    def resolve(self, full_name: str, given_name: str, country: Optional[str]) -> Dict[str, str]:
        if not full_name and not given_name:
            return {
                "category": "unknown",
                "raw_result": "",
                "status": "missing_name",
                "method": self.method_name,
            }

        location = None if not country or country == "unknown" else country
        query_name = full_name or given_name
        try:
            raw = self._resolver.resolveGender(query_name, location)
        except Exception:
            try:
                raw = self._resolver.resolveGender(query_name, None)
                location = None
            except Exception:
                raw = None

        raw_text = "" if raw is None else str(raw).strip().lower()
        category = raw_text if raw_text in {"female", "male"} else "unknown"
        if category in {"female", "male"}:
            status = "resolved_country_aware" if location else "resolved_name_only"
        elif raw_text == "unisex":
            status = "ambiguous"
        else:
            status = "unresolved"

        return {
            "category": category,
            "raw_result": raw_text,
            "status": status,
            "method": self.method_name,
        }


def resolve_demographics(
    author_full_names: object,
    authors_with_affiliations: object,
    gender_resolver: Optional[GenderResolver] = None,
) -> DemographicResolution:
    first_author = extract_first_author(author_full_names)
    given_name = extract_given_name(author_full_names)
    affiliation = extract_first_author_affiliation(authors_with_affiliations)
    country, iso3, region_method = resolve_first_author_country(authors_with_affiliations)
    region = classify_region(iso3)

    if gender_resolver is None:
        gender_result = {
            "category": "unknown",
            "raw_result": "",
            "status": "resolver_not_configured",
            "method": "none",
        }
    else:
        gender_result = gender_resolver.resolve(first_author, given_name, country)

    gender_category = gender_result.get("category", "unknown")
    if gender_category not in GENDER_STATES:
        gender_category = "unknown"

    group = f"{gender_category} x {region}"
    return DemographicResolution(
        first_author=first_author,
        given_name=given_name,
        first_author_affiliation=affiliation,
        country=country,
        country_iso3=iso3,
        region=region,
        region_method=region_method,
        gender_category=gender_category,
        gender_raw_result=gender_result.get("raw_result", ""),
        gender_method=gender_result.get("method", "unknown"),
        gender_resolution_status=gender_result.get("status", "unresolved"),
        group=group,
    )


def enrich_first_author_metadata(
    df: pd.DataFrame,
    author_col: str = "Author full names",
    authors_with_affiliations_col: str = "Authors with affiliations",
    gender_resolver: Optional[GenderResolver] = None,
) -> pd.DataFrame:
    """Add auditable first-author demographic metadata to a dataframe."""
    out = df.copy()
    if author_col not in out.columns:
        raise ValueError(f"Missing author column: {author_col}")
    if authors_with_affiliations_col not in out.columns:
        raise ValueError(f"Missing affiliation column: {authors_with_affiliations_col}")

    resolved = [
        resolve_demographics(author, affiliation, gender_resolver)
        for author, affiliation in zip(out[author_col], out[authors_with_affiliations_col])
    ]
    metadata = pd.DataFrame([item.to_dict() for item in resolved], index=out.index)
    return pd.concat([out, metadata], axis=1)


def _complete_group_series(series: pd.Series, values: Sequence[str]) -> pd.Series:
    return series.reindex(values, fill_value=0.0).astype(float)


def compute_shannon_entropy_fixed(group_values: Mapping[str, float]) -> float:
    """Normalised Shannon entropy over the fixed substantive G=4 universe.

    The mapping may contain document counts or non-negative group citation totals.
    Empty cells remain in the fixed denominator through normalisation by ln(G).
    """
    values = [max(0.0, float(group_values.get(group, 0.0))) for group in ALL_GROUPS]
    total = float(sum(values))
    if total <= 0.0:
        return 0.0

    entropy = 0.0
    for value in values:
        if value > 0.0:
            share = value / total
            entropy -= share * math.log(share)
    return float(entropy / math.log(G))


def compute_citation_entropy_fixed(group_citations: Mapping[str, float]) -> float:
    """Normalised Shannon entropy of citation shares over fixed G=4."""
    return compute_shannon_entropy_fixed(group_citations)


def _weighted_geometric_kcdi(h_d_prime: float, h_c_prime: float, lambda_param: float) -> float:
    """Combine document and citation evenness without undefined 0**0 endpoints."""
    if lambda_param == 0.0:
        return float(h_c_prime)
    if lambda_param == 1.0:
        return float(h_d_prime)
    return float((h_d_prime ** lambda_param) * (h_c_prime ** (1.0 - lambda_param)))


def compute_kratos_fixed_g(
    df: pd.DataFrame,
    group_col: str = "group",
    weight_col: str = "Cited by",
    lambda_param: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute KRATOS quantities under the substantive fixed G=4 regime.

    Only records with resolved female/male gender and Global North/Global South
    geography enter the primary calculation. Unresolved records remain audit
    states and contribute to demographic-coverage reporting. KCDI combines
    document-distribution entropy (H_D_prime) and citation-distribution entropy
    (H_C_prime). P=mean(A*S), KJI=KCDI*P, and KJI<=KCDI by construction.
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if weight_col not in df.columns:
        raise ValueError(f"Missing weight column: {weight_col}")
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must lie in [0, 1]")

    input_work = df[[group_col, weight_col]].copy()
    n_input = int(len(input_work))

    work = input_work[input_work[group_col].isin(ALL_GROUPS)].copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    n_total = int(len(work))
    total_weight = float(work[weight_col].sum())

    counts = _complete_group_series(work[group_col].value_counts(), ALL_GROUPS)
    weights = _complete_group_series(work.groupby(group_col)[weight_col].sum(), ALL_GROUPS)

    h_d_prime = compute_shannon_entropy_fixed(counts.to_dict())
    h_c_prime = compute_citation_entropy_fixed(weights.to_dict())
    kcdi = (
        _weighted_geometric_kcdi(h_d_prime, h_c_prime, lambda_param)
        if n_total > 0
        else 0.0
    )

    p_star = 1.0 / G
    rows = []
    parity_products = []
    for group in ALL_GROUPS:
        n_docs = int(counts.loc[group])
        citations = float(weights.loc[group])
        p_u = n_docs / n_total if n_total else 0.0
        s_u = citations / total_weight if total_weight > 0 else 0.0
        a_u = max(0.0, 1.0 - abs(p_u - p_star) / p_star)
        s_factor = 0.0 if p_u == 0 else max(0.0, 1.0 - abs((s_u / p_u) - 1.0))
        parity = a_u * s_factor
        parity_products.append(parity)
        rows.append(
            {
                "Group": group,
                "n_docs": n_docs,
                "total_weight": citations,
                "doc_share": p_u,
                "weight_share": s_u,
                "citation_share": s_u,
                "signed_gap": s_u - p_u,
                "A_factor": a_u,
                "S_factor": s_factor,
                "A_times_S": parity,
                "KJI_group": kcdi * parity,
            }
        )

    parity_factor = float(np.mean(parity_products)) if parity_products else 0.0
    kji = kcdi * parity_factor
    details = {
        "G": float(G),
        "p_star": p_star,
        "n_docs_input": float(n_input),
        "n_docs": float(n_total),
        "n_docs_resolved": float(n_total),
        "demographic_coverage": (float(n_total / n_input) if n_input else 0.0),
        "total_weight": total_weight,
        "H_D_prime": h_d_prime,
        "H_C_prime": h_c_prime,
        "lambda": float(lambda_param),
        "KCDI": kcdi,
        "P": parity_factor,
        "KJI": kji,
        "R": 1.0 - parity_factor,
        "Delta": kcdi - kji,
    }
    return pd.DataFrame(rows), details


def compute_citation_concentration(weights: Iterable[object]) -> Dict[str, float]:
    """Document-level citation concentration, independent of demographic coverage."""
    values = pd.to_numeric(pd.Series(list(weights)), errors="coerce").fillna(0.0).clip(lower=0.0)
    n = int(len(values))
    total = float(values.sum())
    if n == 0 or total <= 0:
        return {"n": float(n), "total": total, "top10_share": 0.0, "HHI": 0.0, "Gini": 0.0}

    n_top = max(1, int(math.ceil(0.10 * n)))
    top10_share = float(values.nlargest(n_top).sum() / total)
    shares = values / total
    hhi = float(np.square(shares).sum())

    sorted_values = np.sort(values.to_numpy(dtype=float))
    index = np.arange(1, n + 1, dtype=float)
    gini = float((2.0 * np.dot(index, sorted_values)) / (n * total) - (n + 1.0) / n)

    return {
        "n": float(n),
        "total": total,
        "n_top": float(n_top),
        "top10_share": top10_share,
        "HHI": hhi,
        "Gini": gini,
    }
