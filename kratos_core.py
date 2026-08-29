"""Pure computational core for KRATOS.

This module separates demographic metadata resolution from the Streamlit UI and
implements the fixed nine-cell Gender x Region measurement regime used by the
Scientometrics revision.

Design principles
-----------------
* one document is attributed to one first-author cell;
* geography is resolved from the first listed affiliation of the first author;
* name-based gender is a metadata proxy, never self-identified gender;
* ambiguous/unresolved classifications remain ``unknown``;
* the analytical universe is always G=9, including empty cells;
* KJI <= KCDI is an architectural property, not an empirical test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import country_converter as coco
import numpy as np
import pandas as pd


GENDER_STATES: Tuple[str, ...] = ("female", "male", "unknown")
REGION_STATES: Tuple[str, ...] = ("Global North", "Global South", "unknown")
ALL_GROUPS: Tuple[str, ...] = tuple(
    f"{gender} x {region}" for gender in GENDER_STATES for region in REGION_STATES
)
G = len(ALL_GROUPS)

# Canonical KRATOS v2.1.0 set. Note that the executable constant contains 39
# ISO3 codes although an earlier manuscript draft described it as 38.
GLOBAL_NORTH_ISO3 = frozenset(
    {
        "USA", "CAN", "GBR", "FRA", "DEU", "ITA", "ESP", "NLD", "BEL", "LUX",
        "CHE", "AUT", "SWE", "NOR", "DNK", "FIN", "ISL", "IRL", "PRT", "GRC",
        "AUS", "NZL", "JPN", "KOR", "ISR", "CZE", "POL", "HUN", "SVK", "SVN",
        "EST", "LVA", "LTU", "CYP", "MLT", "HRV", "SGP", "HKG", "TWN",
    }
)

_CC = coco.CountryConverter()


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
    country resolution below therefore takes the first recognised country from
    left to right, i.e. the first listed affiliation.
    """
    if not isinstance(authors_with_affiliations, str) or not authors_with_affiliations.strip():
        return ""
    return authors_with_affiliations.split(";")[0].strip()


def _country_to_iso3(candidate: str) -> Optional[str]:
    if not candidate:
        return None
    try:
        result = _CC.convert(names=candidate.strip(), to="ISO3", not_found=None)
    except Exception:
        return None
    if result is None:
        return None
    if isinstance(result, (list, tuple, np.ndarray)):
        result = result[0] if len(result) else None
    if result is None:
        return None
    result = str(result).strip()
    if result in {"", "not found", candidate.strip()}:
        return None
    return result if re.fullmatch(r"[A-Z]{3}", result) else None


def resolve_first_author_country(authors_with_affiliations: object) -> Tuple[str, str, str]:
    """Resolve first-listed first-author country to (country, ISO3, method)."""
    block = extract_first_author_affiliation(authors_with_affiliations)
    if not block:
        return "unknown", "unknown", "first_author_affiliation_unresolved"

    # The first comma-delimited items are normally author/institution/city; the
    # first token recognised as a country identifies the first listed affiliation.
    for token in [part.strip() for part in block.split(",") if part.strip()]:
        iso3 = _country_to_iso3(token)
        if iso3:
            canonical = _CC.convert(names=iso3, to="name_short", not_found=token)
            return str(canonical), iso3, "first_author_first_listed_affiliation"

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
            # Some country strings are unsupported even after country_converter.
            # Retry without location rather than forcing a category.
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


def compute_shannon_entropy_fixed(group_counts: Mapping[str, float]) -> float:
    """Normalised Shannon entropy using the predefined G=9 denominator."""
    total = float(sum(group_counts.get(group, 0.0) for group in ALL_GROUPS))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for group in ALL_GROUPS:
        count = float(group_counts.get(group, 0.0))
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy / math.log(G)


def compute_weight_normalization_fixed(group_weights: Mapping[str, float]) -> float:
    """Range-normalise group citation totals across all nine predefined cells."""
    values = np.array([float(group_weights.get(group, 0.0)) for group in ALL_GROUPS], dtype=float)
    if values.size == 0:
        return 0.0
    w_min = float(values.min())
    w_max = float(values.max())
    if w_max == w_min:
        return 1.0
    return float((values.mean() - w_min) / (w_max - w_min))


def compute_kratos_fixed_g(
    df: pd.DataFrame,
    group_col: str = "group",
    weight_col: str = "Cited by",
    lambda_param: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute revised corpus/group KRATOS quantities with fixed G=9.

    Returns a nine-row group table and corpus-level details. Empty cells are
    retained. ``KJI <= KCDI`` follows by construction because P=mean(A*S) <= 1.
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if weight_col not in df.columns:
        raise ValueError(f"Missing weight column: {weight_col}")
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must lie in [0, 1]")

    work = df[[group_col, weight_col]].copy()
    work[group_col] = work[group_col].where(work[group_col].isin(ALL_GROUPS), "unknown x unknown")
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    n_total = int(len(work))
    total_weight = float(work[weight_col].sum())

    counts = _complete_group_series(work[group_col].value_counts(), ALL_GROUPS)
    weights = _complete_group_series(work.groupby(group_col)[weight_col].sum(), ALL_GROUPS)

    h_prime = compute_shannon_entropy_fixed(counts.to_dict())
    w_norm = compute_weight_normalization_fixed(weights.to_dict())
    kcdi = float((h_prime ** lambda_param) * (w_norm ** (1.0 - lambda_param)))

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
        "n_docs": float(n_total),
        "total_weight": total_weight,
        "H_prime": h_prime,
        "W_norm": w_norm,
        "lambda": float(lambda_param),
        "KCDI": kcdi,
        "P": parity_factor,
        "KJI": kji,
        "R": 1.0 - parity_factor,
        "Delta": kcdi - kji,
    }
    return pd.DataFrame(rows), details


def compute_citation_concentration(weights: Iterable[object]) -> Dict[str, float]:
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
