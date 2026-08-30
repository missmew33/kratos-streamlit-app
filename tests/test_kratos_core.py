import math

import pandas as pd

from kratos_core import (
    ALL_GROUPS,
    G,
    compute_citation_entropy_fixed,
    compute_kratos_fixed_g,
    extract_first_author,
    extract_given_name,
    resolve_first_author_country,
)


def test_first_author_parsing():
    value = "Ibrahim, Mohamed Nageh (57983496500); Ribeiro, Manuel Alector (55892314700)"
    assert extract_first_author(value) == "Ibrahim, Mohamed Nageh"
    assert extract_given_name(value) == "Mohamed"


def test_first_listed_first_author_affiliation_wins():
    value = (
        "Ibrahim, Mohamed Nageh, School of Hospitality and Tourism Management, "
        "University of Surrey, Guildford, United Kingdom, FACULTY OF TOURISM AND HOTELS, "
        "Luxor University, Luxor, Egypt; Ribeiro, Manuel Alector, Sustainability, "
        "Universidade do Algarve, Faro, Portugal"
    )
    country, iso3, method = resolve_first_author_country(value)
    assert iso3 == "GBR"
    assert method == "first_author_first_listed_affiliation_exact_country"
    assert country != "Portugal"


def test_author_surname_lu_is_not_luxembourg():
    value = (
        "Lu, Ying (Tracy), School of Human Environmental Sciences, "
        "University of Kentucky, Lexington, United States; Cai, Liping A., "
        "College of Health and Human Sciences, West Lafayette, United States"
    )
    _, iso3, _ = resolve_first_author_country(value)
    assert iso3 == "USA"


def test_author_surname_pan_is_not_panama():
    value = (
        "Pan, Mengchuan, Department of Marketing, National Chung Hsing University, "
        "Taichung, Taiwan; Lee, Tzong Ru, Department of Marketing, "
        "National Chung Hsing University, Taichung, Taiwan"
    )
    _, iso3, _ = resolve_first_author_country(value)
    assert iso3 == "TWN"


def test_author_surname_li_is_not_liechtenstein():
    value = (
        "Li, Mengyuan, School of Management, Zhejiang University, Hangzhou, China; "
        "Wang, Wei, School of Business, Beijing, China"
    )
    _, iso3, _ = resolve_first_author_country(value)
    assert iso3 == "CHN"


def test_affiliation_region_code_ch_is_not_switzerland():
    value = (
        "Iacuone, Silvia, Department of Economic Studies, University of G. d'Annunzio "
        "Chieti and Pescara, Chieti, CH, Italy; Zarrilli, Luca, Department of Economic "
        "Studies, University of G. d'Annunzio Chieti and Pescara, Chieti, CH, Italy"
    )
    _, iso3, _ = resolve_first_author_country(value)
    assert iso3 == "ITA"


def test_fuzzy_place_name_bengbu_is_not_united_kingdom():
    value = "Doe, Alex, Research Institute, Bengbu"
    country, iso3, method = resolve_first_author_country(value)
    assert country == "unknown"
    assert iso3 == "unknown"
    assert method == "first_author_affiliation_unresolved"


def test_bengbu_followed_by_explicit_china_resolves_china():
    value = "Doe, Alex, Research Institute, Bengbu, China"
    _, iso3, _ = resolve_first_author_country(value)
    assert iso3 == "CHN"


def test_central_macedonia_is_not_forced_to_north_macedonia():
    value = "Doe, Alex, Research Institute, Central Macedonia"
    country, iso3, _ = resolve_first_author_country(value)
    assert country == "unknown"
    assert iso3 == "unknown"


def test_scopus_turkey_alias_resolves_to_turkiye_iso3():
    value = (
        "Kozak, Nazmi, School of Tourism and Hotel Management, Anadolu Üniversitesi, "
        "Eskisehir, Turkey; Kayar, Çaǧil Hale, School of Tourism and Hotel Management, "
        "Anadolu Üniversitesi, Eskisehir, Turkey"
    )
    _, iso3, method = resolve_first_author_country(value)
    assert iso3 == "TUR"
    assert method == "first_author_first_listed_affiliation_exact_country"


def test_short_country_code_like_tokens_are_not_forced():
    value = "Doe, Alex, Research Unit, ZZ"
    country, iso3, method = resolve_first_author_country(value)
    assert country == "unknown"
    assert iso3 == "unknown"
    assert method == "first_author_affiliation_unresolved"


def test_fixed_g4_retains_four_substantive_cells_and_excludes_unknown_from_parity():
    df = pd.DataFrame(
        {
            "group": [
                "female x Global North",
                "female x Global North",
                "male x Global South",
                "unknown x Global North",
            ],
            "Cited by": [10, 5, 2, 1],
        }
    )
    groups, details = compute_kratos_fixed_g(df)
    assert list(groups["Group"]) == list(ALL_GROUPS)
    assert len(groups) == 4
    assert G == 4
    assert details["G"] == 4
    assert abs(details["p_star"] - 0.25) < 1e-12
    assert details["n_docs_input"] == 4
    assert details["n_docs_resolved"] == 3
    assert abs(details["demographic_coverage"] - 0.75) < 1e-12
    assert 0 <= details["KJI"] <= details["KCDI"] <= 1
    assert abs(details["R"] - (1 - details["P"])) < 1e-12
    assert "W_norm" not in details
    assert "H_D_prime" in details
    assert "H_C_prime" in details


def test_unknown_only_records_do_not_create_substantive_parity_cell():
    df = pd.DataFrame(
        {
            "group": ["unknown x Global North", "female x unknown"],
            "Cited by": [10, 5],
        }
    )
    groups, details = compute_kratos_fixed_g(df)
    assert len(groups) == 4
    assert details["n_docs_resolved"] == 0
    assert details["demographic_coverage"] == 0
    assert details["KCDI"] == 0
    assert details["KJI"] == 0


def test_empty_substantive_cells_are_not_dropped_from_entropy_reference():
    df = pd.DataFrame(
        {
            "group": ["female x Global North"] * 5,
            "Cited by": [1, 1, 1, 1, 1],
        }
    )
    groups, details = compute_kratos_fixed_g(df)
    assert len(groups) == 4
    assert details["H_D_prime"] == 0.0
    assert details["H_C_prime"] == 0.0
    assert (groups["n_docs"] == 0).sum() == 3


def test_citation_entropy_is_one_for_equal_group_citation_totals():
    citations = {group: 10.0 for group in ALL_GROUPS}
    assert abs(compute_citation_entropy_fixed(citations) - 1.0) < 1e-12


def test_citation_entropy_is_zero_for_complete_concentration():
    citations = {group: 0.0 for group in ALL_GROUPS}
    citations[ALL_GROUPS[0]] = 100.0
    assert compute_citation_entropy_fixed(citations) == 0.0


def test_citation_entropy_is_near_one_for_near_equal_totals():
    citations = dict(zip(ALL_GROUPS, [10.0, 10.0, 10.0, 11.0]))
    h_c = compute_citation_entropy_fixed(citations)
    assert 0.99 < h_c < 1.0


def test_kcdi_endpoint_lambda_values_are_defined():
    df = pd.DataFrame(
        {
            "group": list(ALL_GROUPS),
            "Cited by": [1, 2, 3, 4],
        }
    )
    _, details_zero = compute_kratos_fixed_g(df, lambda_param=0.0)
    _, details_one = compute_kratos_fixed_g(df, lambda_param=1.0)
    assert math.isclose(details_zero["KCDI"], details_zero["H_C_prime"])
    assert math.isclose(details_one["KCDI"], details_one["H_D_prime"])


def test_uniform_scaling_of_citations_does_not_change_entropy_or_parity():
    base = pd.DataFrame(
        {
            "group": list(ALL_GROUPS),
            "Cited by": [2, 4, 6, 8],
        }
    )
    scaled = base.copy()
    scaled["Cited by"] = scaled["Cited by"] * 10
    _, d1 = compute_kratos_fixed_g(base)
    _, d2 = compute_kratos_fixed_g(scaled)
    assert math.isclose(d1["H_C_prime"], d2["H_C_prime"])
    assert math.isclose(d1["P"], d2["P"])
    assert math.isclose(d1["KCDI"], d2["KCDI"])
    assert math.isclose(d1["KJI"], d2["KJI"])
