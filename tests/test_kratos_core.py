import pandas as pd

from kratos_core import (
    ALL_GROUPS,
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
    assert method == "first_author_first_listed_affiliation"
    assert country != "Portugal"


def test_fixed_g_retains_all_nine_cells_and_kji_is_bounded():
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
    assert len(groups) == 9
    assert details["G"] == 9
    assert abs(details["p_star"] - (1 / 9)) < 1e-12
    assert 0 <= details["KJI"] <= details["KCDI"] <= 1
    assert abs(details["R"] - (1 - details["P"])) < 1e-12


def test_empty_cells_are_not_dropped_from_entropy_reference():
    df = pd.DataFrame(
        {
            "group": ["female x Global North"] * 5,
            "Cited by": [1, 1, 1, 1, 1],
        }
    )
    groups, details = compute_kratos_fixed_g(df)
    assert len(groups) == 9
    assert details["H_prime"] == 0.0
    assert (groups["n_docs"] == 0).sum() == 8
