from kratos_core import GenderComputerResolver


def test_gendercomputer_backend_is_available_and_auditable():
    resolver = GenderComputerResolver()
    result = resolver.resolve("Bogdan", "Bogdan", None)
    assert result["method"].startswith("genderComputer@")
    assert result["category"] in {"male", "female", "unknown"}
    assert result["status"] in {
        "resolved_country_aware",
        "resolved_name_only",
        "ambiguous",
        "unresolved",
        "missing_name",
    }


def test_country_context_is_accepted_without_forcing_ambiguity():
    resolver = GenderComputerResolver()
    result = resolver.resolve("Andrea", "Andrea", "Italy")
    assert result["category"] in {"male", "female", "unknown"}
    if result["raw_result"] == "unisex":
        assert result["category"] == "unknown"
        assert result["status"] == "ambiguous"


def test_upstream_documented_bogdan_example():
    resolver = GenderComputerResolver()
    result = resolver.resolve("Bogdan", "Bogdan", None)
    assert result["raw_result"] == "male"
    assert result["category"] == "male"
    assert result["status"] == "resolved_name_only"


def test_upstream_documented_country_context_changes_andrea():
    resolver = GenderComputerResolver()
    italy = resolver.resolve("Andrea", "Andrea", "Italy")
    germany = resolver.resolve("Andrea", "Andrea", "Germany")
    assert italy["raw_result"] == "male"
    assert germany["raw_result"] == "female"
    assert italy["category"] == "male"
    assert germany["category"] == "female"
    assert italy["status"] == "resolved_country_aware"
    assert germany["status"] == "resolved_country_aware"


def test_upstream_documented_ashley_australia_example():
    resolver = GenderComputerResolver()
    result = resolver.resolve("Ashley Maher", "Ashley", "Australia")
    assert result["raw_result"] == "female"
    assert result["category"] == "female"
    assert result["status"] == "resolved_country_aware"


def test_unisex_or_unresolved_never_becomes_binary():
    resolver = GenderComputerResolver()
    # The adapter rule is the point of this test: any backend result other than
    # exact male/female must remain unknown.
    for name in ["Alex", "Sam", "Robin"]:
        result = resolver.resolve(name, name, None)
        if result["raw_result"] not in {"male", "female"}:
            assert result["category"] == "unknown"
            assert result["status"] in {"ambiguous", "unresolved"}
