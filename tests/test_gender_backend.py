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
