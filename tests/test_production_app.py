def test_production_app_import_and_version():
    import app

    assert app.APP_VERSION == "2.2.0"
    assert callable(app.main)


def test_v22_compatibility_entrypoint_uses_production_main():
    import app
    import app_v2_2

    assert app_v2_2.main is app.main
