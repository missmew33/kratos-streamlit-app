from streamlit.testing.v1 import AppTest


def test_production_app_import_and_version():
    import app

    assert app.APP_VERSION == "2.2.0"
    assert callable(app.main)


def test_v22_compatibility_entrypoint_uses_production_main():
    import app
    import app_v2_2

    assert app_v2_2.main is app.main


def test_production_streamlit_app_starts_without_exception():
    app_test = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app_test.exception
