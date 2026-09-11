"""Compatibility/production entrypoint for KRATOS v2.3.

The primary analysis remains implemented in ``app.py`` and ``kratos_core.py``.
Streamlit automatically exposes the v2.3 robustness page from ``pages/``.
"""

import app

app.APP_VERSION = "2.3.0"

if __name__ == "__main__":
    app.main()
