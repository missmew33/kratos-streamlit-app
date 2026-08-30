"""Compatibility entrypoint for KRATOS v2.2.

The validated production Streamlit interface now lives in ``app.py``. This file
is retained so existing deployment or local commands that reference
``app_v2_2.py`` continue to execute the same production application.
"""

from app import main


if __name__ == "__main__":
    main()
