# Legacy entrypoint retained for compatibility.
# Streamlit will execute this file; it simply imports the new app.
from frontend.app import *  # noqa: F401,F403
