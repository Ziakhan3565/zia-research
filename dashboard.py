# ZIA Research Terminal - actual Streamlit entrypoint
# The production dashboard lives in dashboard_live_v2.py.
# Keeping this tiny launcher ensures Streamlit runs the new UI even when
# the deployment is configured to execute dashboard.py.
from dashboard_live_v2 import *
