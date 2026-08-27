# ZIA Research Terminal - stable entrypoint
# The previous v7 implementation called calculate_all_signals() as if it
# returned a dict. The Research Lab actually returns (features, payload, weights).
# Keep v7 as a compatibility entrypoint and run the consolidated dashboard.
from dashboard import *
