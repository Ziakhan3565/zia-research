from __future__ import annotations

# Stable entrypoint: keep the original dashboard exactly as-is.
# The previous runtime source-injection wrapper could crash Streamlit when
# the original dashboard text changed. This entrypoint deliberately avoids
# that fragile mechanism and imports the original dashboard directly.
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "dashboard_tv.py"), run_name="__main__")
