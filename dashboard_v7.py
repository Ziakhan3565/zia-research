"""ZIA Research Terminal v7 entrypoint.
Use the dedicated multi-crypto live dashboard with timeframe locks."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
exec(compile((ROOT/'dashboard_live_v2.py').read_text(encoding='utf-8'),str(ROOT/'dashboard_live_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'dashboard_live_v2.py')})
