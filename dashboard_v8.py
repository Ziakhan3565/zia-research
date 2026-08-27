"""ZIA Research Terminal v8 compatibility entrypoint.

The Research Lab calculate_all_signals() API returns
(features, payload, dynamic_weights), while dashboard_v7 expected the
payload dictionary directly. This adapter unwraps the tuple before v7
uses it. The underlying Research Lab is not changed.
"""
from __future__ import annotations

from functools import wraps

from src.research_lab import TenPaperResearchLab

_original_calculate_all_signals = TenPaperResearchLab.calculate_all_signals

if not getattr(TenPaperResearchLab, "_zia_dashboard_payload_adapter", False):
    @wraps(_original_calculate_all_signals)
    def _dashboard_calculate_all_signals(*args, **kwargs):
        result = _original_calculate_all_signals(*args, **kwargs)
        # Current Research Lab contract: (features, payload, weights).
        if isinstance(result, tuple) and len(result) >= 2:
            payload = result[1]
            if isinstance(payload, dict):
                return payload
        # Future-proof: if the engine later returns a dict directly,
        # pass it through unchanged.
        return result

    TenPaperResearchLab.calculate_all_signals = _dashboard_calculate_all_signals
    TenPaperResearchLab._zia_dashboard_payload_adapter = True

# Load the polished dashboard after the compatibility adapter is installed.
# dashboard_v7 will therefore receive the dictionary payload it expects.
from dashboard_v7 import *  # noqa: E402,F401,F403
