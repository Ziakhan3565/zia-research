from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import requests


# ============================================================
# CONFIGURATION
# ============================================================

BINANCE_API = "https://api.binance.com/api/v3/klines"

DEFAULT_SYMBOL = "BTCUSDT"

SUPPORTED_TIMEFRAMES = {
    "YEARLY": "1y",
    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
    "5M": "5m",
    "1M_CHART": "1m",  # 1-minute chart timeframe (named uniquely to avoid collision with Monthly)
}

# If you want standard UI naming matching your exact preference:
# "1M" (minute) vs "MONTHLY" (1M capital handled safely)
# Let's map clean keys:
TIMEFRAME_MAPPING = {
    "YEARLY": "1y",
    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
    "5M": "5m",
    "1M": "1m",
}


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class TRILineLevels:
    timeframe: str

    open: float
    high: float
    low: float
    close: float

    body_high: float
    body_low: float

    body_50: float
    upper_50: float
    lower_50: float

    candle_time: int


# ============================================================
# TRI LINE ENGINE
# ============================================================

class TRILineEngine:

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        enabled_timeframes: Optional[Dict[str, bool]] = None,
        timeout: int = 10,
    ):

        self.symbol = symbol.upper()
        self.timeout = timeout

        # ----------------------------------------------------
        # DEFAULT TIMEFRAME SETTINGS
        # ----------------------------------------------------

        self.enabled = {
            "YEARLY": True,
            "MONTHLY": True,
            "WEEKLY": True,
            "DAILY": True,
            "4H": True,
            "1H": True,
            "30M": True,
            "15M": True,
            "5M": True,
            "1M": True,
        }

        # User settings override
        if enabled_timeframes:
            self.enabled.update(enabled_timeframes)

        # ----------------------------------------------------
        # COLORS
        # ----------------------------------------------------

        self.colors = {
            "YEARLY": "sky",
            "MONTHLY": "red",
            "WEEKLY": "green",
            "DAILY": "black",
            "4H": "orange",
            "1H": "purple",
            "30M": "darkgreen",
            "15M": "blue",
            "5M": "magenta",
            "1M": "cyan",
        }

    # ========================================================
    # CHANGE SYMBOL
    # ========================================================

    def set_symbol(self, symbol: str):
        self.symbol = symbol.upper()

    # ========================================================
    # ENABLE / DISABLE TIMEFRAME
    # ========================================================

    def set_timeframe(self, timeframe: str, enabled: bool):

        timeframe = timeframe.upper()

        if timeframe not in TIMEFRAME_MAPPING:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        self.enabled[timeframe] = enabled

    # ========================================================
    # CHANGE COLOR
    # ========================================================

    def set_color(self, timeframe: str, color: str):

        timeframe = timeframe.upper()

        if timeframe not in TIMEFRAME_MAPPING:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        self.colors[timeframe] = color

    # ========================================================
    # GET BINANCE CANDLES
    # ========================================================

    def get_klines(
        self,
        interval: str,
        limit: int = 5,
    ):

        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit,
        }

        response = requests.get(
            BINANCE_API,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    # ========================================================
    # GET PREVIOUS COMPLETED CANDLE
    # ========================================================

    def get_previous_candle(
        self,
        interval: str,
    ):

        candles = self.get_klines(
            interval=interval,
            limit=5,
        )

        if len(candles) < 2:
            raise RuntimeError(
                f"Not enough candle data for {interval}"
            )

        # ----------------------------------------------------
        # Binance normally returns the latest candle last.
        # Last candle can still be forming.
        # Therefore we use [-2] = previous completed candle.
        # ----------------------------------------------------

        candle = candles[-2]

        return {
            "time": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }

    # ========================================================
    # CALCULATE TRI LEVELS
    # ========================================================

    def calculate_levels(
        self,
        timeframe: str,
    ) -> TRILineLevels:

        timeframe = timeframe.upper()

        if timeframe not in TIMEFRAME_MAPPING:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        interval = TIMEFRAME_MAPPING[timeframe]

        candle = self.get_previous_candle(interval)

        o = candle["open"]
        h = candle["high"]
        l = candle["low"]
        c = candle["close"]

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

        body_high = max(o, c)
        body_low = min(o, c)

        # ----------------------------------------------------
        # BODY 50%
        # ----------------------------------------------------

        body_50 = (
            body_high + body_low
        ) / 2.0

        # ----------------------------------------------------
        # UPPER WICK 50%
        # ----------------------------------------------------

        upper_50 = (
            h + body_high
        ) / 2.0

        # ----------------------------------------------------
        # LOWER WICK 50%
        # ----------------------------------------------------

        lower_50 = (
            l + body_low
        ) / 2.0

        return TRILineLevels(
            timeframe=timeframe,
            open=o,
            high=h,
            low=l,
            close=c,
            body_high=body_high,
            body_low=body_low,
            body_50=body_50,
            upper_50=upper_50,
            lower_50=lower_50,
            candle_time=candle["time"],
        )

    # ========================================================
    # CALCULATE ALL ENABLED TIMEFRAMES
    # ========================================================

    def calculate_all(self):

        results = {}

        for timeframe in TIMEFRAME_MAPPING:

            if not self.enabled.get(timeframe, False):
                continue

            try:

                levels = self.calculate_levels(
                    timeframe
                )

                results[timeframe] = levels

            except Exception as error:

                results[timeframe] = {
                    "error": str(error)
                }

        return results

    # ========================================================
    # RETURN DICTIONARY
    # ========================================================

    def calculate_all_dict(self):

        results = self.calculate_all()

        output = {}

        for timeframe, value in results.items():

            if isinstance(value, TRILineLevels):

                output[timeframe] = {
                    "color": self.colors[timeframe],
                    **asdict(value),
                }

            else:

                output[timeframe] = value

        return output

    # ========================================================
    # PRINT LEVELS
    # ========================================================

    def print_levels(self):

        results = self.calculate_all()

        print()
        print("=" * 75)
        print(
            f"TRI LINE ANALYSIS | {self.symbol}"
        )
        print("=" * 75)

        for timeframe, value in results.items():

            print()
            print(f"[{timeframe}]")

            if isinstance(value, TRILineLevels):

                print(f"Color       : {self.colors[timeframe]}")
                print(f"Open        : {value.open:.8f}")
                print(f"High        : {value.high:.8f}")
                print(f"Low         : {value.low:.8f}")
                print(f"Close       : {value.close:.8f}")
                print(f"Body 50%    : {value.body_50:.8f}")
                print(f"Upper 50%   : {value.upper_50:.8f}")
                print(f"Lower 50%   : {value.lower_50:.8f}")

            else:

                print(f"ERROR       : {value['error']}")

        print()
        print("=" * 75)


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    engine = TRILineEngine(
        symbol="BTCUSDT"
    )

    # Enable all standard timeframes including new ones (1M, 5M, etc.)
    for tf in TIMEFRAME_MAPPING.keys():
        engine.set_timeframe(tf, True)

    # Run analysis test
    engine.print_levels()
