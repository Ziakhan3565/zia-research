from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from dataclasses import dataclass
from typing import Dict, Optional, Any


# ============================================================
# CONFIG
# ============================================================

BINANCE_API = "https://api.binance.com/api/v3/klines"

DEFAULT_SYMBOL = "BTCUSDT"

SUPPORTED_TIMEFRAMES = {
    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
}


# ============================================================
# TRI LINE DATA
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

        self.enabled = {
            "MONTHLY": True,
            "WEEKLY": True,
            "DAILY": True,
            "4H": True,
            "1H": True,
            "30M": True,
            "15M": True,
        }

        if enabled_timeframes:
            self.enabled.update(
                enabled_timeframes
            )

    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    def set_symbol(self, symbol: str):

        self.symbol = symbol.upper()

    # --------------------------------------------------------
    # GET KLINES
    # --------------------------------------------------------

    def get_klines(
        self,
        interval: str,
        limit: int = 5
    ):

        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit,
        }

        response = requests.get(
            BINANCE_API,
            params=params,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()

    # --------------------------------------------------------
    # PREVIOUS COMPLETED CANDLE
    # --------------------------------------------------------

    def get_previous_candle(
        self,
        interval: str
    ):

        candles = self.get_klines(
            interval,
            5
        )

        if len(candles) < 2:
            raise RuntimeError(
                "Not enough candle data"
            )

        candle = candles[-2]

        return {
            "time": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }

    # --------------------------------------------------------
    # TRI FORMULAS
    # --------------------------------------------------------

    def calculate_levels(
        self,
        timeframe: str
    ) -> TRILineLevels:

        timeframe = timeframe.upper()

        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        interval = SUPPORTED_TIMEFRAMES[
            timeframe
        ]

        candle = self.get_previous_candle(
            interval
        )

        o = candle["open"]
        h = candle["high"]
        l = candle["low"]
        c = candle["close"]

        body_high = max(o, c)
        body_low = min(o, c)

        # BODY 50%
        body_50 = (
            body_high
            + body_low
        ) / 2.0

        # UPPER WICK 50%
        upper_50 = (
            h
            + body_high
        ) / 2.0

        # LOWER WICK 50%
        lower_50 = (
            l
            + body_low
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

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    def get_current_price(
        self,
        timeframe="15M"
    ):

        interval = SUPPORTED_TIMEFRAMES[
            timeframe.upper()
        ]

        candles = self.get_klines(
            interval,
            1
        )

        return float(
            candles[0][4]
        )

    # --------------------------------------------------------
    # ALL TRI LEVELS
    # --------------------------------------------------------

    def calculate_all(self):

        results = {}

        for timeframe in SUPPORTED_TIMEFRAMES:

            if not self.enabled.get(
                timeframe,
                False
            ):
                continue

            try:

                results[timeframe] = (
                    self.calculate_levels(
                        timeframe
                    )
                )

            except Exception as error:

                results[timeframe] = {
                    "error": str(error)
                }

        return results


# ============================================================
# TEN PAPER RESEARCH LAB
# ============================================================

class TenPaperResearchLab:

    def __init__(
        self,
        target_vol=0.15,
        tri_engine=None
    ):

        self.target_vol = target_vol

        self.tri_engine = (
            tri_engine
            if tri_engine is not None
            else TRILineEngine()
        )

        # ====================================================
        # TRADE MODES
        # ====================================================

        self.trade_modes = {

            "15M": {
                "confirmation": "15M",
                "reference": ["1H", "4H"],
            },

            "1H": {
                "confirmation": "1H",
                "reference": ["DAILY", "WEEKLY"],
            },

            "4H": {
                "confirmation": "4H",
                "reference": ["WEEKLY", "MONTHLY"],
            },

        }

        # ====================================================
        # TRI TOUCH
        # ====================================================

        self.tri_touch_tolerance = 0.0015

        # ====================================================
        # ML
        # ====================================================

        self.scaler = StandardScaler()

        self.base_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )

        self.ml_model = None

        self.is_model_trained = False

        # ====================================================
        # SIX RESEARCH FEATURES
        # ====================================================

        self.feature_names = [

            "BOOK_IMB",
            "TAKER_FLOW",
            "QUANT_IMPLY",
            "ADAPT_CONF",
            "BAYESIAN",
            "FOURIER_TREND",

        ]

        # ====================================================
        # WEIGHTS
        # ====================================================

        self.dynamic_weights = {

            "BOOK_IMB": 0.25,

            "TAKER_FLOW": 0.20,

            "QUANT_IMPLY": 0.15,

            "ADAPT_CONF": 0.15,

            "BAYESIAN": 0.10,

            "FOURIER_TREND": 0.15,

        }

        self.quant_weight = 0.65
        self.ml_weight = 0.35

        self.long_threshold = 0.45
        self.short_threshold = -0.45

        # ====================================================
        # HYSTERESIS
        # ====================================================

        self.last_signal = "NEUTRAL"
        self.cooldown_counter = 0

    # ========================================================
    # SAFE FLOAT
    # ========================================================

    @staticmethod
    def safe_float(
        value,
        default=0.0
    ):

        try:

            value = float(value)

            if np.isfinite(value):
                return value

        except Exception:
            pass

        return default

    # ========================================================
    # CLIP
    # ========================================================

    @staticmethod
    def clip(
        value,
        low=-1.0,
        high=1.0
    ):

        return float(
            np.clip(
                value,
                low,
                high
            )
        )

    # ========================================================
    # ATR
    # ========================================================

    def calculate_atr(
        self,
        df,
        period=14
    ):

        if df is None or df.empty:
            return 0.0

        if not all(
            col in df.columns
            for col in [
                "High",
                "Low",
                "Close"
            ]
        ):

            close = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            vol = (
                close
                .pct_change()
                .rolling(period)
                .std()
                .iloc[-1]
            )

            price = self.safe_float(
                close.iloc[-1]
            )

            return max(
                self.safe_float(vol)
                * price,
                price * 0.001
            )

        high = pd.to_numeric(
            df["High"],
            errors="coerce"
        )

        low = pd.to_numeric(
            df["Low"],
            errors="coerce"
        )

        close = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1
        ).max(axis=1)

        atr = (
            tr
            .ewm(
                alpha=1.0 / period,
                adjust=False,
                min_periods=period
            )
            .mean()
        )

        value = atr.iloc[-1]

        if not np.isfinite(value):

            value = (
                tr
                .tail(period)
                .mean()
            )

        return max(
            self.safe_float(value),
            0.0
        )

    # ========================================================
    # OBI SINGLE LEVEL
    # ========================================================

    def calculate_obi_level(
        self,
        bids,
        asks,
        levels
    ):

        if (
            len(bids) == 0
            or len(asks) == 0
        ):
            return 0.0

        n = min(
            levels,
            len(bids),
            len(asks)
        )

        bid_sum = 0.0
        ask_sum = 0.0

        for i in range(n):

            bid_size = max(
                self.safe_float(
                    bids[i][1]
                ),
                0.0
            )

            ask_size = max(
                self.safe_float(
                    asks[i][1]
                ),
                0.0
            )

            # Nearer levels receive higher weight.
            weight = 1.0 / (
                i + 1.0
            )

            bid_sum += (
                bid_size * weight
            )

            ask_sum += (
                ask_size * weight
            )

        return self.clip(
            (
                bid_sum
                - ask_sum
            )
            /
            (
                bid_sum
                + ask_sum
                + 1e-12
            )
        )

    # ========================================================
    # MULTI LEVEL OBI
    # ========================================================

    def calculate_multi_level_obi(
        self,
        bids,
        asks
    ):

        levels = {
            5: 0.10,
            10: 0.20,
            20: 0.40,
            50: 0.30,
        }

        values = {}

        weights = {}

        for level, weight in levels.items():

            if (
                len(bids) >= level
                and len(asks) >= level
            ):

                values[level] = (
                    self.calculate_obi_level(
                        bids,
                        asks,
                        level
                    )
                )

                weights[level] = weight

        if not values:

            available = min(
                len(bids),
                len(asks)
            )

            if available <= 0:
                return 0.0

            return self.calculate_obi_level(
                bids,
                asks,
                available
            )

        total_weight = sum(
            weights.values()
        )

        final = 0.0

        for level, value in values.items():

            final += (
                value
                * weights[level]
                / total_weight
            )

        return self.clip(
            final
        )

    # ========================================================
    # ACTUAL TAKER FLOW
    # ========================================================

    def calculate_taker_flow(
        self,
        trades
    ):

        if trades is None:
            return 0.0

        buy_volume = 0.0
        sell_volume = 0.0

        try:

            # ------------------------------------------------
            # DATAFRAME
            # ------------------------------------------------

            if isinstance(
                trades,
                pd.DataFrame
            ):

                if trades.empty:
                    return 0.0

                side_col = None

                for col in [
                    "side",
                    "Side",
                    "aggressor_side",
                    "taker_side"
                ]:

                    if col in trades.columns:

                        side_col = col
                        break

                qty_col = None

                for col in [
                    "qty",
                    "quantity",
                    "volume",
                    "size"
                ]:

                    if col in trades.columns:

                        qty_col = col
                        break

                if (
                    side_col is None
                    or qty_col is None
                ):
                    return 0.0

                rows = trades.tail(
                    500
                )

                for _, row in rows.iterrows():

                    side = str(
                        row[side_col]
                    ).lower()

                    qty = max(
                        self.safe_float(
                            row[qty_col]
                        ),
                        0.0
                    )

                    if side in [
                        "buy",
                        "b",
                        "long"
                    ]:

                        buy_volume += qty

                    elif side in [
                        "sell",
                        "s",
                        "short"
                    ]:

                        sell_volume += qty

            # ------------------------------------------------
            # LIST OF DICTS
            # ------------------------------------------------

            elif isinstance(
                trades,
                (list, tuple)
            ):

                for trade in trades[-500:]:

                    if not isinstance(
                        trade,
                        dict
                    ):
                        continue

                    side = str(
                        trade.get(
                            "side",
                            trade.get(
                                "aggressor_side",
                                ""
                            )
                        )
                    ).lower()

                    qty = trade.get(
                        "qty",
                        trade.get(
                            "quantity",
                            trade.get(
                                "volume",
                                trade.get(
                                    "size",
                                    0.0
                                )
                            )
                        )
                    )

                    qty = max(
                        self.safe_float(qty),
                        0.0
                    )

                    if side in [
                        "buy",
                        "b",
                        "long"
                    ]:

                        buy_volume += qty

                    elif side in [
                        "sell",
                        "s",
                        "short"
                    ]:

                        sell_volume += qty

            total = (
                buy_volume
                + sell_volume
                + 1e-12
            )

            return self.clip(
                (
                    buy_volume
                    - sell_volume
                )
                / total
            )

        except Exception:

            return 0.0

    # ========================================================
    # MICROPRICE
    # ========================================================

    def calculate_microprice(
        self,
        bids,
        asks
    ):

        if (
            len(bids) == 0
            or len(asks) == 0
        ):
            return 0.0

        bid = self.safe_float(
            bids[0][0]
        )

        ask = self.safe_float(
            asks[0][0]
        )

        bid_size = max(
            self.safe_float(
                bids[0][1]
            ),
            0.0
        )

        ask_size = max(
            self.safe_float(
                asks[0][1]
            ),
            0.0
        )

        mid = (
            bid + ask
        ) / 2.0

        spread = max(
            ask - bid,
            1e-12
        )

        microprice = (

            ask * bid_size
            +
            bid * ask_size

        ) / (

            bid_size
            + ask_size
            + 1e-12

        )

        return self.clip(
            np.tanh(
                (
                    microprice
                    - mid
                )
                / spread
            )
        )

    # ========================================================
    # BAYESIAN
    # ========================================================

    def calculate_bayesian(
        self,
        book_imbalance,
        performance_history=None
    ):

        prior = 0.50

        try:

            if isinstance(
                performance_history,
                pd.DataFrame
            ):

                col = None

                for candidate in [
                    "outcome",
                    "result",
                    "win",
                    "target",
                    "label"
                ]:

                    if candidate in (
                        performance_history.columns
                    ):

                        col = candidate
                        break

                if col:

                    values = pd.to_numeric(
                        performance_history[col],
                        errors="coerce"
                    ).dropna()

                    if len(values) >= 10:

                        values = values.clip(
                            0,
                            1
                        )

                        prior = (

                            values.sum()
                            + 1.0

                        ) / (

                            len(values)
                            + 2.0

                        )

        except Exception:

            prior = 0.50

        prior = np.clip(
            prior,
            0.05,
            0.95
        )

        obi = np.clip(
            book_imbalance,
            -1,
            1
        )

        strength = (
            0.50
            + 0.45
            * abs(obi)
        )

        if obi >= 0:

            likelihood = strength

        else:

            likelihood = (
                1.0
                - strength
            )

        numerator = (
            likelihood
            * prior
        )

        denominator = (

            numerator
            +
            (
                1.0
                - likelihood
            )
            *
            (
                1.0
                - prior
            )

        )

        posterior = (
            numerator
            /
            (
                denominator
                + 1e-12
            )
        )

        return self.clip(
            (
                posterior
                - 0.5
            )
            * 2.0
        )

    # ========================================================
    # FOURIER TREND
    # ========================================================

    def calculate_fourier(
        self,
        close,
        atr
    ):

        prices = np.asarray(
            close,
            dtype=float
        )

        prices = prices[
            np.isfinite(prices)
        ]

        if len(prices) < 16:
            return 0.0

        prices = prices[
            -min(
                64,
                len(prices)
            ):
        ]

        centered = (
            prices
            - np.mean(prices)
        )

        fft_values = np.fft.fft(
            centered
        )

        n = len(
            fft_values
        )

        keep = max(
            2,
            int(
                n * 0.10
            )
        )

        mask = np.zeros(
            n,
            dtype=bool
        )

        mask[:keep] = True
        mask[-keep:] = True

        filtered = np.where(
            mask,
            fft_values,
            0
        )

        curve = np.real(
            np.fft.ifft(
                filtered
            )
        )

        delta = (
            curve[-1]
            - curve[-2]
        )

        denominator = max(
            atr,
            np.mean(prices)
            * 1e-6,
            1e-12
        )

        return self.clip(
            np.tanh(
                delta
                / denominator
            )
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(
        self,
        df,
        bids,
        asks,
        trades=None,
        performance_history=None
    ):

        results = {
            feature: 0.0
            for feature in self.feature_names
        }

        if (
            df is None
            or df.empty
            or len(df) < 15
            or len(bids) == 0
            or len(asks) == 0
        ):

            return results

        try:

            close = pd.to_numeric(
                df["Close"],
                errors="coerce"
            ).ffill().bfill()

            current_price = self.safe_float(
                close.iloc[-1]
            )

            atr = self.calculate_atr(
                df
            )

            atr = max(
                atr,
                current_price * 0.0001,
                1e-12
            )

            # ------------------------------------------------
            # BOOK IMBALANCE
            # ------------------------------------------------

            results["BOOK_IMB"] = (
                self.calculate_multi_level_obi(
                    bids,
                    asks
                )
            )

            # ------------------------------------------------
            # ACTUAL TAKER FLOW
            # ------------------------------------------------

            results["TAKER_FLOW"] = (
                self.calculate_taker_flow(
                    trades
                )
            )

            # ------------------------------------------------
            # MICROPRICE
            # ------------------------------------------------

            results["QUANT_IMPLY"] = (
                self.calculate_microprice(
                    bids,
                    asks
                )
            )

            # ------------------------------------------------
            # ADAPTIVE TREND
            # ------------------------------------------------

            ema20 = (
                close
                .ewm(
                    span=20,
                    adjust=False
                )
                .mean()
            )

            ema50 = (
                close
                .ewm(
                    span=50,
                    adjust=False
                )
                .mean()
            )

            trend = (
                ema20.iloc[-1]
                - ema50.iloc[-1]
            ) / atr

            results["ADAPT_CONF"] = (
                self.clip(
                    np.tanh(
                        trend / 3.0
                    )
                )
            )

            # ------------------------------------------------
            # BAYESIAN
            # ------------------------------------------------

            results["BAYESIAN"] = (
                self.calculate_bayesian(
                    results["BOOK_IMB"],
                    performance_history
                )
            )

            # ------------------------------------------------
            # FOURIER
            # ------------------------------------------------

            results["FOURIER_TREND"] = (
                self.calculate_fourier(
                    close.values,
                    atr
                )
            )

            for feature in self.feature_names:

                results[feature] = (
                    self.clip(
                        self.safe_float(
                            results[feature]
                        )
                    )
                )

            return results

        except Exception:

            return {
                feature: 0.0
                for feature in self.feature_names
            }

    # ========================================================
    # ML TRAINING
    # ========================================================

    def train_ml(
        self,
        feature_data,
        labels
    ):

        try:

            X = pd.DataFrame(
                feature_data
            )[
                self.feature_names
            ].astype(float)

            y = pd.Series(
                labels
            ).astype(int)

            if len(X) != len(y):
                return False

            X = X.replace(
                [np.inf, -np.inf],
                np.nan
            )

            valid = X.notna().all(
                axis=1
            )

            X = X.loc[valid]
            y = y.loc[valid]

            if len(X) < 50:
                return False

            if y.nunique() < 2:
                return False

            # Chronological split
            split = int(
                len(X) * 0.80
            )

            X_train = X.iloc[
                :split
            ]

            y_train = y.iloc[
                :split
            ]

            X_cal = X.iloc[
                split:
            ]

            y_cal = y.iloc[
                split:
            ]

            if (
                y_train.nunique() < 2
                or y_cal.nunique() < 2
            ):

                return False

            X_train_scaled = (
                self.scaler.fit_transform(
                    X_train
                )
            )

            X_cal_scaled = (
                self.scaler.transform(
                    X_cal
                )
            )

            self.base_model.fit(
                X_train_scaled,
                y_train
            )

            self.ml_model = (
                CalibratedClassifierCV(
                    self.base_model,
                    method="sigmoid",
                    cv="prefit"
                )
            )

            self.ml_model.fit(
                X_cal_scaled,
                y_cal
            )

            self.is_model_trained = True

            return True

        except Exception:

            self.is_model_trained = False
            self.ml_model = None

            return False

    # ========================================================
    # ML PREDICTION
    # ========================================================

    def predict_ml(
        self,
        features
    ):

        if (
            not self.is_model_trained
            or self.ml_model is None
        ):

            return (
                0.50,
                0.0
            )

        try:

            X = np.array(
                [
                    features[
                        name
                    ]
                    for name in
                    self.feature_names
                ],
                dtype=float
            ).reshape(
                1,
                -1
            )

            X = np.nan_to_num(
                X
            )

            X_scaled = (
                self.scaler.transform(
                    X
                )
            )

            probabilities = (
                self.ml_model
                .predict_proba(
                    X_scaled
                )[0]
            )

            classes = list(
                self.ml_model.classes_
            )

            if 1 not in classes:

                return (
                    0.50,
                    0.0
                )

            index = classes.index(
                1
            )

            probability_up = float(
                probabilities[index]
            )

            ml_score = (
                probability_up
                - 0.50
            ) * 2.0

            return (

                float(
                    np.clip(
                        probability_up,
                        0,
                        1
                    )
                ),

                self.clip(
                    ml_score
                )

            )

        except Exception:

            return (
                0.50,
                0.0
            )

    # ========================================================
    # TRI LEVELS
    # ========================================================

    def get_tri_levels(
        self,
        timeframes
    ):

        result = {}

        for timeframe in timeframes:

            try:

                result[timeframe] = (
                    self.tri_engine
                    .calculate_levels(
                        timeframe
                    )
                )

            except Exception:

                result[timeframe] = None

        return result

    # ========================================================
    # ALL LINES
    # ========================================================

    def get_line_prices(
        self,
        levels
    ):

        prices = []

        for timeframe, data in (
            levels.items()
        ):

            if data is None:
                continue

            prices.extend(
                [
                    float(data.body_50),
                    float(data.upper_50),
                    float(data.lower_50),
                ]
            )

        return sorted(
            set(
                p for p in prices
                if p > 0
            )
        )

    # ========================================================
    # LINE TOUCH
    # ========================================================

    def line_touch(
        self,
        price,
        line
    ):

        if line <= 0:
            return False

        distance = (
            abs(
                price - line
            )
            /
            line
        )

        return (
            distance
            <= self.tri_touch_tolerance
        )

    # ========================================================
    # NEXT TRI LINE
    # ========================================================

    def next_tri_line(
        self,
        price,
        direction,
        levels
    ):

        lines = self.get_line_prices(
            levels
        )

        if direction == "LONG":

            higher = [
                x for x in lines
                if x > price
            ]

            if higher:
                return min(
                    higher
                )

        elif direction == "SHORT":

            lower = [
                x for x in lines
                if x < price
            ]

            if lower:
                return max(
                    lower
                )

        return None

    # ========================================================
    # TRADE MODE TRI SETUP
    # ========================================================

    def calculate_tri_setup(
        self,
        trade_mode,
        current_price,
        confirmation_score,
        ml_probability,
        df
    ):

        result = {

            "trade_mode":
                trade_mode,

            "tri_signal":
                "NEUTRAL",

            "tri_touched":
                False,

            "tri_timeframe":
                None,

            "tri_line":
                None,

            "next_tri_target":
                None,

            "tri_stop_loss":
                None,

            "tri_rr":
                0.0,

            "tri_reason":
                "NO_SETUP",

        }

        if trade_mode not in (
            self.trade_modes
        ):

            return result

        reference_timeframes = (
            self.trade_modes[
                trade_mode
            ]["reference"]
        )

        levels = self.get_tri_levels(
            reference_timeframes
        )

        # ----------------------------------------------------
        # FIND NEAREST REFERENCE LINE
        # ----------------------------------------------------

        closest = None
        closest_distance = float(
            "inf"
        )

        for timeframe, data in (
            levels.items()
        ):

            if data is None:
                continue

            line_map = {

                "BODY_50":
                    data.body_50,

                "UPPER_50":
                    data.upper_50,

                "LOWER_50":
                    data.lower_50,

            }

            for line_name, line_price in (
                line_map.items()
            ):

                if line_price <= 0:
                    continue

                distance = (
                    abs(
                        current_price
                        - line_price
                    )
                    /
                    line_price
                )

                if distance < closest_distance:

                    closest_distance = distance

                    closest = (
                        timeframe,
                        line_name,
                        line_price
                    )

        if closest is None:

            result[
                "tri_reason"
            ] = "NO_REFERENCE_LINE"

            return result

        timeframe, line_name, line_price = (
            closest
        )

        # ----------------------------------------------------
        # MUST TOUCH REFERENCE LINE
        # ----------------------------------------------------

        if not self.line_touch(
            current_price,
            line_price
        ):

            result[
                "tri_reason"
            ] = (
                "WAITING_FOR_"
                + timeframe
                + "_LINE"
            )

            return result

        # ----------------------------------------------------
        # CONFIRMATION
        # ----------------------------------------------------

        if (
            confirmation_score >= 0.25
            and ml_probability >= 0.55
        ):

            direction = "LONG"

        elif (
            confirmation_score <= -0.25
            and ml_probability <= 0.45
        ):

            direction = "SHORT"

        else:

            result[
                "tri_reason"
            ] = "CONFIRMATION_FAILED"

            return result

        # ----------------------------------------------------
        # NEXT TRI TARGET
        # ----------------------------------------------------

        target = self.next_tri_line(
            current_price,
            direction,
            levels
        )

        if target is None:

            result[
                "tri_reason"
            ] = "NO_NEXT_TRI_LINE"

            return result

        # ----------------------------------------------------
        # ATR STOP
        # ----------------------------------------------------

        atr = max(
            self.calculate_atr(
                df,
                14
            ),
            current_price
            * 0.0005
        )

        if direction == "LONG":

            stop = (
                current_price
                - atr
            )

            reward = (
                target
                - current_price
            )

            risk = (
                current_price
                - stop
            )

        else:

            stop = (
                current_price
                + atr
            )

            reward = (
                current_price
                - target
            )

            risk = (
                stop
                - current_price
            )

        if risk <= 0:
            return result

        rr = reward / risk

        # Minimum 1:2
        if rr < 2.0:

            result[
                "tri_reason"
            ] = "RR_BELOW_1_TO_2"

            return result

        result.update({

            "tri_signal":
                direction,

            "tri_touched":
                True,

            "tri_timeframe":
                timeframe,

            "tri_line":
                float(line_price),

            "next_tri_target":
                float(target),

            "tri_stop_loss":
                float(stop),

            "tri_rr":
                float(rr),

            "tri_reason":
                (
                    "TRI_LINE_TOUCH_"
                    "CONFIRMED"
                ),

        })

        return result

    # ========================================================
    # SL / TP
    # ========================================================

    def calculate_sl_tp(
        self,
        df,
        current_price,
        direction,
        tri_setup
    ):

        atr = max(
            self.calculate_atr(
                df,
                14
            ),
            current_price
            * 0.0005
        )

        max_risk = (
            current_price
            * 0.006
        )

        risk = min(
            atr,
            max_risk
        )

        # TRI target first
        if (
            tri_setup
            and tri_setup.get(
                "tri_signal"
            ) == direction
            and tri_setup.get(
                "next_tri_target"
            ) is not None
        ):

            stop = tri_setup[
                "tri_stop_loss"
            ]

            target = tri_setup[
                "next_tri_target"
            ]

            if direction == "LONG":

                stop = max(
                    stop,
                    current_price
                    - max_risk
                )

            elif direction == "SHORT":

                stop = min(
                    stop,
                    current_price
                    + max_risk
                )

            return (
                round(
                    float(stop),
                    4
                ),
                round(
                    float(target),
                    4
                )
            )

        # ATR fallback
        if direction == "LONG":

            stop = (
                current_price
                - risk
            )

            target = (
                current_price
                + 2.0 * risk
            )

        elif direction == "SHORT":

            stop = (
                current_price
                + risk
            )

            target = (
                current_price
                - 2.0 * risk
            )

        else:

            stop = (
                current_price
                - risk
            )

            target = (
                current_price
                + 2.0 * risk
            )

        return (
            round(
                float(stop),
                4
            ),
            round(
                float(target),
                4
            )
        )

    # ========================================================
    # FINAL SIGNAL ENGINE
    # ========================================================

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=None,
        trades=None,
        trade_mode="15M"
    ):

        trade_mode = str(
            trade_mode
        ).upper()

        if trade_mode not in (
            self.trade_modes
        ):

            trade_mode = "15M"

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        features = (
            self.extract_features(
                df,
                bids,
                asks,
                trades,
                performance_history
            )
        )

        # ----------------------------------------------------
        # QUANT SCORE
        # ----------------------------------------------------

        vector = np.array(
            [
                features[name]
                for name in
                self.feature_names
            ],
            dtype=float
        )

        weights = np.array(
            [
                self.dynamic_weights[name]
                for name in
                self.feature_names
            ],
            dtype=float
        )

        quant_score = float(
            np.dot(
                vector,
                weights
            )
        )

        quant_score = self.clip(
            quant_score
        )

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        (
            ml_probability,
            ml_score
        ) = self.predict_ml(
            features
        )

        # ----------------------------------------------------
        # FINAL QUANT + ML
        # ----------------------------------------------------

        if self.is_model_trained:

            final_score = (

                self.quant_weight
                * quant_score

                +

                self.ml_weight
                * ml_score

            )

        else:

            final_score = (
                quant_score
            )

        final_score = self.clip(
            final_score
        )

        # ----------------------------------------------------
        # BASIC DIRECTION
        # ----------------------------------------------------

        if (
            final_score
            >= self.long_threshold
        ):

            intent = "LONG"

        elif (
            final_score
            <= self.short_threshold
        ):

            intent = "SHORT"

        else:

            intent = "NEUTRAL"

        # ----------------------------------------------------
        # CURRENT PRICE
        # ----------------------------------------------------

        current_price = 0.0

        if (
            df is not None
            and not df.empty
        ):

            current_price = (
                self.safe_float(
                    df["Close"].iloc[-1]
                )
            )

        # ----------------------------------------------------
        # TRI SETUP
        # ----------------------------------------------------

        tri_setup = (
            self.calculate_tri_setup(
                trade_mode,
                current_price,
                quant_score,
                ml_probability,
                df
            )
        )

        # ----------------------------------------------------
        # TRI CONFIRMATION
        # ----------------------------------------------------

        if (
            tri_setup[
                "tri_signal"
            ] in [
                "LONG",
                "SHORT"
            ]
        ):

            tri_direction = (
                tri_setup[
                    "tri_signal"
                ]
            )

            if intent == "NEUTRAL":

                intent = (
                    tri_direction
                )

            elif intent != tri_direction:

                intent = "NEUTRAL"

        # ----------------------------------------------------
        # HYSTERESIS
        # ----------------------------------------------------

        if intent != self.last_signal:

            if self.cooldown_counter > 0:

                self.cooldown_counter -= 1

                intent = (
                    self.last_signal
                )

            else:

                self.cooldown_counter = 3

                self.last_signal = (
                    intent
                )

        else:

            self.cooldown_counter = 3

        # ----------------------------------------------------
        # SL / TP
        # ----------------------------------------------------

        stop_loss, take_profit = (
            self.calculate_sl_tp(
                df,
                current_price,
                intent,
                tri_setup
            )
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        payload = {

            "intent":
                intent,

            "score":
                final_score,

            "quant_score":
                quant_score,

            "ml_probability":
                ml_probability,

            "ml_score":
                ml_score,

            "ml_trained":
                self.is_model_trained,

            "current_price":
                current_price,

            "stop_loss":
                stop_loss,

            "take_profit":
                take_profit,

            "trade_mode":
                trade_mode,

            # TRI
            "tri_signal":
                tri_setup[
                    "tri_signal"
                ],

            "tri_touched":
                tri_setup[
                    "tri_touched"
                ],

            "tri_timeframe":
                tri_setup[
                    "tri_timeframe"
                ],

            "tri_line":
                tri_setup[
                    "tri_line"
                ],

            "next_tri_target":
                tri_setup[
                    "next_tri_target"
                ],

            "tri_stop_loss":
                tri_setup[
                    "tri_stop_loss"
                ],

            "tri_rr":
                tri_setup[
                    "tri_rr"
                ],

            "tri_reason":
                tri_setup[
                    "tri_reason"
                ],

        }

        return (
            features,
            payload,
            self.dynamic_weights
        )


# ============================================================
# POWER TRADING RISK ENGINE
# ============================================================

class PowerTradingRiskEngine:

    def __init__(self):
        pass

    @staticmethod
    def safe_float(
        value,
        default=0.0
    ):

        try:

            value = float(value)

            if np.isfinite(value):
                return value

        except Exception:
            pass

        return default

    def calculate_risk_metrics(
        self,
        liquidation_volumes,
        displayed_vol,
        cancelled_vol,
        time_exists,
        obs_window,
        open_interest,
        leverage,
        volatility,
    ):

        liquidation_volumes = np.asarray(
            liquidation_volumes,
            dtype=float
        )

        liquidation_volumes = (
            liquidation_volumes[
                np.isfinite(
                    liquidation_volumes
                )
            ]
        )

        displayed_vol = max(
            self.safe_float(
                displayed_vol
            ),
            0.0
        )

        cancelled_vol = max(
            self.safe_float(
                cancelled_vol
            ),
            0.0
        )

        time_exists = max(
            self.safe_float(
                time_exists
            ),
            0.0
        )

        obs_window = max(
            self.safe_float(
                obs_window
            ),
            1e-8
        )

        open_interest = max(
            self.safe_float(
                open_interest
            ),
            0.0
        )

        leverage = max(
            self.safe_float(
                leverage
            ),
            0.0
        )

        volatility = max(
            self.safe_float(
                volatility
            ),
            0.0
        )

        # ----------------------------------------------------
        # LIQUIDATION
        # ----------------------------------------------------

        if len(
            liquidation_volumes
        ):

            total_ltz = float(
                np.sum(
                    liquidation_volumes
                )
            )

            max_ltz = float(
                np.max(
                    liquidation_volumes
                )
            )

        else:

            total_ltz = 0.0
            max_ltz = 0.0

        ltz_score = (
            max_ltz
            /
            (
                total_ltz
                + 1e-12
            )
        ) * 100.0

        ltz_score = float(
            np.clip(
                ltz_score,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # SPOOF
        # ----------------------------------------------------

        spoof_ratio = (
            cancelled_vol
            /
            (
                displayed_vol
                + 1e-12
            )
        )

        persistence = np.clip(
            time_exists
            /
            obs_window,
            0,
            1
        )

        spoof_score = (
            spoof_ratio
            * (
                1
                - persistence
            )
            * 100.0
        )

        spoof_score = float(
            np.clip(
                spoof_score,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # SQUEEZE
        # ----------------------------------------------------

        liquidation_intensity = (
            total_ltz
            /
            (
                total_ltz
                + displayed_vol
                + 1e-12
            )
        )

        leverage_factor = np.clip(
            leverage / 20.0,
            0,
            5
        )

        volatility_factor = np.clip(
            volatility,
            0,
            1
        )

        squeeze_risk = (
            liquidation_intensity
            * leverage_factor
            * volatility_factor
            * 100.0
        )

        squeeze_risk = float(
            np.clip(
                squeeze_risk,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # FINAL RISK
        # ----------------------------------------------------

        market_risk = (

            0.40
            * ltz_score

            +

            0.25
            * spoof_score

            +

            0.35
            * squeeze_risk

        )

        market_risk = float(
            np.clip(
                market_risk,
                0,
                100
            )
        )

        return {

            "LTZ_Score":
                ltz_score,

            "Spoof_Score":
                spoof_score,

            "Squeeze_Risk":
                squeeze_risk,

            "Market_Risk":
                market_risk,

        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    tri_engine = TRILineEngine(
        symbol="BTCUSDT"
    )

    research_lab = (
        TenPaperResearchLab(
            tri_engine=tri_engine
        )
    )

    risk_engine = (
        PowerTradingRiskEngine()
    )

    print("=" * 70)
    print("RESEARCH LAB INITIALIZED")
    print("=" * 70)

    print()
    print("Features:")

    for feature in (
        research_lab.feature_names
    ):

        print(
            " -",
            feature
        )

    print()
    print("TRI TRADE MODES:")

    print(
        "15M -> 1H + 4H"
    )

    print(
        "1H  -> DAILY + WEEKLY"
    )

    print(
        "4H  -> WEEKLY + MONTHLY"
    )

    print()
    print(
        "ML trained:",
        research_lab.is_model_trained
    )

    print()
    print(
        "System ready."
    )
