import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


class TenPaperResearchLab:

    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

        # =====================================================
        # MACHINE LEARNING
        # =====================================================
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

        # Same six research features
        self.feature_names = [
            "BOOK_IMB",
            "TAKER_FLOW",
            "QUANT_IMPLY",
            "ADAPT_CONF",
            "BAYESIAN",
            "FOURIER_TREND",
        ]

        # =====================================================
        # MULTI LEVEL OBI
        # =====================================================
        self.obi_level_weights = {
            5: 0.10,
            10: 0.20,
            20: 0.40,
            50: 0.30,
        }

        # =====================================================
        # RESEARCH WEIGHTS
        # =====================================================
        self.dynamic_weights = {
            "BOOK_IMB": 0.25,
            "TAKER_FLOW": 0.20,
            "QUANT_IMPLY": 0.15,
            "ADAPT_CONF": 0.15,
            "BAYESIAN": 0.10,
            "FOURIER_TREND": 0.15,
        }

        # =====================================================
        # FINAL SIGNAL SETTINGS
        # =====================================================
        self.quant_weight = 0.65
        self.ml_weight = 0.35

        self.long_threshold = 0.45
        self.short_threshold = -0.45

        self.last_signal = "NEUTRAL"
        self.cooldown_counter = 0

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            value = float(value)
            if np.isfinite(value):
                return value
        except Exception:
            pass
        return default

    @staticmethod
    def _clip(value, low=-1.0, high=1.0):
        return float(np.clip(value, low, high))

    # =========================================================
    # ATR
    # =========================================================

    def _calculate_atr(self, df, period=14):

        if df is None or df.empty:
            return 0.0

        if not all(
            c in df.columns
            for c in ["High", "Low", "Close"]
        ):
            close = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            returns = close.pct_change()

            vol = returns.rolling(
                period
            ).std().iloc[-1]

            price = self._safe_float(
                close.iloc[-1]
            )

            return max(
                self._safe_float(vol) * price,
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

        previous_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        atr = tr.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period
        ).mean()

        value = atr.iloc[-1]

        if not np.isfinite(value):
            value = tr.tail(period).mean()

        return max(
            self._safe_float(value),
            0.0
        )

    # =========================================================
    # MULTI LEVEL OBI
    # =========================================================

    def _calculate_obi_level(
        self,
        bids,
        asks,
        levels
    ):
        """
        OBI_N =
        (Weighted Bid Volume - Weighted Ask Volume)
        /
        (Weighted Bid Volume + Weighted Ask Volume)
        """

        if len(bids) == 0 or len(asks) == 0:
            return 0.0

        n = min(
            levels,
            len(bids),
            len(asks)
        )

        if n <= 0:
            return 0.0

        bid_total = 0.0
        ask_total = 0.0

        for i in range(n):

            bid_size = max(
                self._safe_float(
                    bids[i][1]
                ),
                0.0
            )

            ask_size = max(
                self._safe_float(
                    asks[i][1]
                ),
                0.0
            )

            # Nearest levels receive greater weight.
            weight = 1.0 / (i + 1.0)

            bid_total += (
                bid_size * weight
            )

            ask_total += (
                ask_size * weight
            )

        denominator = (
            bid_total
            + ask_total
            + 1e-12
        )

        return self._clip(
            (
                bid_total
                - ask_total
            ) / denominator
        )

    def _calculate_multi_level_obi(
        self,
        bids,
        asks
    ):
        """
        OBI_MULTI =
            0.10*OBI5
          + 0.20*OBI10
          + 0.40*OBI20
          + 0.30*OBI50
        """

        values = {}
        weights = {}

        for level, weight in (
            self.obi_level_weights.items()
        ):

            if (
                len(bids) >= level
                and len(asks) >= level
            ):

                values[level] = (
                    self._calculate_obi_level(
                        bids,
                        asks,
                        level
                    )
                )

                weights[level] = weight

        # If fewer than 5 levels are available.
        if not values:

            available = min(
                len(bids),
                len(asks)
            )

            if available <= 0:
                return 0.0

            return self._calculate_obi_level(
                bids,
                asks,
                available
            )

        total_weight = sum(
            weights.values()
        )

        result = 0.0

        for level in values:

            result += (
                values[level]
                * weights[level]
                / total_weight
            )

        return self._clip(result)

    # =========================================================
    # ACTUAL TAKER FLOW
    # =========================================================

    def _calculate_taker_flow(
        self,
        trades
    ):
        """
        Actual trade-flow formula:

        Taker Flow =
        (Aggressive Buy Volume - Aggressive Sell Volume)
        /
        (Aggressive Buy Volume + Aggressive Sell Volume)

        Supported trade formats:

        dict:
            {"side": "buy", "qty": 1.2}

        dict:
            {"side": "sell", "quantity": 1.2}

        DataFrame:
            side + qty/quantity/volume
        """

        if trades is None:
            return 0.0

        try:

            buy_volume = 0.0
            sell_volume = 0.0

            # -------------------------------------------------
            # DATAFRAME
            # -------------------------------------------------

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

                for _, row in trades.tail(500).iterrows():

                    side = str(
                        row[side_col]
                    ).lower()

                    qty = max(
                        self._safe_float(
                            row[qty_col]
                        ),
                        0.0
                    )

                    if side in [
                        "buy",
                        "b",
                        "long",
                        "bid"
                    ]:
                        buy_volume += qty

                    elif side in [
                        "sell",
                        "s",
                        "short",
                        "ask"
                    ]:
                        sell_volume += qty

            # -------------------------------------------------
            # LIST / TUPLE OF DICTS
            # -------------------------------------------------

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
                                    0
                                )
                            )
                        )
                    )

                    qty = max(
                        self._safe_float(qty),
                        0.0
                    )

                    if side in [
                        "buy",
                        "b",
                        "long",
                        "bid"
                    ]:
                        buy_volume += qty

                    elif side in [
                        "sell",
                        "s",
                        "short",
                        "ask"
                    ]:
                        sell_volume += qty

            total = (
                buy_volume
                + sell_volume
                + 1e-12
            )

            return self._clip(
                (
                    buy_volume
                    - sell_volume
                ) / total
            )

        except Exception:
            return 0.0

    # =========================================================
    # MICROPRICE
    # =========================================================

    def _calculate_microprice(
        self,
        bids,
        asks
    ):

        if (
            len(bids) == 0
            or len(asks) == 0
        ):
            return 0.0

        bid = self._safe_float(
            bids[0][0]
        )

        ask = self._safe_float(
            asks[0][0]
        )

        bid_size = max(
            self._safe_float(
                bids[0][1]
            ),
            0.0
        )

        ask_size = max(
            self._safe_float(
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
            + bid * ask_size
        ) / (
            bid_size
            + ask_size
            + 1e-12
        )

        return self._clip(
            np.tanh(
                (
                    microprice
                    - mid
                ) / spread
            )
        )

    # =========================================================
    # BAYESIAN FEATURE
    # =========================================================

    def _calculate_bayesian(
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

                for name in [
                    "outcome",
                    "result",
                    "win",
                    "target",
                    "label"
                ]:

                    if name in performance_history.columns:
                        col = name
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
            -1.0,
            1.0
        )

        strength = (
            0.50
            + 0.45 * abs(obi)
        )

        if obi >= 0:
            likelihood = strength
        else:
            likelihood = 1.0 - strength

        numerator = (
            likelihood * prior
        )

        denominator = (
            numerator
            + (1.0 - likelihood)
            * (1.0 - prior)
        )

        posterior = (
            numerator
            / (denominator + 1e-12)
        )

        return self._clip(
            (
                posterior
                - 0.5
            ) * 2.0
        )

    # =========================================================
    # FOURIER
    # =========================================================

    def _calculate_fourier(
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

        # Recent observations only.
        prices = prices[
            -min(64, len(prices)):
        ]

        mean_price = np.mean(
            prices
        )

        centered = (
            prices - mean_price
        )

        fft_values = np.fft.fft(
            centered
        )

        n = len(
            fft_values
        )

        keep = max(
            2,
            int(n * 0.10)
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
            mean_price * 1e-6,
            1e-12
        )

        return self._clip(
            np.tanh(
                delta / denominator
            )
        )

    # =========================================================
    # FEATURE EXTRACTION
    # =========================================================

    def extract_features(
        self,
        df,
        bids,
        asks,
        trades=None,
        performance_history=None
    ):

        results = {
            key: 0.0
            for key in self.feature_names
        }

        if (
            df is None
            or df.empty
            or len(df) < 15
            or len(bids) == 0
            or len(asks) == 0
        ):
            return results

        if "Close" not in df.columns:
            return results

        try:

            close = pd.to_numeric(
                df["Close"],
                errors="coerce"
            ).ffill().bfill()

            if len(close) < 15:
                return results

            current_price = self._safe_float(
                close.iloc[-1]
            )

            best_bid = self._safe_float(
                bids[0][0],
                current_price
            )

            best_ask = self._safe_float(
                asks[0][0],
                current_price
            )

            mid_price = (
                best_bid
                + best_ask
            ) / 2.0

            atr = self._calculate_atr(
                df,
                14
            )

            atr = max(
                atr,
                current_price * 0.0001,
                1e-12
            )

            # =================================================
            # 1. MULTI LEVEL OBI
            # =================================================

            results["BOOK_IMB"] = (
                self._calculate_multi_level_obi(
                    bids,
                    asks
                )
            )

            # =================================================
            # 2. ACTUAL TAKER FLOW
            # =================================================

            results["TAKER_FLOW"] = (
                self._calculate_taker_flow(
                    trades
                )
            )

            # =================================================
            # 3. MICROPRICE / QUANT IMPLY
            # =================================================

            results["QUANT_IMPLY"] = (
                self._calculate_microprice(
                    bids,
                    asks
                )
            )

            # =================================================
            # 4. EMA20 / EMA50 + ATR
            # =================================================

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
                self._clip(
                    np.tanh(
                        trend / 3.0
                    )
                )
            )

            # =================================================
            # 5. BAYESIAN
            # =================================================

            results["BAYESIAN"] = (
                self._calculate_bayesian(
                    results["BOOK_IMB"],
                    performance_history
                )
            )

            # =================================================
            # 6. FOURIER
            # =================================================

            results["FOURIER_TREND"] = (
                self._calculate_fourier(
                    close.values,
                    atr
                )
            )

            # Final normalization
            for key in self.feature_names:

                results[key] = (
                    self._clip(
                        self._safe_float(
                            results[key]
                        )
                    )
                )

            return results

        except Exception:
            return {
                key: 0.0
                for key in self.feature_names
            }

    # =========================================================
    # ML TRAINING
    # =========================================================

    def train_ml(
        self,
        feature_data,
        labels
    ):
        """
        Train only on historical data.

        feature_data:
            DataFrame containing the six feature columns.

        labels:
            1 = future bullish outcome
            0 = future bearish outcome
        """

        try:

            X = pd.DataFrame(
                feature_data
            )[
                self.feature_names
            ].astype(float)

            y = pd.Series(
                labels
            ).astype(int)

            valid = (
                X.replace(
                    [np.inf, -np.inf],
                    np.nan
                ).notna().all(axis=1)
                & y.notna()
            )

            X = X.loc[valid]
            y = y.loc[valid]

            if len(X) < 50:
                return False

            if y.nunique() < 2:
                return False

            # Fit scaler ONLY on training data.
            X_scaled = (
                self.scaler.fit_transform(
                    X
                )
            )

            self.base_model.fit(
                X_scaled,
                y
            )

            # Calibration needs enough observations.
            self.ml_model = (
                CalibratedClassifierCV(
                    self.base_model,
                    method="sigmoid",
                    cv="prefit"
                )
            )

            # Use a separate calibration subset.
            calibration_size = max(
                20,
                int(len(X_scaled) * 0.20)
            )

            if (
                len(X_scaled)
                >= calibration_size + 30
            ):

                split = (
                    len(X_scaled)
                    - calibration_size
                )

                self.base_model.fit(
                    X_scaled[:split],
                    y.iloc[:split]
                )

                self.ml_model = (
                    CalibratedClassifierCV(
                        self.base_model,
                        method="sigmoid",
                        cv="prefit"
                    )
                )

                self.ml_model.fit(
                    X_scaled[split:],
                    y.iloc[split:]
                )

            else:

                self.ml_model = None

            self.is_model_trained = True

            return True

        except Exception:
            self.is_model_trained = False
            self.ml_model = None
            return False

    # =========================================================
    # ML PREDICTION
    # =========================================================

    def predict_ml(
        self,
        feature_results
    ):

        if (
            not self.is_model_trained
            or self.ml_model is None
        ):
            return 0.50, 0.0

        try:

            X = np.array(
                [
                    feature_results[key]
                    for key in self.feature_names
                ],
                dtype=float
            ).reshape(
                1,
                -1
            )

            X = np.nan_to_num(
                X,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            )

            X_scaled = (
                self.scaler.transform(
                    X
                )
            )

            probability = (
                self.ml_model
                .predict_proba(
                    X_scaled
                )[0]
            )

            classes = (
                self.ml_model
                .classes_
            )

            if 1 in classes:

                idx = list(
                    classes
                ).index(1)

                p_up = float(
                    probability[idx]
                )

            else:

                p_up = 0.50

            ml_score = (
                p_up - 0.50
            ) * 2.0

            return (
                float(
                    np.clip(
                        p_up,
                        0.0,
                        1.0
                    )
                ),
                self._clip(
                    ml_score
                )
            )

        except Exception:
            return 0.50, 0.0

    # =========================================================
    # SL / TP
    # =========================================================

    def calculate_fourier_sl_tp(
        self,
        df,
        current_price,
        signal_intent
    ):

        current_price = self._safe_float(
            current_price
        )

        if current_price <= 0:
            return 0.0, 0.0

        atr = max(
            self._calculate_atr(
                df,
                14
            ),
            current_price * 0.0005
        )

        # Hard maximum risk = 0.6%
        max_risk = (
            current_price * 0.006
        )

        # Base risk around ATR.
        risk_distance = min(
            max(
                atr,
                current_price * 0.0005
            ),
            max_risk
        )

        if signal_intent == "LONG":

            sl = (
                current_price
                - risk_distance
            )

            tp = (
                current_price
                + 2.0
                * risk_distance
            )

        elif signal_intent == "SHORT":

            sl = (
                current_price
                + risk_distance
            )

            tp = (
                current_price
                - 2.0
                * risk_distance
            )

        else:

            sl = (
                current_price
                - risk_distance
            )

            tp = (
                current_price
                + 2.0
                * risk_distance
            )

        return (
            round(float(sl), 4),
            round(float(tp), 4)
        )

    # =========================================================
    # FINAL SIGNAL
    # =========================================================

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=None,
        trades=None
    ):

        # -----------------------------------------------------
        # FEATURES
        # -----------------------------------------------------

        results = self.extract_features(
            df=df,
            bids=bids,
            asks=asks,
            trades=trades,
            performance_history=performance_history
        )

        # -----------------------------------------------------
        # QUANT SCORE
        # -----------------------------------------------------

        feature_vector = np.array(
            [
                results[key]
                for key in self.feature_names
            ],
            dtype=float
        )

        weight_vector = np.array(
            [
                self.dynamic_weights[key]
                for key in self.feature_names
            ],
            dtype=float
        )

        quant_score = float(
            np.dot(
                feature_vector,
                weight_vector
            )
        )

        quant_score = self._clip(
            quant_score
        )

        # -----------------------------------------------------
        # ML SCORE
        # -----------------------------------------------------

        ml_probability, ml_score = (
            self.predict_ml(
                results
            )
        )

        # -----------------------------------------------------
        # FINAL SCORE
        # -----------------------------------------------------

        if self.is_model_trained:

            final_score = (
                self.quant_weight
                * quant_score
                +
                self.ml_weight
                * ml_score
            )

        else:

            # No trained ML model:
            # don't pretend ML is working.
            final_score = quant_score

        final_score = self._clip(
            final_score
        )

        # -----------------------------------------------------
        # SIGNAL
        # -----------------------------------------------------

        current_intent = "NEUTRAL"

        if (
            final_score
            >= self.long_threshold
        ):

            current_intent = "LONG"

        elif (
            final_score
            <= self.short_threshold
        ):

            current_intent = "SHORT"

        # -----------------------------------------------------
        # HYSTERESIS / COOLDOWN
        # -----------------------------------------------------

        if (
            current_intent
            != self.last_signal
        ):

            if self.cooldown_counter > 0:

                self.cooldown_counter -= 1

                current_intent = (
                    self.last_signal
                )

            else:

                self.cooldown_counter = 3

                self.last_signal = (
                    current_intent
                )

        else:

            self.cooldown_counter = 3

        # -----------------------------------------------------
        # PRICE
        # -----------------------------------------------------

        if (
            df is not None
            and not df.empty
            and "Close" in df.columns
        ):

            current_price = (
                self._safe_float(
                    df["Close"].iloc[-1]
                )
            )

        else:

            current_price = 0.0

        # -----------------------------------------------------
        # SL / TP
        # -----------------------------------------------------

        stop_loss, take_profit = (
            self.calculate_fourier_sl_tp(
                df,
                current_price,
                current_intent
            )
        )

        # -----------------------------------------------------
        # PAYLOAD
        # -----------------------------------------------------

        signal_payload = {
            "intent": current_intent,
            "score": final_score,
            "quant_score": quant_score,
            "ml_probability": ml_probability,
            "ml_score": ml_score,
            "ml_trained": self.is_model_trained,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

        return (
            results,
            signal_payload,
            self.dynamic_weights
        )


# =============================================================
# POWER TRADING RISK ENGINE
# =============================================================

class PowerTradingRiskEngine:

    def __init__(self):
        pass

    @staticmethod
    def _safe_float(
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
            self._safe_float(
                displayed_vol
            ),
            0.0
        )

        cancelled_vol = max(
            self._safe_float(
                cancelled_vol
            ),
            0.0
        )

        time_exists = max(
            self._safe_float(
                time_exists
            ),
            0.0
        )

        obs_window = max(
            self._safe_float(
                obs_window
            ),
            1e-8
        )

        open_interest = max(
            self._safe_float(
                open_interest
            ),
            0.0
        )

        leverage = max(
            self._safe_float(
                leverage
            ),
            0.0
        )

        volatility = max(
            self._safe_float(
                volatility
            ),
            0.0
        )

        # =====================================================
        # LIQUIDATION
        # =====================================================

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
                0.0,
                100.0
            )
        )

        # =====================================================
        # SPOOF
        # =====================================================

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
            0.0,
            1.0
        )

        spoof_score = (
            spoof_ratio
            * (1.0 - persistence)
            * 100.0
        )

        spoof_score = float(
            np.clip(
                spoof_score,
                0.0,
                100.0
            )
        )

        # =====================================================
        # SQUEEZE
        # =====================================================

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
            0.0,
            5.0
        )

        volatility_factor = np.clip(
            volatility,
            0.0,
            1.0
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
                0.0,
                100.0
            )
        )

        # =====================================================
        # FINAL MARKET RISK
        # =====================================================

        market_risk = (
            0.40 * ltz_score
            + 0.25 * spoof_score
            + 0.35 * squeeze_risk
        )

        market_risk = float(
            np.clip(
                market_risk,
                0.0,
                100.0
            )
        )

        return {
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze_risk,
            "Market_Risk": market_risk,
        }
