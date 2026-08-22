import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# TEN PAPER RESEARCH LAB
# ============================================================

class TenPaperResearchLab:

    def __init__(
        self,
        target_vol=0.15,
        min_move_threshold=0.30,
        strong_move_threshold=1.00,
        ofi_threshold=0.10,
        obi_threshold=0.15,
        confidence_threshold=0.60
    ):

        self.target_vol = target_vol

        # ----------------------------------------------------
        # Signal thresholds
        # ----------------------------------------------------

        self.min_move_threshold = min_move_threshold
        self.strong_move_threshold = strong_move_threshold

        self.ofi_threshold = ofi_threshold
        self.obi_threshold = obi_threshold

        self.confidence_threshold = confidence_threshold

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        self.scaler = StandardScaler()

        self.ml_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            learning_rate="optimal",
            max_iter=1000,
            random_state=42
        )

        self.is_model_trained = False

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        self.feature_names = [

            # Microstructure
            "BOOK_IMB",
            "OFI",
            "TAKER_FLOW",

            # Price / volatility
            "MOVE_SIGNIFICANCE",
            "IMPULSE_SCORE",
            "REALIZED_VOL",
            "VOLUME_ACCEL",

            # Statistical
            "QUANTILES",
            "BAYESIAN",

            # Trend / regime
            "ADAPT_CONF",
            "RMT_DOM",

            # Risk / reward
            "REWARD_RISK",
            "FRAC_KELLY",

            # Hawkes-style activity
            "HAWKES"

        ]

        self.dynamic_weights = {
            name: 1.0 / len(self.feature_names)
            for name in self.feature_names
        }

        # ----------------------------------------------------
        # Previous order book
        # ----------------------------------------------------

        self.previous_bids = None
        self.previous_asks = None

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def safe_float(value, default=0.0):

        try:
            value = float(value)

            if np.isfinite(value):
                return value

        except Exception:
            pass

        return default

    # ========================================================
    # ORDER BOOK IMBALANCE
    # ========================================================

    def calculate_book_imbalance(self, bids, asks):

        if len(bids) == 0 or len(asks) == 0:
            return 0.0

        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])

        return float(
            (bid_volume - ask_volume)
            /
            (bid_volume + ask_volume + 1e-8)
        )

    # ========================================================
    # ORDER FLOW IMBALANCE
    # ========================================================

    def calculate_ofi(self, bids, asks):

        if (
            self.previous_bids is None
            or self.previous_asks is None
            or len(bids) == 0
            or len(asks) == 0
        ):
            return 0.0

        try:

            current_bid_price = bids[0, 0]
            current_bid_size = bids[0, 1]

            previous_bid_price = self.previous_bids[0, 0]
            previous_bid_size = self.previous_bids[0, 1]

            current_ask_price = asks[0, 0]
            current_ask_size = asks[0, 1]

            previous_ask_price = self.previous_asks[0, 0]
            previous_ask_size = self.previous_asks[0, 1]

            # Bid contribution
            if current_bid_price > previous_bid_price:

                bid_flow = current_bid_size

            elif current_bid_price < previous_bid_price:

                bid_flow = -previous_bid_size

            else:

                bid_flow = current_bid_size - previous_bid_size

            # Ask contribution
            if current_ask_price < previous_ask_price:

                ask_flow = current_ask_size

            elif current_ask_price > previous_ask_price:

                ask_flow = -previous_ask_size

            else:

                ask_flow = current_ask_size - previous_ask_size

            raw_ofi = bid_flow - ask_flow

            total_depth = (
                current_bid_size
                + current_ask_size
                + 1e-8
            )

            normalized_ofi = raw_ofi / total_depth

            return float(
                np.clip(normalized_ofi, -1.0, 1.0)
            )

        except Exception:

            return 0.0

    # ========================================================
    # HAWKES-STYLE ACTIVITY
    # ========================================================

    def calculate_activity_intensity(self, df):

        if "Volume" not in df.columns or len(df) < 15:
            return 0.0

        try:

            volume = df["Volume"].astype(float)

            short_activity = volume.iloc[-3:].mean()

            long_activity = volume.iloc[-15:].mean()

            ratio = (
                short_activity
                /
                (long_activity + 1e-8)
            )

            # Activity acceleration
            intensity = (ratio - 1.0)

            return float(
                np.clip(intensity, -1.0, 1.0)
            )

        except Exception:

            return 0.0

    # ========================================================
    # MOVE SIGNIFICANCE
    # ========================================================

    def calculate_move_significance(self, returns):

        if len(returns) < 5:
            return 0.0

        try:

            current_move = abs(float(returns.iloc[-1]))

            volatility = float(
                returns.iloc[-20:].std()
            )

            if volatility <= 1e-8:
                return 0.0

            significance = current_move / volatility

            return float(
                np.clip(significance, 0.0, 5.0)
            )

        except Exception:

            return 0.0

    # ========================================================
    # IMPULSE SCORE
    # ========================================================

    def calculate_impulse_score(
        self,
        returns,
        volume_acceleration,
        ofi
    ):

        if len(returns) < 5:
            return 0.0

        try:

            move = abs(float(returns.iloc[-1]))

            volatility = float(
                returns.iloc[-20:].std()
            ) + 1e-8

            normalized_move = move / volatility

            volume_component = np.clip(
                volume_acceleration,
                0.0,
                3.0
            )

            ofi_component = abs(ofi)

            impulse = (
                0.50 * np.clip(
                    normalized_move / 3.0,
                    0.0,
                    1.0
                )
                +
                0.25 * np.clip(
                    volume_component / 3.0,
                    0.0,
                    1.0
                )
                +
                0.25 * ofi_component
            )

            return float(
                np.clip(impulse, 0.0, 1.0)
            )

        except Exception:

            return 0.0

    # ========================================================
    # MAIN FEATURE EXTRACTION
    # ========================================================

    def extract_features(
        self,
        df,
        bids,
        asks
    ):

        results = {
            name: 0.0
            for name in self.feature_names
        }

        if (
            df is None
            or df.empty
            or len(df) < 20
            or len(bids) == 0
            or len(asks) == 0
        ):

            return results

        try:

            close = df["Close"].astype(float)

            volume = df["Volume"].astype(float)

            # ------------------------------------------------
            # Price returns
            # ------------------------------------------------

            returns = close.pct_change().dropna()

            if len(returns) < 5:
                return results

            current_return = float(
                returns.iloc[-1]
            )

            returns_5 = (
                close.iloc[-1]
                /
                (close.iloc[-5] + 1e-8)
                - 1.0
            )

            realized_vol = float(
                returns.iloc[-20:].std()
            ) + 1e-8

            # ------------------------------------------------
            # Order book
            # ------------------------------------------------

            bid_volume = np.sum(bids[:, 1])

            ask_volume = np.sum(asks[:, 1])

            mid_price = (
                bids[0, 0]
                +
                asks[0, 0]
            ) / 2.0

            # ------------------------------------------------
            # OBI
            # ------------------------------------------------

            book_imbalance = (
                bid_volume - ask_volume
            ) / (
                bid_volume
                +
                ask_volume
                +
                1e-8
            )

            results["BOOK_IMB"] = float(
                np.clip(
                    book_imbalance,
                    -1.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # OFI
            # ------------------------------------------------

            ofi = self.calculate_ofi(
                bids,
                asks
            )

            results["OFI"] = ofi

            # ------------------------------------------------
            # TAKER FLOW
            #
            # Approximation only when true aggressor
            # trade data isn't available.
            # ------------------------------------------------

            current_volume = float(
                volume.iloc[-1]
            )

            if current_return > 0:

                taker_flow = 1.0

            elif current_return < 0:

                taker_flow = -1.0

            else:

                taker_flow = 0.0

            results["TAKER_FLOW"] = taker_flow

            # ------------------------------------------------
            # Volume acceleration
            # ------------------------------------------------

            short_volume = float(
                volume.iloc[-3:].mean()
            )

            long_volume = float(
                volume.iloc[-20:].mean()
            ) + 1e-8

            volume_acceleration = (
                short_volume
                /
                long_volume
            )

            results["VOLUME_ACCEL"] = float(
                np.clip(
                    volume_acceleration - 1.0,
                    -1.0,
                    3.0
                )
            )

            # ------------------------------------------------
            # Move significance
            # ------------------------------------------------

            move_significance = (
                self.calculate_move_significance(
                    returns
                )
            )

            results["MOVE_SIGNIFICANCE"] = (
                float(
                    np.clip(
                        move_significance / 3.0,
                        0.0,
                        1.0
                    )
                )
            )

            # ------------------------------------------------
            # Impulse
            # ------------------------------------------------

            results["IMPULSE_SCORE"] = (
                self.calculate_impulse_score(
                    returns,
                    volume_acceleration,
                    ofi
                )
            )

            # ------------------------------------------------
            # Realized volatility
            # ------------------------------------------------

            results["REALIZED_VOL"] = float(
                np.clip(
                    realized_vol / self.target_vol,
                    0.0,
                    5.0
                )
            )

            # ------------------------------------------------
            # Quantiles
            # ------------------------------------------------

            q90 = float(
                returns.iloc[-20:].quantile(
                    0.90
                )
            )

            q10 = float(
                returns.iloc[-20:].quantile(
                    0.10
                )
            )

            denominator = (
                q90 - q10
            ) + 1e-8

            quantile_position = (
                returns_5 - q10
            ) / denominator

            results["QUANTILES"] = float(
                np.clip(
                    quantile_position * 2.0 - 1.0,
                    -1.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # Bayesian-style evidence score
            #
            # Not hardcoded 74.5% anymore.
            # Combines independent directional evidence.
            # ------------------------------------------------

            evidence = np.mean([
                book_imbalance,
                ofi,
                taker_flow
            ])

            results["BAYESIAN"] = float(
                np.clip(
                    evidence,
                    -1.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # Adaptive trend
            # ------------------------------------------------

            fast_ma = float(
                close.rolling(5).mean().iloc[-1]
            )

            slow_ma = float(
                close.rolling(20).mean().iloc[-1]
            )

            trend_strength = (
                fast_ma - slow_ma
            ) / (
                mid_price * realized_vol
                +
                1e-8
            )

            results["ADAPT_CONF"] = float(
                np.clip(
                    trend_strength,
                    -1.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # RMT-style dominance
            # ------------------------------------------------

            dominance = (
                abs(returns_5)
                /
                (
                    realized_vol
                    *
                    np.sqrt(5)
                    +
                    1e-8
                )
            )

            results["RMT_DOM"] = float(
                np.clip(
                    dominance / 3.0
                    *
                    np.sign(returns_5),
                    -1.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # Fractional Kelly
            #
            # Used as risk information, NOT direction.
            # ------------------------------------------------

            estimated_probability = (
                0.50
                +
                0.25
                *
                abs(
                    book_imbalance
                )
            )

            reward_risk = (
                abs(q90)
                /
                (
                    abs(q10)
                    +
                    1e-8
                )
            )

            kelly = (
                estimated_probability
                -
                (
                    (
                        1.0
                        -
                        estimated_probability
                    )
                    /
                    max(
                        reward_risk,
                        1.0
                    )
                )
            )

            results["FRAC_KELLY"] = float(
                np.clip(
                    kelly,
                    -1.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # Reward / Risk
            # ------------------------------------------------

            if reward_risk >= 1.2:

                results["REWARD_RISK"] = 1.0

            elif reward_risk < 0.8:

                results["REWARD_RISK"] = -1.0

            else:

                results["REWARD_RISK"] = 0.0

            # ------------------------------------------------
            # Hawkes-style activity
            # ------------------------------------------------

            activity = (
                self.calculate_activity_intensity(
                    df
                )
            )

            results["HAWKES"] = float(
                np.clip(
                    activity
                    *
                    np.sign(returns_5),
                    -1.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # Save current book
            # ------------------------------------------------

            self.previous_bids = np.copy(bids)

            self.previous_asks = np.copy(asks)

            return results

        except Exception:

            return results

    # ========================================================
    # DIRECTION SCORE
    # ========================================================

    def calculate_direction_score(
        self,
        features
    ):

        directional_features = [

            "BOOK_IMB",
            "OFI",
            "TAKER_FLOW",
            "QUANTILES",
            "BAYESIAN",
            "ADAPT_CONF",
            "RMT_DOM",
            "HAWKES"

        ]

        values = []

        for feature in directional_features:

            values.append(
                features.get(
                    feature,
                    0.0
                )
            )

        return float(
            np.clip(
                np.mean(values),
                -1.0,
                1.0
            )
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def calculate_confidence(
        self,
        features,
        ml_probability=None
    ):

        direction_strength = abs(
            self.calculate_direction_score(
                features
            )
        )

        move_strength = features.get(
            "MOVE_SIGNIFICANCE",
            0.0
        )

        impulse_strength = features.get(
            "IMPULSE_SCORE",
            0.0
        )

        if ml_probability is not None:

            ml_confidence = abs(
                ml_probability - 0.5
            ) * 2.0

        else:

            ml_confidence = 0.0

        confidence = (

            0.35 * direction_strength
            +
            0.25 * move_strength
            +
            0.20 * impulse_strength
            +
            0.20 * ml_confidence

        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

    # ========================================================
    # ML TRAINING
    # ========================================================

    def train_from_history(
        self,
        performance_history
    ):

        if not performance_history:
            return False

        X = []
        y = []

        for record in performance_history:

            features = record.get(
                "features"
            )

            outcome = record.get(
                "outcome"
            )

            # ----------------------------------------------
            # IMPORTANT:
            # Do NOT train on random/fake features.
            # ----------------------------------------------

            if not isinstance(
                features,
                dict
            ):
                continue

            if outcome not in (
                "WIN",
                "LOSS"
            ):
                continue

            row = []

            valid = True

            for feature in self.feature_names:

                value = features.get(
                    feature
                )

                if value is None:
                    valid = False
                    break

                try:

                    value = float(value)

                    if not np.isfinite(value):
                        valid = False
                        break

                except Exception:

                    valid = False
                    break

                row.append(value)

            if not valid:
                continue

            X.append(row)

            y.append(
                1 if outcome == "WIN"
                else 0
            )

        if len(X) < 20:
            return False

        if len(set(y)) < 2:
            return False

        try:

            X = np.asarray(
                X,
                dtype=float
            )

            y = np.asarray(
                y,
                dtype=int
            )

            self.scaler.fit(X)

            X_scaled = self.scaler.transform(
                X
            )

            self.ml_model.fit(
                X_scaled,
                y
            )

            self.is_model_trained = True

            return True

        except Exception:

            return False

    # ========================================================
    # MAIN SIGNAL CALCULATION
    # ========================================================

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=None
    ):

        features = self.extract_features(
            df,
            bids,
            asks
        )

        feature_vector = np.array([
            features[name]
            for name in self.feature_names
        ]).reshape(
            1,
            -1
        )

        # ----------------------------------------------------
        # Train only using REAL historical features
        # ----------------------------------------------------

        if performance_history:

            self.train_from_history(
                performance_history
            )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        direction_score = (
            self.calculate_direction_score(
                features
            )
        )

        # ----------------------------------------------------
        # ML probability
        # ----------------------------------------------------

        ml_probability = None

        if self.is_model_trained:

            try:

                scaled_features = (
                    self.scaler.transform(
                        feature_vector
                    )
                )

                ml_probability = float(
                    self.ml_model.predict_proba(
                        scaled_features
                    )[0][1]
                )

            except Exception:

                ml_probability = None

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = self.calculate_confidence(
            features,
            ml_probability
        )

        # ----------------------------------------------------
        # Move filter
        # ----------------------------------------------------

        move_significance_raw = (
            features["MOVE_SIGNIFICANCE"]
            * 3.0
        )

        # ----------------------------------------------------
        # SMALL MOVE BLOCK
        # ----------------------------------------------------

        if (
            move_significance_raw
            <
            self.min_move_threshold
        ):

            signal = "NO TRADE"

            reason = "SMALL_MOVE"

        else:

            # ------------------------------------------------
            # Direction confirmation
            # ------------------------------------------------

            bullish_confirmation = (
                features["BOOK_IMB"]
                >= self.obi_threshold
                and
                features["OFI"]
                >= self.ofi_threshold
            )

            bearish_confirmation = (
                features["BOOK_IMB"]
                <= -self.obi_threshold
                and
                features["OFI"]
                <= -self.ofi_threshold
            )

            # ------------------------------------------------
            # Strong move
            #
            # Sudden pump/dump is NOT automatically blocked.
            # It still needs directional confirmation.
            # ------------------------------------------------

            if (
                move_significance_raw
                >= self.strong_move_threshold
            ):

                if (
                    bullish_confirmation
                    and
                    direction_score > 0
                ):

                    signal = "BUY"

                    reason = (
                        "STRONG_UP_MOVE"
                    )

                elif (
                    bearish_confirmation
                    and
                    direction_score < 0
                ):

                    signal = "SELL"

                    reason = (
                        "STRONG_DOWN_MOVE"
                    )

                else:

                    signal = "WAIT"

                    reason = (
                        "STRONG_MOVE_NO_CONFIRMATION"
                    )

            # ------------------------------------------------
            # Normal meaningful movement
            # ------------------------------------------------

            else:

                if (
                    bullish_confirmation
                    and
                    direction_score > 0
                    and
                    confidence
                    >= self.confidence_threshold
                ):

                    signal = "BUY"

                    reason = (
                        "NORMAL_BUY_CONFIRMATION"
                    )

                elif (
                    bearish_confirmation
                    and
                    direction_score < 0
                    and
                    confidence
                    >= self.confidence_threshold
                ):

                    signal = "SELL"

                    reason = (
                        "NORMAL_SELL_CONFIRMATION"
                    )

                else:

                    signal = "WAIT"

                    reason = (
                        "INSUFFICIENT_CONFIRMATION"
                    )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        ml_direction = 0.0

        if ml_probability is not None:

            ml_direction = (
                ml_probability - 0.5
            ) * 2.0

        final_score = (
            0.65 * direction_score
            +
            0.35 * ml_direction
        )

        final_score = float(
            np.clip(
                final_score,
                -1.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        diagnostics = {

            "SIGNAL": signal,

            "REASON": reason,

            "FINAL_SCORE": final_score,

            "DIRECTION_SCORE": direction_score,

            "CONFIDENCE": confidence,

            "ML_PROBABILITY": ml_probability,

            "MOVE_SIGNIFICANCE":
                move_significance_raw,

            "IMPULSE_SCORE":
                features["IMPULSE_SCORE"],

            "OBI":
                features["BOOK_IMB"],

            "OFI":
                features["OFI"],

            "REALIZED_VOL":
                features["REALIZED_VOL"],

            "VOLUME_ACCEL":
                features["VOLUME_ACCEL"],

            "MODEL_TRAINED":
                self.is_model_trained,

            "FEATURES":
                features

        }

        return (
            diagnostics,
            final_score,
            self.dynamic_weights
        )


# ============================================================
# POWER TRADING RISK ENGINE
# ============================================================

class PowerTradingRiskEngine:

    def __init__(self):

        self.risk_levels = {
            "LOW": 25,
            "MEDIUM": 50,
            "HIGH": 75
        }

    # ========================================================
    # SAFE NORMALIZATION
    # ========================================================

    @staticmethod
    def safe_ratio(
        numerator,
        denominator
    ):

        return float(
            numerator
            /
            (
                abs(denominator)
                +
                1e-8
            )
        )

    # ========================================================
    # RISK METRICS
    # ========================================================

    def calculate_risk_metrics(
        self,
        liquidation_volumes,
        displayed_vol,
        cancelled_vol,
        time_exists,
        obs_window,
        open_interest,
        leverage,
        volatility
    ):

        # ----------------------------------------------------
        # Liquidation Target Zone
        # ----------------------------------------------------

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

        if len(liquidation_volumes) > 0:

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

        ltz_concentration = (
            max_ltz
            /
            (
                total_ltz
                +
                1e-8
            )
        )

        ltz_score = float(
            np.clip(
                ltz_concentration
                * 100.0,
                0.0,
                100.0
            )
        )

        # ----------------------------------------------------
        # Spoof risk
        # ----------------------------------------------------

        displayed_vol = max(
            float(displayed_vol),
            0.0
        )

        cancelled_vol = max(
            float(cancelled_vol),
            0.0
        )

        spoof_ratio = (
            cancelled_vol
            /
            (
                displayed_vol
                +
                1e-8
            )
        )

        persistence = np.clip(
            float(time_exists)
            /
            (
                float(obs_window)
                +
                1e-8
            ),
            0.0,
            1.0
        )

        spoof_score = float(
            np.clip(
                spoof_ratio
                *
                (
                    1.0
                    -
                    persistence
                )
                *
                100.0,
                0.0,
                100.0
            )
        )

        # ----------------------------------------------------
        # Squeeze risk
        #
        # Raw multiplication can become enormous.
        # Normalize it to a 0-100 score.
        # ----------------------------------------------------

        open_interest = max(
            float(open_interest),
            0.0
        )

        leverage = max(
            float(leverage),
            0.0
        )

        volatility = max(
            float(volatility),
            0.0
        )

        raw_squeeze = (
            total_ltz
            *
            open_interest
            *
            leverage
            *
            volatility
        )

        # Log normalization prevents gigantic values
        squeeze_score = float(
            np.clip(
                np.log1p(
                    raw_squeeze
                )
                * 10.0,
                0.0,
                100.0
            )
        )

        # ----------------------------------------------------
        # Combined market risk
        # ----------------------------------------------------

        market_risk = (
            0.35 * ltz_score
            +
            0.35 * spoof_score
            +
            0.30 * squeeze_score
        )

        market_risk = float(
            np.clip(
                market_risk,
                0.0,
                100.0
            )
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if market_risk >= 75:

            risk_level = "EXTREME"

        elif market_risk >= 50:

            risk_level = "HIGH"

        elif market_risk >= 25:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        return {

            "LTZ_Score":
                ltz_score,

            "Spoof_Score":
                spoof_score,

            "Squeeze_Risk":
                squeeze_score,

            "Market_Risk":
                market_risk,

            "Risk_Level":
                risk_level
        }
