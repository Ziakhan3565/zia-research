# ============================================================
# engine.py
# Research / Quant Signal Engine
# ============================================================

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def clip(value, low=-1.0, high=1.0):
    return float(np.clip(value, low, high))


# ============================================================
# TEN PAPER RESEARCH LAB
# ============================================================

class TenPaperResearchLab:

    def __init__(
        self,
        target_vol=0.15,

        # Main move filter
        min_move_percent=0.40,
        move_lookback=5,

        # Order book confirmation
        obi_threshold=0.15,
        ofi_threshold=0.10,

        # Minimum confidence
        confidence_threshold=0.60,

        # Risk / reward
        minimum_rr=2.0,
        strong_rr=3.0,

        # Stop-loss
        atr_period=14,
        atr_multiplier=1.0,

        # Manipulation
        spoof_threshold=0.60
    ):

        self.target_vol = target_vol

        # ====================================================
        # MAIN USER RULES
        # ====================================================

        self.min_move_percent = min_move_percent
        self.move_lookback = move_lookback

        self.minimum_rr = minimum_rr
        self.strong_rr = strong_rr

        # ====================================================
        # SIGNAL SETTINGS
        # ====================================================

        self.obi_threshold = obi_threshold
        self.ofi_threshold = ofi_threshold
        self.confidence_threshold = confidence_threshold

        # ====================================================
        # STOP LOSS SETTINGS
        # ====================================================

        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

        # ====================================================
        # MANIPULATION
        # ====================================================

        self.spoof_threshold = spoof_threshold

        # ====================================================
        # MACHINE LEARNING
        # ====================================================

        self.scaler = StandardScaler()

        self.ml_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )

        self.is_model_trained = False

        # ====================================================
        # FEATURES
        # ====================================================

        self.feature_names = [

            "BOOK_IMB",
            "OFI",
            "TAKER_FLOW",

            "MOVE_PERCENT",
            "MOVE_SIGNIFICANCE",
            "IMPULSE_SCORE",

            "REALIZED_VOL",
            "VOLUME_ACCEL",

            "QUANTILES",
            "BAYESIAN",

            "ADAPT_CONF",
            "RMT_DOM",

            "REWARD_RISK",
            "FRAC_KELLY",

            "HAWKES",

            "SPOOF_RISK"

        ]

        # ====================================================
        # FALLBACK WEIGHTS
        # ====================================================

        self.dynamic_weights = {
            name: 1.0 / len(self.feature_names)
            for name in self.feature_names
        }

        # ====================================================
        # ORDER BOOK MEMORY
        # ====================================================

        self.previous_bids = None
        self.previous_asks = None

    # ========================================================
    # ORDER BOOK IMBALANCE
    # ========================================================

    def calculate_book_imbalance(self, bids, asks):

        if len(bids) == 0 or len(asks) == 0:
            return 0.0

        try:

            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])

            imbalance = (
                bid_volume - ask_volume
            ) / (
                bid_volume + ask_volume + 1e-8
            )

            return clip(
                imbalance,
                -1.0,
                1.0
            )

        except Exception:

            return 0.0

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

            cbp = float(bids[0, 0])
            cbs = float(bids[0, 1])

            pbp = float(self.previous_bids[0, 0])
            pbs = float(self.previous_bids[0, 1])

            cap = float(asks[0, 0])
            cas = float(asks[0, 1])

            pap = float(self.previous_asks[0, 0])
            pas = float(self.previous_asks[0, 1])

            # Bid flow
            if cbp > pbp:
                bid_event = cbs

            elif cbp < pbp:
                bid_event = -pbs

            else:
                bid_event = cbs - pbs

            # Ask flow
            if cap < pap:
                ask_event = cas

            elif cap > pap:
                ask_event = -pas

            else:
                ask_event = cas - pas

            raw_ofi = bid_event - ask_event

            depth = (
                cbs
                + cas
                + 1e-8
            )

            normalized_ofi = (
                raw_ofi / depth
            )

            return clip(
                normalized_ofi,
                -1.0,
                1.0
            )

        except Exception:

            return 0.0

    # ========================================================
    # MOVE PERCENT
    #
    # Last 5 candles by default
    # ========================================================

    def calculate_move_percent(self, close):

        if len(close) <= self.move_lookback:
            return 0.0

        try:

            reference = float(
                close.iloc[-self.move_lookback]
            )

            current = float(
                close.iloc[-1]
            )

            if reference <= 0:
                return 0.0

            move = (
                abs(current - reference)
                /
                reference
            ) * 100.0

            return float(move)

        except Exception:

            return 0.0

    # ========================================================
    # MOVE DIRECTION
    # ========================================================

    def calculate_move_direction(self, close):

        if len(close) <= self.move_lookback:
            return 0.0

        try:

            reference = float(
                close.iloc[-self.move_lookback]
            )

            current = float(
                close.iloc[-1]
            )

            if current > reference:
                return 1.0

            if current < reference:
                return -1.0

            return 0.0

        except Exception:

            return 0.0

    # ========================================================
    # REALIZED VOLATILITY
    # ========================================================

    def calculate_realized_vol(self, returns):

        if len(returns) < 5:
            return 0.0

        try:

            return float(
                returns.iloc[-20:].std()
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

            current_move = abs(
                float(returns.iloc[-1])
            )

            volatility = (
                self.calculate_realized_vol(
                    returns
                )
                + 1e-8
            )

            return float(
                np.clip(
                    current_move / volatility,
                    0.0,
                    5.0
                )
            )

        except Exception:

            return 0.0

    # ========================================================
    # VOLUME ACCELERATION
    # ========================================================

    def calculate_volume_acceleration(
        self,
        volume
    ):

        if len(volume) < 20:
            return 0.0, 0.0

        try:

            short_volume = float(
                volume.iloc[-3:].mean()
            )

            long_volume = float(
                volume.iloc[-20:].mean()
            ) + 1e-8

            ratio = (
                short_volume
                /
                long_volume
            )

            acceleration = ratio - 1.0

            return (
                float(acceleration),
                float(ratio)
            )

        except Exception:

            return 0.0, 0.0

    # ========================================================
    # IMPULSE / SUDDEN MOVE
    # ========================================================

    def calculate_impulse_score(
        self,
        returns,
        volume_ratio,
        ofi
    ):

        if len(returns) < 5:
            return 0.0

        try:

            last_move = abs(
                float(
                    returns.iloc[-1]
                )
            )

            volatility = (
                self.calculate_realized_vol(
                    returns
                )
                + 1e-8
            )

            normalized_move = (
                last_move
                /
                volatility
            )

            price_component = np.clip(
                normalized_move / 3.0,
                0.0,
                1.0
            )

            volume_component = np.clip(
                (volume_ratio - 1.0) / 3.0,
                0.0,
                1.0
            )

            flow_component = np.clip(
                abs(ofi),
                0.0,
                1.0
            )

            score = (
                0.50 * price_component
                +
                0.25 * volume_component
                +
                0.25 * flow_component
            )

            return float(
                np.clip(
                    score,
                    0.0,
                    1.0
                )
            )

        except Exception:

            return 0.0

    # ========================================================
    # HAWKES-STYLE ACTIVITY
    # ========================================================

    def calculate_activity_intensity(
        self,
        df
    ):

        if (
            "Volume" not in df.columns
            or len(df) < 15
        ):
            return 0.0

        try:

            volume = (
                df["Volume"]
                .astype(float)
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
            )

            short_activity = float(
                volume.iloc[-3:].mean()
            )

            long_activity = float(
                volume.iloc[-15:].mean()
            ) + 1e-8

            ratio = (
                short_activity
                /
                long_activity
            )

            return clip(
                ratio - 1.0,
                -1.0,
                1.0
            )

        except Exception:

            return 0.0

    # ========================================================
    # SPOOF / MANIPULATION DETECTION
    #
    # Uses changes between current and previous book.
    # ========================================================

    def calculate_spoof_risk(
        self,
        bids,
        asks
    ):

        if (
            self.previous_bids is None
            or self.previous_asks is None
        ):
            return 0.0

        try:

            current_bid = np.sum(
                bids[:, 1]
            )

            current_ask = np.sum(
                asks[:, 1]
            )

            previous_bid = np.sum(
                self.previous_bids[:, 1]
            )

            previous_ask = np.sum(
                self.previous_asks[:, 1]
            )

            bid_change = (
                current_bid
                -
                previous_bid
            ) / (
                abs(previous_bid)
                +
                1e-8
            )

            ask_change = (
                current_ask
                -
                previous_ask
            ) / (
                abs(previous_ask)
                +
                1e-8
            )

            # Large book disappearance
            bid_cancel = max(
                -bid_change,
                0.0
            )

            ask_cancel = max(
                -ask_change,
                0.0
            )

            risk = max(
                bid_cancel,
                ask_cancel
            )

            return float(
                np.clip(
                    risk,
                    0.0,
                    1.0
                )
            )

        except Exception:

            return 0.0

    # ========================================================
    # ATR
    # ========================================================

    def calculate_atr(self, df):

        required = [
            "High",
            "Low",
            "Close"
        ]

        if not all(
            col in df.columns
            for col in required
        ):
            return 0.0

        if len(df) < self.atr_period + 2:
            return 0.0

        try:

            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            close = df["Close"].astype(float)

            previous_close = (
                close.shift(1)
            )

            tr1 = high - low

            tr2 = (
                high
                -
                previous_close
            ).abs()

            tr3 = (
                low
                -
                previous_close
            ).abs()

            true_range = pd.concat(
                [
                    tr1,
                    tr2,
                    tr3
                ],
                axis=1
            ).max(axis=1)

            atr = float(
                true_range
                .rolling(
                    self.atr_period
                )
                .mean()
                .iloc[-1]
            )

            if not np.isfinite(atr):
                return 0.0

            return atr

        except Exception:

            return 0.0

    # ========================================================
    # DYNAMIC SL
    # ========================================================

    def calculate_stop_loss(
        self,
        df,
        direction,
        entry_price
    ):

        close = df["Close"].astype(float)

        atr = self.calculate_atr(df)

        # ====================================================
        # ATR based risk
        # ====================================================

        if atr > 0:

            volatility_risk = (
                atr
                *
                self.atr_multiplier
            )

        else:

            volatility_risk = (
                entry_price
                *
                0.0015
            )

        # ====================================================
        # Recent structure
        # ====================================================

        structure_lookback = min(
            10,
            len(df)
        )

        recent_low = float(
            close.iloc[
                -structure_lookback:
            ].min()
        )

        recent_high = float(
            close.iloc[
                -structure_lookback:
            ].max()
        )

        if direction == "BUY":

            structure_risk = (
                entry_price
                -
                recent_low
            )

        else:

            structure_risk = (
                recent_high
                -
                entry_price
            )

        structure_risk = max(
            structure_risk,
            0.0
        )

        # ====================================================
        # Use conservative risk
        # ====================================================

        risk_distance = max(
            volatility_risk,
            structure_risk * 0.50,
            entry_price * 0.001
        )

        if direction == "BUY":

            stop_loss = (
                entry_price
                -
                risk_distance
            )

        else:

            stop_loss = (
                entry_price
                +
                risk_distance
            )

        return (
            float(stop_loss),
            float(risk_distance),
            float(atr)
        )

    # ========================================================
    # TP LEVELS
    #
    # TP1 = 1:2
    # TP2 = 1:3
    # ========================================================

    def calculate_targets(
        self,
        direction,
        entry_price,
        risk_distance
    ):

        if risk_distance <= 0:
            return 0.0, 0.0

        tp1_distance = (
            risk_distance
            *
            2.0
        )

        tp2_distance = (
            risk_distance
            *
            3.0
        )

        if direction == "BUY":

            tp1 = (
                entry_price
                +
                tp1_distance
            )

            tp2 = (
                entry_price
                +
                tp2_distance
            )

        else:

            tp1 = (
                entry_price
                -
                tp1_distance
            )

            tp2 = (
                entry_price
                -
                tp2_distance
            )

        return (
            float(tp1),
            float(tp2)
        )

    # ========================================================
    # DIRECTION SCORE
    # ========================================================

    def calculate_direction_score(
        self,
        features
    ):

        values = [

            features.get(
                "BOOK_IMB",
                0.0
            ),

            features.get(
                "OFI",
                0.0
            ),

            features.get(
                "TAKER_FLOW",
                0.0
            ),

            features.get(
                "BAYESIAN",
                0.0
            ),

            features.get(
                "ADAPT_CONF",
                0.0
            ),

            features.get(
                "RMT_DOM",
                0.0
            ),

            features.get(
                "HAWKES",
                0.0
            )

        ]

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

        move_percent = features.get(
            "MOVE_PERCENT",
            0.0
        )

        move_strength = np.clip(
            move_percent
            /
            (
                self.min_move_percent
                +
                1e-8
            ),
            0.0,
            2.0
        )

        move_strength = (
            move_strength / 2.0
        )

        impulse = features.get(
            "IMPULSE_SCORE",
            0.0
        )

        spoof_risk = features.get(
            "SPOOF_RISK",
            0.0
        )

        if ml_probability is not None:

            ml_confidence = abs(
                ml_probability
                -
                0.5
            ) * 2.0

        else:

            ml_confidence = 0.0

        confidence = (
            0.35 * direction_strength
            +
            0.20 * move_strength
            +
            0.20 * impulse
            +
            0.15 * ml_confidence
            +
            0.10 * (1.0 - spoof_risk)
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

    # ========================================================
    # FEATURE EXTRACTION
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

            close = (
                df["Close"]
                .astype(float)
            )

            volume = (
                df["Volume"]
                .astype(float)
            )

            returns = (
                close
                .pct_change()
                .dropna()
            )

            # =================================================
            # BOOK
            # =================================================

            bid_volume = float(
                np.sum(
                    bids[:, 1]
                )
            )

            ask_volume = float(
                np.sum(
                    asks[:, 1]
                )
            )

            mid_price = (
                float(bids[0, 0])
                +
                float(asks[0, 0])
            ) / 2.0

            # =================================================
            # 1. BOOK IMBALANCE
            # =================================================

            book_imbalance = (
                bid_volume
                -
                ask_volume
            ) / (
                bid_volume
                +
                ask_volume
                +
                1e-8
            )

            results["BOOK_IMB"] = clip(
                book_imbalance
            )

            # =================================================
            # 2. OFI
            # =================================================

            ofi = self.calculate_ofi(
                bids,
                asks
            )

            results["OFI"] = ofi

            # =================================================
            # 3. TAKER FLOW
            #
            # Approximation unless actual
            # trade aggressor data is available.
            # =================================================

            if len(returns) > 0:

                last_return = float(
                    returns.iloc[-1]
                )

            else:

                last_return = 0.0

            results["TAKER_FLOW"] = (
                1.0
                if last_return > 0
                else
                -1.0
                if last_return < 0
                else
                0.0
            )

            # =================================================
            # 4. MOVE %
            # =================================================

            move_percent = (
                self.calculate_move_percent(
                    close
                )
            )

            results["MOVE_PERCENT"] = (
                move_percent
            )

            # =================================================
            # 5. MOVE SIGNIFICANCE
            # =================================================

            move_significance = (
                self.calculate_move_significance(
                    returns
                )
            )

            results["MOVE_SIGNIFICANCE"] = float(
                np.clip(
                    move_significance / 3.0,
                    0.0,
                    1.0
                )
            )

            # =================================================
            # 6. REALIZED VOL
            # =================================================

            realized_vol = (
                self.calculate_realized_vol(
                    returns
                )
            )

            results["REALIZED_VOL"] = float(
                np.clip(
                    realized_vol
                    /
                    (
                        self.target_vol
                        +
                        1e-8
                    ),
                    0.0,
                    5.0
                )
            )

            # =================================================
            # 7. VOLUME
            # =================================================

            volume_accel, volume_ratio = (
                self.calculate_volume_acceleration(
                    volume
                )
            )

            results["VOLUME_ACCEL"] = float(
                np.clip(
                    volume_accel,
                    -1.0,
                    3.0
                )
            )

            # =================================================
            # 8. IMPULSE
            # =================================================

            results["IMPULSE_SCORE"] = (
                self.calculate_impulse_score(
                    returns,
                    volume_ratio,
                    ofi
                )
            )

            # =================================================
            # 9. QUANTILES
            # =================================================

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

            q_range = (
                q90
                -
                q10
                +
                1e-8
            )

            move_5 = (
                close.iloc[-1]
                /
                (
                    close.iloc[-5]
                    +
                    1e-8
                )
                -
                1.0
            )

            q_position = (
                move_5
                -
                q10
            ) / q_range

            results["QUANTILES"] = float(
                np.clip(
                    q_position * 2.0 - 1.0,
                    -1.0,
                    1.0
                )
            )

            # =================================================
            # 10. BAYESIAN EVIDENCE
            # =================================================

            evidence = np.mean([
                book_imbalance,
                ofi,
                results["TAKER_FLOW"]
            ])

            results["BAYESIAN"] = clip(
                evidence
            )

            # =================================================
            # 11. ADAPTIVE TREND
            # =================================================

            fast_ma = float(
                close.rolling(
                    5
                ).mean().iloc[-1]
            )

            slow_ma = float(
                close.rolling(
                    20
                ).mean().iloc[-1]
            )

            trend_strength = (
                fast_ma
                -
                slow_ma
            ) / (
                mid_price
                *
                (
                    realized_vol
                    +
                    1e-8
                )
                +
                1e-8
            )

            results["ADAPT_CONF"] = clip(
                trend_strength
            )

            # =================================================
            # 12. RMT-STYLE DOMINANCE
            # =================================================

            dominance = (
                abs(move_5)
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
                    (
                        dominance / 3.0
                    )
                    *
                    np.sign(move_5),
                    -1.0,
                    1.0
                )
            )

            # =================================================
            # 13. REWARD/RISK MARKET CONDITION
            # =================================================

            rr_market = (
                abs(q90)
                /
                (
                    abs(q10)
                    +
                    1e-8
                )
            )

            if rr_market >= 1.2:

                results["REWARD_RISK"] = 1.0

            elif rr_market < 0.8:

                results["REWARD_RISK"] = -1.0

            else:

                results["REWARD_RISK"] = 0.0

            # =================================================
            # 14. FRACTIONAL KELLY
            # =================================================

            estimated_probability = (
                0.50
                +
                0.20
                *
                abs(book_imbalance)
            )

            kelly = (
                estimated_probability
                -
                (
                    1.0
                    -
                    estimated_probability
                )
                /
                max(
                    rr_market,
                    1.0
                )
            )

            results["FRAC_KELLY"] = float(
                np.clip(
                    kelly,
                    -1.0,
                    1.0
                )
            )

            # =================================================
            # 15. ACTIVITY
            # =================================================

            activity = (
                self.calculate_activity_intensity(
                    df
                )
            )

            results["HAWKES"] = clip(
                activity
                *
                np.sign(move_5)
            )

            # =================================================
            # 16. SPOOF RISK
            # =================================================

            spoof_risk = (
                self.calculate_spoof_risk(
                    bids,
                    asks
                )
            )

            results["SPOOF_RISK"] = (
                spoof_risk
            )

            # =================================================
            # SAVE BOOK FOR NEXT UPDATE
            # =================================================

            self.previous_bids = np.copy(
                bids
            )

            self.previous_asks = np.copy(
                asks
            )

            return results

        except Exception:

            return results

    # ========================================================
    # TRAIN ML USING REAL FEATURES
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

            for name in self.feature_names:

                value = features.get(
                    name
                )

                try:

                    value = float(value)

                    if not np.isfinite(
                        value
                    ):

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
                1
                if outcome == "WIN"
                else 0
            )

        # Need real history
        if len(X) < 20:
            return False

        # Need WIN + LOSS
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

            X_scaled = (
                self.scaler.transform(X)
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

        # ====================================================
        # FEATURES
        # ====================================================

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

        # ====================================================
        # ML TRAINING
        # ====================================================

        if performance_history:

            self.train_from_history(
                performance_history
            )

        # ====================================================
        # DIRECTION
        # ====================================================

        direction_score = (
            self.calculate_direction_score(
                features
            )
        )

        # ====================================================
        # ML PROBABILITY
        # ====================================================

        ml_probability = None

        if self.is_model_trained:

            try:

                scaled = (
                    self.scaler.transform(
                        feature_vector
                    )
                )

                ml_probability = float(
                    self.ml_model.predict_proba(
                        scaled
                    )[0][1]
                )

            except Exception:

                ml_probability = None

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = (
            self.calculate_confidence(
                features,
                ml_probability
            )
        )

        # ====================================================
        # PRICE
        # ====================================================

        try:

            entry_price = float(
                df["Close"].iloc[-1]
            )

        except Exception:

            entry_price = 0.0

        # ====================================================
        # MAIN 0.40% FILTER
        # ====================================================

        move_percent = float(
            features.get(
                "MOVE_PERCENT",
                0.0
            )
        )

        # Default values
        signal = "NO TRADE"
        reason = "NO_SETUP"

        stop_loss = 0.0
        tp1 = 0.0
        tp2 = 0.0
        risk_distance = 0.0
        rr_tp1 = 0.0
        rr_tp2 = 0.0

        # ====================================================
        # RULE 1:
        # MOVE MUST BE >= 0.40%
        # ====================================================

        if move_percent < self.min_move_percent:

            signal = "NO TRADE"

            reason = (
                f"MOVE_BELOW_"
                f"{self.min_move_percent:.2f}%"
            )

        else:

            # =================================================
            # DIRECTION
            # =================================================

            move_direction = (
                self.calculate_move_direction(
                    df["Close"]
                )
            )

            # =================================================
            # OBI / OFI CONFIRMATION
            # =================================================

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

            # =================================================
            # MANIPULATION FILTER
            # =================================================

            spoof_risk = features.get(
                "SPOOF_RISK",
                0.0
            )

            manipulation_detected = (
                spoof_risk
                >=
                self.spoof_threshold
            )

            # =================================================
            # DETERMINE DIRECTION
            # =================================================

            direction = None

            if (
                bullish_confirmation
                and
                direction_score > 0
                and
                move_direction > 0
            ):

                direction = "BUY"

            elif (
                bearish_confirmation
                and
                direction_score < 0
                and
                move_direction < 0
            ):

                direction = "SELL"

            # =================================================
            # NO DIRECTION
            # =================================================

            if direction is None:

                signal = "WAIT"

                reason = (
                    "MOVE_WITHOUT_DIRECTION_CONFIRMATION"
                )

            # =================================================
            # MANIPULATION
            # =================================================

            elif manipulation_detected:

                signal = "NO TRADE"

                reason = (
                    "POSSIBLE_ORDERBOOK_MANIPULATION"
                )

            else:

                # =================================================
                # DYNAMIC STOP LOSS
                # =================================================

                (
                    stop_loss,
                    risk_distance,
                    atr
                ) = self.calculate_stop_loss(
                    df,
                    direction,
                    entry_price
                )

                # =================================================
                # TARGETS
                # =================================================

                (
                    tp1,
                    tp2
                ) = self.calculate_targets(
                    direction,
                    entry_price,
                    risk_distance
                )

                # =================================================
                # ACTUAL RR
                # =================================================

                if risk_distance > 0:

                    rr_tp1 = (
                        abs(
                            tp1
                            -
                            entry_price
                        )
                        /
                        risk_distance
                    )

                    rr_tp2 = (
                        abs(
                            tp2
                            -
                            entry_price
                        )
                        /
                        risk_distance
                    )

                # =================================================
                # MINIMUM RR CHECK
                # =================================================

                if rr_tp1 < self.minimum_rr:

                    signal = "NO TRADE"

                    reason = (
                        "RR_BELOW_1_2"
                    )

                # =================================================
                # CONFIDENCE CHECK
                # =================================================

                elif (
                    confidence
                    <
                    self.confidence_threshold
                ):

                    signal = "WAIT"

                    reason = (
                        "CONFIDENCE_TOO_LOW"
                    )

                # =================================================
                # STRONG 1:3 SETUP
                # =================================================

                elif rr_tp2 >= self.strong_rr:

                    signal = direction

                    reason = (
                        "STRONG_1_3_RR_SETUP"
                    )

                # =================================================
                # NORMAL 1:2 SETUP
                # =================================================

                else:

                    signal = direction

                    reason = (
                        "VALID_1_2_RR_SETUP"
                    )

        # ====================================================
        # FINAL SCORE
        # ====================================================

        ml_direction = 0.0

        if ml_probability is not None:

            ml_direction = (
                ml_probability
                -
                0.5
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

        # ====================================================
        # RESULT
        # ====================================================

        results = {

            # ------------------------------
            # SIGNAL
            # ------------------------------

            "SIGNAL":
                signal,

            "RAW_SIGNAL":
                signal,

            "REASON":
                reason,

            # ------------------------------
            # SCORE
            # ------------------------------

            "SCORE":
                final_score,

            "FINAL_SCORE":
                final_score,

            "DIRECTION_SCORE":
                direction_score,

            # ------------------------------
            # CONFIDENCE
            # ------------------------------

            "CONFIDENCE":
                confidence,

            "ML_PROBABILITY":
                ml_probability,

            "MODEL_TRAINED":
                self.is_model_trained,

            # ------------------------------
            # MOVE
            # ------------------------------

            "MOVE_PERCENT":
                move_percent,

            "MIN_MOVE_PERCENT":
                self.min_move_percent,

            "MOVE_LOOKBACK":
                self.move_lookback,

            # ------------------------------
            # ORDER BOOK
            # ------------------------------

            "OBI":
                features["BOOK_IMB"],

            "BOOK_IMB":
                features["BOOK_IMB"],

            "OFI":
                features["OFI"],

            # ------------------------------
            # IMPULSE
            # ------------------------------

            "IMPULSE_SCORE":
                features["IMPULSE_SCORE"],

            "MOVE_SIGNIFICANCE":
                features["MOVE_SIGNIFICANCE"],

            # ------------------------------
            # MANIPULATION
            # ------------------------------

            "SPOOF_RISK":
                features["SPOOF_RISK"],

            "MANIPULATION":
                (
                    features["SPOOF_RISK"]
                    >=
                    self.spoof_threshold
                ),

            # ------------------------------
            # PRICE
            # ------------------------------

            "ENTRY_PRICE":
                entry_price,

            # ------------------------------
            # STOP LOSS
            # ------------------------------

            "STOP_LOSS":
                stop_loss,

            "RISK_DISTANCE":
                risk_distance,

            # ------------------------------
            # TARGETS
            # ------------------------------

            "TP1":
                tp1,

            "TP2":
                tp2,

            # ------------------------------
            # RISK REWARD
            # ------------------------------

            "RR_TP1":
                rr_tp1,

            "RR_TP2":
                rr_tp2,

            "MINIMUM_RR":
                self.minimum_rr,

            "STRONG_RR":
                self.strong_rr,

            # ------------------------------
            # VOLATILITY
            # ------------------------------

            "REALIZED_VOL":
                features["REALIZED_VOL"],

            "VOLUME_ACCEL":
                features["VOLUME_ACCEL"],

            # ------------------------------
            # ALL FEATURES
            # ------------------------------

            "FEATURES":
                features
        }

        return (
            results,
            final_score,
            self.dynamic_weights
        )


# ============================================================
# POWER TRADING RISK ENGINE
# ============================================================

class PowerTradingRiskEngine:

    def __init__(self):

        self.risk_levels = {
            "LOW": 25.0,
            "MEDIUM": 50.0,
            "HIGH": 75.0
        }

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

        # ====================================================
        # LIQUIDATION DATA
        # ====================================================

        try:

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

        except Exception:

            liquidation_volumes = np.array([])

        if len(
            liquidation_volumes
        ) > 0:

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

        # ====================================================
        # LTZ SCORE
        # ====================================================

        ltz_score = (
            max_ltz
            /
            (
                total_ltz
                +
                1e-8
            )
        ) * 100.0

        ltz_score = float(
            np.clip(
                ltz_score,
                0.0,
                100.0
            )
        )

        # ====================================================
        # SPOOF SCORE
        # ====================================================

        displayed_vol = max(
            safe_float(
                displayed_vol
            ),
            0.0
        )

        cancelled_vol = max(
            safe_float(
                cancelled_vol
            ),
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
            safe_float(
                time_exists
            )
            /
            (
                safe_float(
                    obs_window
                )
                +
                1e-8
            ),
            0.0,
            1.0
        )

        spoof_score = (
            spoof_ratio
            *
            (
                1.0
                -
                persistence
            )
            *
            100.0
        )

        spoof_score = float(
            np.clip(
                spoof_score,
                0.0,
                100.0
            )
        )

        # ====================================================
        # SQUEEZE RISK
        # ====================================================

        open_interest = max(
            safe_float(
                open_interest
            ),
            0.0
        )

        leverage = max(
            safe_float(
                leverage
            ),
            0.0
        )

        volatility = max(
            safe_float(
                volatility
            ),
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

        # Log scaling prevents giant raw values
        squeeze_score = float(
            np.clip(
                np.log1p(
                    raw_squeeze
                )
                *
                10.0,
                0.0,
                100.0
            )
        )

        # ====================================================
        # COMBINED RISK
        # ====================================================

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

        # ====================================================
        # LEVEL
        # ====================================================

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


# ============================================================
# INTEGRATED TRADING ENGINE
#
# Optional wrapper if dashboard.py imports:
# from engine import IntegratedTradingEngine
# ============================================================

class IntegratedTradingEngine:

    def __init__(
        self,
        target_vol=0.15,
        min_move_percent=0.40,
        move_lookback=5
    ):

        self.research = TenPaperResearchLab(

            target_vol=target_vol,

            min_move_percent=min_move_percent,

            move_lookback=move_lookback,

            # OBI
            obi_threshold=0.15,

            # OFI
            ofi_threshold=0.10,

            # Confidence
            confidence_threshold=0.60,

            # RR
            minimum_rr=2.0,
            strong_rr=3.0,

            # SL
            atr_period=14,
            atr_multiplier=1.0,

            # Manipulation
            spoof_threshold=0.60
        )

        self.risk = (
            PowerTradingRiskEngine()
        )

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=None,
        liquidation_volumes=None,
        displayed_vol=0.0,
        cancelled_vol=0.0,
        time_exists=0.0,
        obs_window=1.0,
        open_interest=0.0,
        leverage=1.0,
        volatility=0.0
    ):

        if liquidation_volumes is None:
            liquidation_volumes = []

        # ====================================================
        # RESEARCH ENGINE
        # ====================================================

        (
            signal_data,
            final_score,
            weights
        ) = self.research.calculate_all_signals(

            df=df,

            bids=bids,

            asks=asks,

            current_inventory=current_inventory,

            performance_history=performance_history
        )

        # ====================================================
        # RISK ENGINE
        # ====================================================

        risk_data = (
            self.risk.calculate_risk_metrics(

                liquidation_volumes=
                    liquidation_volumes,

                displayed_vol=
                    displayed_vol,

                cancelled_vol=
                    cancelled_vol,

                time_exists=
                    time_exists,

                obs_window=
                    obs_window,

                open_interest=
                    open_interest,

                leverage=
                    leverage,

                volatility=
                    volatility
            )
        )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        output = {

            "SIGNAL":
                signal_data["SIGNAL"],

            "RAW_SIGNAL":
                signal_data["RAW_SIGNAL"],

            "REASON":
                signal_data["REASON"],

            "SCORE":
                signal_data["SCORE"],

            "FINAL_SCORE":
                final_score,

            "CONFIDENCE":
                signal_data["CONFIDENCE"],

            "RISK":
                risk_data,

            "ENTRY_PRICE":
                signal_data["ENTRY_PRICE"],

            "STOP_LOSS":
                signal_data["STOP_LOSS"],

            "TP1":
                signal_data["TP1"],

            "TP2":
                signal_data["TP2"],

            "RR_TP1":
                signal_data["RR_TP1"],

            "RR_TP2":
                signal_data["RR_TP2"],

            "MOVE_PERCENT":
                signal_data["MOVE_PERCENT"],

            "MIN_MOVE_PERCENT":
                signal_data["MIN_MOVE_PERCENT"],

            "OBI":
                signal_data["OBI"],

            "OFI":
                signal_data["OFI"],

            "SPOOF_RISK":
                signal_data["SPOOF_RISK"],

            "MANIPULATION":
                signal_data["MANIPULATION"],

            "IMPULSE_SCORE":
                signal_data["IMPULSE_SCORE"],

            "MOVE_SIGNIFICANCE":
                signal_data["MOVE_SIGNIFICANCE"],

            "FEATURES":
                signal_data["FEATURES"],

            "WEIGHTS":
                weights
        }

        return output
