# engine.py

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SignalConfig:

    # --------------------------------------------------------
    # MINIMUM MOVE
    # --------------------------------------------------------
    # Trade tabhi allow hoga jab expected/recent move >= 0.40%
    min_move_pct: float = 0.004

    # --------------------------------------------------------
    # ORDER BOOK CONFIRMATION
    # --------------------------------------------------------
    obi_threshold: float = 0.15
    ofi_threshold: float = 0.10

    # --------------------------------------------------------
    # SIGNAL CONFIDENCE
    # --------------------------------------------------------
    min_confidence: float = 0.70
    strong_confidence: float = 0.80

    # --------------------------------------------------------
    # RISK / REWARD
    # --------------------------------------------------------
    rr_tp1: float = 2.0       # 1:2
    rr_tp2: float = 3.0       # 1:3

    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------
    atr_period: int = 14
    atr_multiplier: float = 1.0

    min_stop_pct: float = 0.0015      # 0.15%
    max_stop_pct: float = 0.0080      # 0.80%

    # --------------------------------------------------------
    # VOLATILITY REGIME
    # --------------------------------------------------------
    vol_window: int = 30

    min_vol_ratio: float = 0.80
    max_vol_ratio: float = 3.50

    # --------------------------------------------------------
    # IMPULSE / EXHAUSTION
    # --------------------------------------------------------
    impulse_z: float = 2.5
    exhaustion_z: float = 3.5

    # --------------------------------------------------------
    # MANIPULATION
    # --------------------------------------------------------
    cancel_ratio_limit: float = 0.70
    persistence_min: float = 0.25

    # --------------------------------------------------------
    # ONLINE ML
    # --------------------------------------------------------
    min_training_samples: int = 30
    max_training_samples: int = 500

    random_state: int = 42


# ============================================================
# RESEARCH FEATURES
# ============================================================

class ResearchFeatures:

    names = [

        "HAWKES",
        "BOOK_IMB",
        "OFI",
        "TAKER_FLOW",
        "DEPTH_SKEW",
        "BAYESIAN",
        "QUANTILES",
        "MOVE_FILTER",
        "ADAPTIVE_TREND",
        "CONFORMAL",
        "KELLY",
        "RMT_PROXY",
        "REWARD_RISK",
        "VOL_REGIME",
        "IMPULSE",
        "EXHAUSTION",

    ]

    # ========================================================
    # SAFE SERIES
    # ========================================================

    @staticmethod
    def _safe_series(
        df: pd.DataFrame,
        col: str,
        default=0.0
    ):

        if col in df.columns:

            return (
                pd.to_numeric(
                    df[col],
                    errors="coerce"
                )
                .fillna(default)
            )

        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    # ========================================================
    # Z SCORE
    # ========================================================

    @staticmethod
    def _zscore(
        x: pd.Series,
        window: int = 30
    ) -> float:

        x = (
            pd.to_numeric(
                x,
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(x) < max(
            10,
            window // 2
        ):
            return 0.0

        w = x.iloc[-window:]

        mu = w.mean()

        sd = w.std(
            ddof=1
        )

        if (
            not np.isfinite(sd)
            or sd < EPS
        ):
            return 0.0

        return float(
            (w.iloc[-1] - mu) /
            sd
        )

    # ========================================================
    # ATR
    # ========================================================

    @staticmethod
    def atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> float:

        if len(df) < 2:
            return 0.0

        close = ResearchFeatures._safe_series(
            df,
            "Close"
        )

        high = ResearchFeatures._safe_series(
            df,
            "High",
            close
        )

        low = ResearchFeatures._safe_series(
            df,
            "Low",
            close
        )

        previous_close = close.shift(1)

        tr = pd.concat(
            [

                high - low,

                (
                    high -
                    previous_close
                ).abs(),

                (
                    low -
                    previous_close
                ).abs(),

            ],
            axis=1
        ).max(axis=1)

        value = (
            tr
            .rolling(
                period,
                min_periods=max(
                    3,
                    period // 2
                )
            )
            .mean()
            .iloc[-1]
        )

        if np.isfinite(value):
            return float(value)

        return float(
            tr.iloc[-1]
        )

    # ========================================================
    # HAWKES INTENSITY
    # ========================================================

    @staticmethod
    def hawkes_intensity_from_events(
        event_times: np.ndarray,
        decay: float = 1.0,
        baseline: float = 1.0
    ) -> float:

        t = np.asarray(
            event_times,
            dtype=float
        )

        t = t[
            np.isfinite(t)
        ]

        if len(t) < 3:
            return 0.0

        t = np.sort(t)

        gaps = np.diff(t)

        positive = gaps[
            gaps > 0
        ]

        if len(positive) == 0:
            return 0.0

        scale = float(
            np.median(
                positive
            )
        )

        if scale <= 0:
            return 0.0

        normalized_time = (
            t - t[-1]
        ) / scale

        alpha = 1.0

        excitation = np.sum(
            alpha *
            np.exp(
                decay *
                normalized_time[:-1]
            )
        )

        recent = (
            excitation /
            max(
                len(t) - 1,
                1
            )
        )

        return float(
            np.clip(
                (
                    recent -
                    baseline
                ) /
                (
                    baseline +
                    EPS
                ),
                -1,
                1
            )
        )

    # ========================================================
    # ORDER BOOK IMBALANCE
    # ========================================================

    @staticmethod
    def order_book_imbalance(
        bids: np.ndarray,
        asks: np.ndarray,
        levels: int = 20
    ) -> float:

        b = np.asarray(
            bids,
            dtype=float
        )[:levels]

        a = np.asarray(
            asks,
            dtype=float
        )[:levels]

        if (
            len(b) == 0
            or len(a) == 0
        ):
            return 0.0

        bid_volume = np.clip(
            b[:, 1],
            0,
            None
        ).sum()

        ask_volume = np.clip(
            a[:, 1],
            0,
            None
        ).sum()

        return float(
            (
                bid_volume -
                ask_volume
            )
            /
            (
                bid_volume +
                ask_volume +
                EPS
            )
        )

    # ========================================================
    # DEPTH SKEW
    # ========================================================

    @staticmethod
    def depth_skew(
        bids: np.ndarray,
        asks: np.ndarray
    ) -> float:

        b = np.asarray(
            bids,
            dtype=float
        )

        a = np.asarray(
            asks,
            dtype=float
        )

        if (
            len(b) == 0
            or len(a) == 0
        ):
            return 0.0

        bid1 = max(
            float(b[0, 1]),
            0.0
        )

        ask1 = max(
            float(a[0, 1]),
            0.0
        )

        return float(
            (
                bid1 -
                ask1
            )
            /
            (
                bid1 +
                ask1 +
                EPS
            )
        )

    # ========================================================
    # OFI
    # ========================================================

    @staticmethod
    def ofi_from_snapshots(
        prev_bids: np.ndarray,
        prev_asks: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
        levels: int = 1
    ) -> float:

        pb = np.asarray(
            prev_bids,
            dtype=float
        )[:levels]

        pa = np.asarray(
            prev_asks,
            dtype=float
        )[:levels]

        cb = np.asarray(
            bids,
            dtype=float
        )[:levels]

        ca = np.asarray(
            asks,
            dtype=float
        )[:levels]

        if (
            len(pb) == 0
            or len(pa) == 0
            or len(cb) == 0
            or len(ca) == 0
        ):
            return 0.0

        previous_bid_price, previous_bid_size = pb[0]

        previous_ask_price, previous_ask_size = pa[0]

        current_bid_price, current_bid_size = cb[0]

        current_ask_price, current_ask_size = ca[0]

        # Bid event
        if current_bid_price > previous_bid_price:

            bid_event = current_bid_size

        elif current_bid_price < previous_bid_price:

            bid_event = -previous_bid_size

        else:

            bid_event = (
                current_bid_size -
                previous_bid_size
            )

        # Ask event
        if current_ask_price < previous_ask_price:

            ask_event = current_ask_size

        elif current_ask_price > previous_ask_price:

            ask_event = -previous_ask_size

        else:

            ask_event = (
                current_ask_size -
                previous_ask_size
            )

        raw_ofi = (
            bid_event -
            ask_event
        )

        depth = (
            abs(current_bid_size) +
            abs(current_ask_size) +
            EPS
        )

        return float(
            np.tanh(
                raw_ofi /
                depth
            )
        )

    # ========================================================
    # TAKER FLOW
    # ========================================================

    @staticmethod
    def taker_flow(
        df: pd.DataFrame
    ) -> float:

        # Prefer actual taker data
        if (
            "TakerBuyVolume" in df.columns
            and
            "TakerSellVolume" in df.columns
        ):

            buy = float(
                pd.to_numeric(
                    df[
                        "TakerBuyVolume"
                    ],
                    errors="coerce"
                )
                .fillna(0)
                .iloc[-1]
            )

            sell = float(
                pd.to_numeric(
                    df[
                        "TakerSellVolume"
                    ],
                    errors="coerce"
                )
                .fillna(0)
                .iloc[-1]
            )

            return float(
                (
                    buy -
                    sell
                )
                /
                (
                    buy +
                    sell +
                    EPS
                )
            )

        # Fallback signed volume
        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        volume = (
            ResearchFeatures
            ._safe_series(
                df,
                "Volume"
            )
        )

        if len(close) < 2:
            return 0.0

        direction = (
            np.sign(
                close.diff()
            )
            .fillna(0)
        )

        signed_volume = float(
            (
                direction *
                volume
            )
            .tail(5)
            .sum()
        )

        total_volume = float(
            volume
            .tail(5)
            .sum()
        )

        return float(
            np.clip(
                signed_volume /
                (
                    total_volume +
                    EPS
                ),
                -1,
                1
            )
        )

    # ========================================================
    # EMPIRICAL BAYES
    # ========================================================

    @staticmethod
    def empirical_bayesian(
        book_imb: float,
        history: Optional[
            List[dict]
        ]
    ) -> float:

        wins = 1.0
        losses = 1.0

        if history:

            for h in history[-500:]:

                outcome = str(
                    h.get(
                        "outcome",
                        ""
                    )
                ).upper()

                direction = str(
                    h.get(
                        "direction",
                        ""
                    )
                ).upper()

                if outcome not in {
                    "WIN",
                    "LOSS"
                }:
                    continue

                if (
                    book_imb >= 0
                    and
                    direction == "BUY"
                ):

                    if outcome == "WIN":
                        wins += 1
                    else:
                        losses += 1

                elif (
                    book_imb < 0
                    and
                    direction == "SELL"
                ):

                    if outcome == "WIN":
                        wins += 1
                    else:
                        losses += 1

        probability = (
            wins /
            (
                wins +
                losses
            )
        )

        score = (
            probability -
            0.5
        ) * 2.0

        return float(
            np.clip(
                score *
                np.sign(book_imb),
                -1,
                1
            )
        )

    # ========================================================
    # QUANTILE
    # ========================================================

    @staticmethod
    def quantile_score(
        returns: pd.Series
    ) -> float:

        r = returns.dropna()

        if len(r) < 10:
            return 0.0

        q10 = r.quantile(
            0.10
        )

        q50 = r.quantile(
            0.50
        )

        q90 = r.quantile(
            0.90
        )

        current = float(
            r.iloc[-1]
        )

        if current >= q50:

            denominator = max(
                q90 - q50,
                EPS
            )

            score = (
                current -
                q50
            ) / denominator

        else:

            denominator = max(
                q50 - q10,
                EPS
            )

            score = (
                current -
                q50
            ) / denominator

        return float(
            np.clip(
                score,
                -1,
                1
            )
        )

    # ========================================================
    # ADAPTIVE TREND
    # ========================================================

    @staticmethod
    def adaptive_trend(
        df: pd.DataFrame
    ) -> float:

        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        if len(close) < 20:
            return 0.0

        fast = (
            close
            .ewm(
                span=5,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        slow = (
            close
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        volatility = (
            returns
            .tail(20)
            .std()
        )

        if (
            not np.isfinite(
                volatility
            )
            or
            volatility < EPS
        ):
            return 0.0

        return float(
            np.clip(
                (
                    (
                        fast -
                        slow
                    )
                    /
                    close.iloc[-1]
                )
                /
                volatility,
                -1,
                1
            )
        )

    # ========================================================
    # CONFORMAL
    # ========================================================

    @staticmethod
    def conformal_score(
        df: pd.DataFrame,
        window: int = 50
    ) -> float:

        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        if len(returns) < 20:
            return 0.0

        history = (
            returns
            .iloc[:-1]
            .tail(window)
        )

        current = float(
            returns.iloc[-1]
        )

        center = float(
            history.median()
        )

        scores = (
            history -
            center
        ).abs()

        threshold = float(
            scores.quantile(
                0.90
            )
        )

        if threshold < EPS:
            return float(
                np.sign(current)
            )

        magnitude = (
            abs(
                current -
                center
            )
            /
            threshold
        )

        return float(
            np.clip(
                np.sign(current) *
                magnitude,
                -1,
                1
            )
        )

    # ========================================================
    # KELLY
    # ========================================================

    @staticmethod
    def empirical_kelly(
        history: Optional[
            List[dict]
        ],
        rr: float = 2.0,
        max_fraction: float = 0.25
    ) -> float:

        wins = 0
        losses = 0

        if history:

            for h in history[-500:]:

                outcome = str(
                    h.get(
                        "outcome",
                        ""
                    )
                ).upper()

                if outcome == "WIN":
                    wins += 1

                elif outcome == "LOSS":
                    losses += 1

        probability = (
            wins + 1
        ) / (
            wins +
            losses +
            2
        )

        q = (
            1 -
            probability
        )

        b = max(
            rr,
            0.1
        )

        kelly = (
            b *
            probability -
            q
        ) / b

        return float(
            np.clip(
                kelly,
                -max_fraction,
                max_fraction
            )
        )

    # ========================================================
    # RMT PROXY
    # ========================================================

    @staticmethod
    def rmt_proxy(
        df: pd.DataFrame
    ) -> float:

        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        if len(returns) < 20:
            return 0.0

        current = float(
            returns.iloc[-1]
        )

        volatility = float(
            returns
            .tail(20)
            .std()
        )

        if volatility < EPS:
            return 0.0

        z = (
            current /
            volatility
        )

        return float(
            np.clip(
                z / 3.0,
                -1,
                1
            )
        )

    # ========================================================
    # REWARD / RISK DISTRIBUTION
    # ========================================================

    @staticmethod
    def reward_risk_from_distribution(
        df: pd.DataFrame
    ) -> float:

        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        if len(returns) < 20:
            return 0.0

        upside = returns[
            returns > 0
        ]

        downside = -returns[
            returns < 0
        ]

        if (
            len(upside) < 3
            or
            len(downside) < 3
        ):
            return 0.0

        expected_up = float(
            upside.quantile(
                0.75
            )
        )

        expected_down = float(
            downside.quantile(
                0.75
            )
        )

        ratio = (
            expected_up /
            (
                expected_down +
                EPS
            )
        )

        if ratio >= 1.2:

            return float(
                np.clip(
                    ratio - 1.0,
                    0,
                    1
                )
            )

        if ratio <= 0.8:

            return float(
                -np.clip(
                    1.0 - ratio,
                    0,
                    1
                )
            )

        return float(
            (
                ratio -
                1.0
            ) * 2.0
        )

    # ========================================================
    # VOLATILITY REGIME
    # ========================================================

    @staticmethod
    def volatility_regime(
        df: pd.DataFrame,
        window: int = 30
    ) -> float:

        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        if len(returns) < (
            window + 5
        ):
            return 1.0

        short_vol = float(
            returns
            .tail(5)
            .std()
        )

        long_vol = float(
            returns
            .tail(window)
            .std()
        )

        if long_vol < EPS:
            return 1.0

        return float(
            short_vol /
            long_vol
        )

    # ========================================================
    # IMPULSE
    # ========================================================

    @staticmethod
    def impulse_score(
        df: pd.DataFrame
    ) -> float:

        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        volume = (
            ResearchFeatures
            ._safe_series(
                df,
                "Volume"
            )
        )

        if len(close) < 20:
            return 0.0

        returns = (
            close
            .pct_change()
            .dropna()
        )

        price_z = (
            ResearchFeatures
            ._zscore(
                returns,
                30
            )
        )

        volume_change = (
            volume
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        volume_z = (
            ResearchFeatures
            ._zscore(
                volume_change,
                30
            )
        )

        score = (
            0.6 *
            price_z
            +
            0.4 *
            volume_z
        )

        return float(
            np.clip(
                score / 3.0,
                -1,
                1
            )
        )

    # ========================================================
    # EXHAUSTION
    # ========================================================

    @staticmethod
    def exhaustion_score(
        df: pd.DataFrame
    ) -> float:

        close = (
            ResearchFeatures
            ._safe_series(
                df,
                "Close"
            )
        )

        if len(close) < 30:
            return 0.0

        returns = (
            close
            .pct_change()
            .dropna()
        )

        z = (
            ResearchFeatures
            ._zscore(
                returns,
                30
            )
        )

        if abs(z) < 2.0:
            return 0.0

        return float(
            np.clip(
                np.sign(z)
                *
                (
                    abs(z) -
                    2.0
                )
                /
                2.0,
                -1,
                1
            )
        )

    # ========================================================
    # MAIN FEATURE EXTRACTION
    # ========================================================

    @classmethod
    def extract(
        cls,
        df: pd.DataFrame,
        bids: np.ndarray,
        asks: np.ndarray,
        prev_bids=None,
        prev_asks=None,
        performance_history=None,
        config=None
    ) -> Dict[str, float]:

        cfg = (
            config
            if config is not None
            else SignalConfig()
        )

        if (
            df.empty
            or len(df) < 20
            or len(bids) == 0
            or len(asks) == 0
        ):

            return {
                key: 0.0
                for key in cls.names
            }

        close = (
            cls
            ._safe_series(
                df,
                "Close"
            )
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        if len(close) >= 6:

            move_5 = float(
                (
                    close.iloc[-1] -
                    close.iloc[-6]
                )
                /
                (
                    abs(
                        close.iloc[-6]
                    )
                    +
                    EPS
                )
            )

        else:

            move_5 = 0.0

        # ----------------------------------------------------
        # HAWKES
        # ----------------------------------------------------

        event_times = None

        if "EventTime" in df.columns:

            event_times = (
                pd.to_numeric(
                    df["EventTime"],
                    errors="coerce"
                )
                .dropna()
                .values
            )

        elif "Timestamp" in df.columns:

            event_times = (
                pd.to_numeric(
                    df["Timestamp"],
                    errors="coerce"
                )
                .dropna()
                .values
            )

        if (
            event_times is not None
            and
            len(event_times) >= 5
        ):

            hawkes = (
                cls
                .hawkes_intensity_from_events(
                    event_times
                )
            )

            hawkes *= np.sign(
                move_5
            )

        else:

            volume = (
                cls
                ._safe_series(
                    df,
                    "Volume"
                )
            )

            volume_change = (
                volume
                .pct_change()
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
            )

            z = cls._zscore(
                volume_change,
                30
            )

            hawkes = float(
                np.clip(
                    z / 3.0,
                    -1,
                    1
                )
            ) * np.sign(
                move_5
            )

        # ----------------------------------------------------
        # ORDER BOOK
        # ----------------------------------------------------

        obi = (
            cls
            .order_book_imbalance(
                bids,
                asks,
                20
            )
        )

        depth = (
            cls
            .depth_skew(
                bids,
                asks
            )
        )

        # ----------------------------------------------------
        # OFI
        # ----------------------------------------------------

        if (
            prev_bids is not None
            and
            prev_asks is not None
        ):

            ofi = (
                cls
                .ofi_from_snapshots(
                    prev_bids,
                    prev_asks,
                    bids,
                    asks
                )
            )

        else:

            ofi = 0.0

        # ----------------------------------------------------
        # OTHER FEATURES
        # ----------------------------------------------------

        taker = cls.taker_flow(
            df
        )

        bayesian = (
            cls
            .empirical_bayesian(
                obi,
                performance_history
            )
        )

        quantiles = cls.quantile_score(
            returns
        )

        trend = cls.adaptive_trend(
            df
        )

        conformal = cls.conformal_score(
            df
        )

        kelly = cls.empirical_kelly(
            performance_history,
            cfg.rr_tp1
        )

        rmt = cls.rmt_proxy(
            df
        )

        rr = (
            cls
            .reward_risk_from_distribution(
                df
            )
        )

        vol_ratio = (
            cls
            .volatility_regime(
                df,
                cfg.vol_window
            )
        )

        impulse = cls.impulse_score(
            df
        )

        exhaustion = cls.exhaustion_score(
            df
        )

        return {

            "HAWKES":
                float(
                    np.clip(
                        hawkes,
                        -1,
                        1
                    )
                ),

            "BOOK_IMB":
                float(
                    np.clip(
                        obi,
                        -1,
                        1
                    )
                ),

            "OFI":
                float(
                    np.clip(
                        ofi,
                        -1,
                        1
                    )
                ),

            "TAKER_FLOW":
                float(
                    np.clip(
                        taker,
                        -1,
                        1
                    )
                ),

            "DEPTH_SKEW":
                float(
                    np.clip(
                        depth,
                        -1,
                        1
                    )
                ),

            "BAYESIAN":
                float(
                    np.clip(
                        bayesian,
                        -1,
                        1
                    )
                ),

            "QUANTILES":
                float(
                    np.clip(
                        quantiles,
                        -1,
                        1
                    )
                ),

            # 0.40% move filter
            "MOVE_FILTER":
                float(
                    np.clip(
                        move_5 /
                        cfg.min_move_pct,
                        -1,
                        1
                    )
                ),

            "ADAPTIVE_TREND":
                float(
                    np.clip(
                        trend,
                        -1,
                        1
                    )
                ),

            "CONFORMAL":
                float(
                    np.clip(
                        conformal,
                        -1,
                        1
                    )
                ),

            "KELLY":
                float(
                    np.clip(
                        kelly * 4.0,
                        -1,
                        1
                    )
                ),

            "RMT_PROXY":
                float(
                    np.clip(
                        rmt,
                        -1,
                        1
                    )
                ),

            "REWARD_RISK":
                float(
                    np.clip(
                        rr,
                        -1,
                        1
                    )
                ),

            "VOL_REGIME":
                float(
                    np.clip(
                        (
                            vol_ratio -
                            1.0
                        ) / 2.0,
                        -1,
                        1
                    )
                ),

            "IMPULSE":
                float(
                    np.clip(
                        impulse,
                        -1,
                        1
                    )
                ),

            "EXHAUSTION":
                float(
                    np.clip(
                        exhaustion,
                        -1,
                        1
                    )
                ),
        }


# ============================================================
# ONLINE ML MODEL
# ============================================================

class OnlineSignalModel:

    def __init__(
        self,
        feature_names,
        random_state=42
    ):

        self.feature_names = list(
            feature_names
        )

        self.scaler = (
            StandardScaler()
        )

        self.model = (
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                learning_rate="optimal",
                random_state=random_state,
                max_iter=1,
                tol=None
            )
        )

        self.ready = False

        self.samples = 0

        self.classes = np.array(
            [0, 1]
        )

    # ========================================================
    # ONLINE UPDATE
    # ========================================================

    def update(
        self,
        features,
        outcome
    ):

        x = np.array(
            [
                [
                    features.get(
                        name,
                        0.0
                    )
                    for name in
                    self.feature_names
                ]
            ],
            dtype=float
        )

        if not np.isfinite(
            x
        ).all():

            return

        self.scaler.partial_fit(
            x
        )

        x_scaled = (
            self.scaler
            .transform(x)
        )

        self.model.partial_fit(
            x_scaled,
            np.array(
                [
                    int(outcome)
                ]
            ),
            classes=self.classes
        )

        self.ready = True

        self.samples += 1

    # ========================================================
    # PROBABILITY
    # ========================================================

    def probability(
        self,
        features
    ) -> float:

        if not self.ready:

            return 0.5

        x = np.array(
            [
                [
                    features.get(
                        name,
                        0.0
                    )
                    for name in
                    self.feature_names
                ]
            ],
            dtype=float
        )

        x_scaled = (
            self.scaler
            .transform(x)
        )

        return float(
            self.model
            .predict_proba(
                x_scaled
            )[0][1]
        )


# ============================================================
# MANIPULATION RISK
# ============================================================

class ManipulationRisk:

    @staticmethod
    def score(
        displayed_volume,
        cancelled_volume,
        time_exists,
        observation_window
    ) -> float:

        displayed_volume = max(
            float(displayed_volume),
            0.0
        )

        cancelled_volume = max(
            float(cancelled_volume),
            0.0
        )

        observation_window = max(
            float(observation_window),
            EPS
        )

        cancel_ratio = (
            cancelled_volume /
            (
                displayed_volume +
                EPS
            )
        )

        persistence = np.clip(
            float(time_exists) /
            observation_window,
            0,
            1
        )

        risk = (
            cancel_ratio *
            (
                1 -
                persistence
            )
        )

        return float(
            np.clip(
                risk,
                0,
                1
            )
        )


# ============================================================
# POWER RISK ENGINE
# ============================================================

class PowerTradingRiskEngine:

    def calculate_risk_metrics(
        self,
        liquidation_volumes,
        displayed_vol=0.0,
        cancelled_vol=0.0,
        time_exists=0.0,
        obs_window=1.0,
        open_interest=0.0,
        leverage=1.0,
        volatility=0.0
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

        liquidation_volumes = np.clip(
            liquidation_volumes,
            0,
            None
        )

        total_ltz = (
            float(
                liquidation_volumes.sum()
            )
            if len(
                liquidation_volumes
            )
            else 0.0
        )

        max_ltz = (
            float(
                liquidation_volumes.max()
            )
            if len(
                liquidation_volumes
            )
            else 0.0
        )

        if total_ltz > 0:

            ltz_score = (
                max_ltz /
                (
                    total_ltz +
                    EPS
                )
            ) * 100

        else:

            ltz_score = 0.0

        spoof_score = (
            ManipulationRisk
            .score(
                displayed_vol,
                cancelled_vol,
                time_exists,
                obs_window
            )
        )

        # Log scaling prevents huge OI/leverage numbers
        # from completely dominating the risk score.

        squeeze_raw = (

            np.log1p(
                total_ltz
            )

            *

            np.log1p(
                max(
                    float(
                        open_interest
                    ),
                    0.0
                )
            )

            *

            np.log1p(
                max(
                    float(
                        leverage
                    ),
                    1.0
                )
            )

            *

            max(
                float(
                    volatility
                ),
                0.0
            )
        )

        squeeze_score = float(
            np.clip(
                squeeze_raw *
                10.0,
                0,
                100
            )
        )

        market_risk = float(
            np.clip(

                0.50 *
                min(
                    ltz_score,
                    100
                )

                +

                0.25 *
                spoof_score *
                100

                +

                0.25 *
                squeeze_score,

                0,
                100
            )
        )

        return {

            "LTZ_Score":
                float(
                    np.clip(
                        ltz_score,
                        0,
                        100
                    )
                ),

            "Spoof_Score":
                float(
                    spoof_score
                ),

            "Squeeze_Risk":
                squeeze_score,

            "Market_Risk":
                market_risk,
        }


# ============================================================
# MAIN RESEARCH LAB
# ============================================================

class TenPaperResearchLab:

    def __init__(
        self,
        target_vol=0.15,
        config=None
    ):

        self.target_vol = (
            target_vol
        )

        self.config = (
            config
            if config is not None
            else SignalConfig()
        )

        self.feature_names = (
            ResearchFeatures.names
        )

        # ----------------------------------------------------
        # INITIAL WEIGHTS
        # ----------------------------------------------------

        raw_weights = {

            "HAWKES": 0.07,

            "BOOK_IMB": 0.10,

            "OFI": 0.12,

            "TAKER_FLOW": 0.10,

            "DEPTH_SKEW": 0.05,

            "BAYESIAN": 0.05,

            "QUANTILES": 0.05,

            "MOVE_FILTER": 0.08,

            "ADAPTIVE_TREND": 0.08,

            "CONFORMAL": 0.05,

            "KELLY": 0.03,

            "RMT_PROXY": 0.06,

            "REWARD_RISK": 0.07,

            "VOL_REGIME": 0.04,

            "IMPULSE": 0.08,

            "EXHAUSTION": 0.07,
        }

        total = sum(
            raw_weights.values()
        )

        self.dynamic_weights = {

            key:
            value / total

            for key, value
            in raw_weights.items()
        }

        # ----------------------------------------------------
        # ONLINE ML
        # ----------------------------------------------------

        self.ml_model = (
            OnlineSignalModel(
                self.feature_names,
                self.config.random_state
            )
        )

        self.last_features = None

    # ========================================================
    # EXTRACT
    # ========================================================

    def extract_features(
        self,
        df,
        bids,
        asks,
        prev_bids=None,
        prev_asks=None,
        performance_history=None
    ):

        features = (
            ResearchFeatures
            .extract(

                df=pd.DataFrame(
                    df
                ),

                bids=np.asarray(
                    bids,
                    dtype=float
                ),

                asks=np.asarray(
                    asks,
                    dtype=float
                ),

                prev_bids=(
                    None
                    if prev_bids is None
                    else
                    np.asarray(
                        prev_bids,
                        dtype=float
                    )
                ),

                prev_asks=(
                    None
                    if prev_asks is None
                    else
                    np.asarray(
                        prev_asks,
                        dtype=float
                    )
                ),

                performance_history=
                performance_history,

                config=self.config
            )
        )

        self.last_features = (
            features
        )

        return features

    # ========================================================
    # WEIGHTED SCORE
    # ========================================================

    def _weighted_score(
        self,
        features
    ):

        values = np.array(
            [
                features.get(
                    key,
                    0.0
                )
                for key in
                self.feature_names
            ],
            dtype=float
        )

        weights = np.array(
            [
                self.dynamic_weights[
                    key
                ]
                for key in
                self.feature_names
            ],
            dtype=float
        )

        return float(
            np.clip(
                np.dot(
                    values,
                    weights
                ),
                -1,
                1
            )
        )

    # ========================================================
    # UPDATE ML AFTER TRADE
    # ========================================================

    def update_model(
        self,
        features,
        outcome
    ):

        outcome = str(
            outcome
        ).upper()

        if outcome not in {
            "WIN",
            "LOSS"
        }:

            return

        self.ml_model.update(

            features,

            1
            if outcome == "WIN"
            else 0
        )

    # ========================================================
    # RECENT MOVE
    # ========================================================

    def _actual_move_pct(
        self,
        df
    ):

        close = (
            pd.to_numeric(
                df["Close"],
                errors="coerce"
            )
            .dropna()
        )

        if len(close) < 6:

            return 0.0

        return float(

            abs(
                close.iloc[-1] -
                close.iloc[-6]
            )

            /

            (
                abs(
                    close.iloc[-6]
                )
                +
                EPS
            )
        )

    # ========================================================
    # VOLATILITY
    # ========================================================

    def _volatility_ratio(
        self,
        df
    ):

        close = (
            pd.to_numeric(
                df["Close"],
                errors="coerce"
            )
            .dropna()
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        if len(returns) < (
            self.config.vol_window +
            5
        ):

            return 1.0

        short_vol = (
            returns
            .tail(5)
            .std()
        )

        long_vol = (
            returns
            .tail(
                self.config.vol_window
            )
            .std()
        )

        if (
            not np.isfinite(
                long_vol
            )
            or
            long_vol < EPS
        ):

            return 1.0

        return float(
            short_vol /
            long_vol
        )

    # ========================================================
    # TRADE LEVELS
    # ========================================================

    def _build_trade(
        self,
        direction,
        entry,
        stop_distance
    ):

        stop_distance = max(

            stop_distance,

            entry *
            self.config
            .min_stop_pct
        )

        stop_distance = min(

            stop_distance,

            entry *
            self.config
            .max_stop_pct
        )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if direction == "BUY":

            stop_loss = (
                entry -
                stop_distance
            )

            tp1 = (
                entry +
                stop_distance *
                self.config.rr_tp1
            )

            tp2 = (
                entry +
                stop_distance *
                self.config.rr_tp2
            )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        else:

            stop_loss = (
                entry +
                stop_distance
            )

            tp1 = (
                entry -
                stop_distance *
                self.config.rr_tp1
            )

            tp2 = (
                entry -
                stop_distance *
                self.config.rr_tp2
            )

        return {

            "ENTRY":
                float(entry),

            "SL":
                float(stop_loss),

            "TP1_1_2":
                float(tp1),

            "TP2_1_3":
                float(tp2),

            "RISK_DISTANCE":
                float(stop_distance),

            "RR_TP1":
                float(
                    self.config.rr_tp1
                ),

            "RR_TP2":
                float(
                    self.config.rr_tp2
                ),
        }

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    def calculate_all_signals(

        self,

        df,

        bids,

        asks,

        current_inventory=0,

        performance_history=None,

        prev_bids=None,

        prev_asks=None,

        manipulation_risk=0.0,

        liquidation_volumes=None,

        displayed_vol=0.0,

        cancelled_vol=0.0,

        time_exists=0.0,

        obs_window=1.0,

        open_interest=0.0,

        leverage=1.0
    ):

        df = pd.DataFrame(
            df
        )

        # ----------------------------------------------------
        # DATA CHECK
        # ----------------------------------------------------

        if (
            df.empty
            or
            "Close" not in df.columns
        ):

            return {

                "SIGNAL":
                    "NO_DATA",

                "CONFIDENCE":
                    0.0,

                "SCORE":
                    0.0,

                "REASON":
                    "Missing OHLC data"
            }

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        features = (
            self.extract_features(

                df,

                bids,

                asks,

                prev_bids,

                prev_asks,

                performance_history
            )
        )

        # ----------------------------------------------------
        # WEIGHTED RESEARCH SCORE
        # ----------------------------------------------------

        weighted_score = (
            self._weighted_score(
                features
            )
        )

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        ml_probability = (
            self.ml_model
            .probability(
                features
            )
        )

        ml_score = (
            ml_probability -
            0.5
        ) * 2.0

        # ----------------------------------------------------
        # ENSEMBLE
        # ----------------------------------------------------

        if (
            self.ml_model.ready
            and
            self.ml_model.samples
            >=
            self.config
            .min_training_samples
        ):

            final_score = (

                0.55 *
                weighted_score

                +

                0.45 *
                ml_score
            )

        else:

            final_score = (
                weighted_score
            )

        final_score = float(
            np.clip(
                final_score,
                -1,
                1
            )
        )

        confidence = abs(
            final_score
        )

        # ----------------------------------------------------
        # MOVE FILTER
        # ----------------------------------------------------

        move_pct = (
            self._actual_move_pct(
                df
            )
        )

        move_ok = (
            move_pct >=
            self.config
            .min_move_pct
        )

        # ----------------------------------------------------
        # VOLATILITY FILTER
        # ----------------------------------------------------

        volatility_ratio = (
            self._volatility_ratio(
                df
            )
        )

        volatility_ok = (

            self.config
            .min_vol_ratio

            <=

            volatility_ratio

            <=

            self.config
            .max_vol_ratio
        )

        # ----------------------------------------------------
        # DIRECTION
        # ----------------------------------------------------

        direction = (

            "BUY"
            if final_score > 0
            else
            "SELL"
        )

        # ----------------------------------------------------
        # OBI / OFI
        # ----------------------------------------------------

        obi = features[
            "BOOK_IMB"
        ]

        ofi = features[
            "OFI"
        ]

        buy_confirmation = (

            obi >=
            self.config
            .obi_threshold

            and

            ofi >=
            self.config
            .ofi_threshold
        )

        sell_confirmation = (

            obi <=
            -self.config
            .obi_threshold

            and

            ofi <=
            -self.config
            .ofi_threshold
        )

        if direction == "BUY":

            flow_confirmation = (
                buy_confirmation
            )

        else:

            flow_confirmation = (
                sell_confirmation
            )

        # ----------------------------------------------------
        # IMPULSE / EXHAUSTION
        # ----------------------------------------------------

        impulse = features[
            "IMPULSE"
        ]

        exhaustion = features[
            "EXHAUSTION"
        ]

        if direction == "BUY":

            impulse_confirmation = (
                impulse >= 0.25
            )

            exhaustion_bad = (

                exhaustion >=
                0.75

                and

                not
                buy_confirmation
            )

        else:

            impulse_confirmation = (
                impulse <= -0.25
            )

            exhaustion_bad = (

                exhaustion <=
                -0.75

                and

                not
                sell_confirmation
            )

        # ----------------------------------------------------
        # MANIPULATION
        # ----------------------------------------------------

        calculated_manipulation = (

            ManipulationRisk
            .score(

                displayed_vol,

                cancelled_vol,

                time_exists,

                obs_window
            )
        )

        manipulation = max(

            float(
                manipulation_risk
            ),

            calculated_manipulation
        )

        # ----------------------------------------------------
        # REASONS
        # ----------------------------------------------------

        reasons = []

        if not move_ok:

            reasons.append(
                "MOVE_BELOW_0.40_PERCENT"
            )

        if not flow_confirmation:

            reasons.append(
                "OBI_OFI_NOT_CONFIRMED"
            )

        if not volatility_ok:

            reasons.append(
                "BAD_VOLATILITY_REGIME"
            )

        if manipulation >= 0.70:

            reasons.append(
                "HIGH_MANIPULATION_RISK"
            )

        if exhaustion_bad:

            reasons.append(
                "EXTREME_UNCONFIRMED_MOVE"
            )

        if features[
            "REWARD_RISK"
        ] < -0.50:

            reasons.append(
                "WEAK_REWARD_RISK"
            )

        # ----------------------------------------------------
        # FINAL TRADE GATE
        # ----------------------------------------------------

        can_trade = (

            move_ok

            and

            flow_confirmation

            and

            volatility_ok

            and

            manipulation < 0.70

            and

            not exhaustion_bad

            and

            confidence >=
            self.config
            .min_confidence

            and

            (
                (
                    direction ==
                    "BUY"
                    and
                    final_score > 0
                )

                or

                (
                    direction ==
                    "SELL"
                    and
                    final_score < 0
                )
            )
        )

        if can_trade:

            signal = direction

        else:

            signal = "NO TRADE"

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        close = float(

            pd.to_numeric(
                df["Close"],
                errors="coerce"
            )
            .dropna()
            .iloc[-1]
        )

        # ----------------------------------------------------
        # ATR STOP
        # ----------------------------------------------------

        atr = (
            ResearchFeatures
            .atr(
                df,
                self.config
                .atr_period
            )
        )

        if atr <= 0:

            stop_distance = (

                close *
                self.config
                .min_stop_pct
            )

        else:

            stop_distance = (

                atr *
                self.config
                .atr_multiplier
            )

        trade = (
            self._build_trade(

                direction,

                close,

                stop_distance
            )
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {

            "SIGNAL":
                signal,

            "DIRECTION":
                direction,

            "SCORE":
                final_score,

            "CONFIDENCE":
                confidence,

            "ML_PROBABILITY":
                ml_probability,

            "ML_READY":
                self.ml_model.ready,

            "ML_SAMPLES":
                self.ml_model.samples,

            # -----------------------------------------------
            # MOVE
            # -----------------------------------------------

            "MOVE_5_CANDLE_PCT":
                move_pct * 100.0,

            "MIN_MOVE_PCT":
                self.config
                .min_move_pct *
                100.0,

            # -----------------------------------------------
            # MICROSTRUCTURE
            # -----------------------------------------------

            "OBI":
                obi,

            "OFI":
                ofi,

            # -----------------------------------------------
            # RISK FILTERS
            # -----------------------------------------------

            "IMPULSE":
                impulse,

            "EXHAUSTION":
                exhaustion,

            "VOLATILITY_RATIO":
                volatility_ratio,

            "MANIPULATION_RISK":
                manipulation,

            # -----------------------------------------------
            # FEATURES
            # -----------------------------------------------

            "FEATURES":
                features,

            "WEIGHTS":
                self.dynamic_weights,

            # -----------------------------------------------
            # TRADE LEVELS
            # -----------------------------------------------

            "ENTRY":
                trade["ENTRY"],

            "SL":
                trade["SL"],

            "TP1_1_2":
                trade["TP1_1_2"],

            "TP2_1_3":
                trade["TP2_1_3"],

            "RISK_DISTANCE":
                trade["RISK_DISTANCE"],

            "RR_TP1":
                trade["RR_TP1"],

            "RR_TP2":
                trade["RR_TP2"],

            # -----------------------------------------------
            # STATUS
            # -----------------------------------------------

            "CAN_TRADE":
                can_trade,

            "REASON":

                "OK"
                if can_trade
                else
                " | ".join(
                    reasons
                ),
        }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

PowerTradingRiskEngine = (
    PowerTradingRiskEngine
)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    config = SignalConfig()

    print(
        "===================================="
    )

    print(
        "Research Lab Engine Loaded"
    )

    print(
        "===================================="
    )

    print(
        "Minimum Move:",
        config.min_move_pct * 100,
        "%"
    )

    print(
        "TP1:",
        "1:",
        config.rr_tp1
    )

    print(
        "TP2:",
        "1:",
        config.rr_tp2
    )

    print(
        "OBI Threshold:",
        config.obi_threshold
    )

    print(
        "OFI Threshold:",
        config.ofi_threshold
    )

    print(
        "ML Training Samples:",
        config.min_training_samples
    )
