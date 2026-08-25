import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


class TenPaperResearchLab:

    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        self.scaler = StandardScaler()

        # Online Machine Learning Classifier
        base_model = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=0.0001, max_iter=1000, random_state=42
        )
        self.ml_model = CalibratedClassifierCV(
            estimator=base_model, method="sigmoid", cv="prefit"
        )
        self.is_model_trained = False

        # Total 6 Features including Fourier Trend
        self.feature_names = [
            "BOOK_IMB",
            "TAKER_FLOW",
            "QUANT_IMPLY",
            "ADAPT_CONF",
            "BAYESIAN",
            "FOURIER_TREND",
        ]
        self.dynamic_weights = {
            "BOOK_IMB": 0.25,
            "TAKER_FLOW": 0.20,
            "QUANT_IMPLY": 0.15,
            "ADAPT_CONF": 0.15,
            "BAYESIAN": 0.10,
            "FOURIER_TREND": 0.15,
        }

        # State tracking for Hysteresis / Cooldown
        self.last_signal = "NEUTRAL"
        self.cooldown_counter = 0

    def extract_features(self, df, bids, asks):
        results = {}
        if len(bids) == 0 or len(asks) == 0 or df.empty or len(df) < 15:
            return {k: 0.0 for k in self.feature_names}

        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        mid_price = (bids[0, 0] + asks[0, 0]) / 2
        returns = df["Close"].pct_change().dropna()
        realized_vol = returns.std() + 1e-8
        delta_p = df["Close"].iloc[-1] - df["Close"].iloc[-2]

        # 1. Book Imbalance
        results["BOOK_IMB"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        # 2. Taker Flow
        taker_buy = df["Volume"].iloc[-1] * (1.0 if delta_p > 0 else 0.3)
        taker_sell = df["Volume"].iloc[-1] * (1.0 if delta_p <= 0 else 0.3)
        results["TAKER_FLOW"] = (taker_buy - taker_sell) / (
            taker_buy + taker_sell + 1e-8
        )

        # 3. Quantities Imply (Depth Skew)
        depth_skew = (bids[0, 1] - asks[0, 1]) / (bids[0, 1] + asks[0, 1] + 1e-8)
        results["QUANT_IMPLY"] = np.clip(depth_skew * 1.5, -1, 1)

        # 4. Adaptive Conformal Band
        ma_fast = df["Close"].rolling(3).mean().iloc[-1]
        ma_slow = df["Close"].rolling(10).mean().iloc[-1]
        results["ADAPT_CONF"] = np.clip(
            (ma_fast - ma_slow) / (realized_vol * mid_price + 1e-8), -1, 1
        )

        # 5. Bayesian Probability
        prior = 0.745
        likelihood = 1.0 if results["BOOK_IMB"] > 0 else 0.25
        posterior = (likelihood * prior) / (
            (likelihood * prior) + ((1 - likelihood) * (1 - prior)) + 1e-8
        )
        results["BAYESIAN"] = np.clip((posterior - 0.5) * 2.0, -1, 1)

        # 6. Fourier Series / IFFT Trend Curve Feature
        prices = df["Close"].values
        xc = prices - np.mean(prices)
        fft_vals = np.fft.fft(xc)
        num_keep = max(1, int(len(fft_vals) * 0.15))
        fft_masked = np.zeros_like(fft_vals)
        fft_masked[:num_keep] = fft_vals[:num_keep]
        fft_masked[-num_keep:] = fft_vals[-num_keep:]
        trend_curve = np.real(np.fft.ifft(fft_masked))
        
        current_trend_val = trend_curve[-1] - trend_curve[-2]
        results["FOURIER_TREND"] = np.clip(
            current_trend_val / (realized_vol * mid_price + 1e-8), -1, 1
        )

        return results

    def calculate_fourier_sl_tp(self, df, current_price, signal_intent):
        """
        Fourier Cycle Peaks/Lows + Hard Safety Cap based SL & TP Calculator
        """
        prices = df['current_price'].values if 'current_price' in df.columns else df['Close'].values
        if len(prices) < 15:
            # Fallback agar data kam ho
            tp = current_price * (1.0 + 0.006)
            sl = current_price * (1.0 - 0.004)
            return sl, tp

        xc = prices - np.mean(prices)
        fft_vals = np.fft.fft(xc)
        num_keep = max(1, int(len(fft_vals) * 0.15))
        fft_masked = np.zeros_like(fft_vals)
        fft_masked[:num_keep] = fft_vals[:num_keep]
        fft_masked[-num_keep:] = fft_vals[-num_keep:]
        trend_curve = np.real(np.fft.ifft(fft_masked))

        cycle_low = np.min(trend_curve)
        cycle_peak = np.max(trend_curve)
        
        # Relative scaling ya absolute fallback if curve is flat
        noise_buffer = 0.001 * current_price
        max_risk_cap = 0.006  # 0.6% Hard Safety Cap

        if signal_intent == "LONG":
            fourier_sl = current_price - abs(current_price - cycle_low) - noise_buffer
            hard_sl = current_price * (1.0 - max_risk_cap)
            final_sl = max(fourier_sl, hard_sl) # Jo entry ke zyada kareeb aur safe ho
            final_tp = max(current_price + abs(cycle_peak - current_price), current_price + 1.5 * (current_price - final_sl))
        elif signal_intent == "SHORT":
            fourier_sl = current_price + abs(cycle_peak - current_price) + noise_buffer
            hard_sl = current_price * (1.0 + max_risk_cap)
            final_sl = min(fourier_sl, hard_sl)
            final_tp = min(current_price - abs(current_price - cycle_low), current_price - 1.5 * (final_sl - current_price))
        else:
            final_sl = current_price * (1.0 - 0.004)
            final_tp = current_price * (1.0 + 0.006)

        return round(float(final_sl), 4), round(float(final_tp), 4)

    def calculate_all_signals(
        self, df, bids, asks, current_inventory=0, performance_history=None
    ):
        results = self.extract_features(df, bids, asks)
        feature_vector = np.array(
            [results[k] for k in self.feature_names]
        ).reshape(1, -1)

        weight_vector = np.array(
            [self.dynamic_weights[k] for k in self.feature_names]
        )
        raw_score = float(np.dot(feature_vector[0], weight_vector))

        # --- HYSTERESIS & DYNAMIC THRESHOLD LOGIC ---
        threshold = 0.45
        final_score = 0.0

        current_intent = "NEUTRAL"
        if raw_score > threshold:
            current_intent = "LONG"
        elif raw_score < -threshold:
            current_intent = "SHORT"

        if current_intent != self.last_signal:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                current_intent = self.last_signal
            else:
                self.cooldown_counter = 3
                self.last_signal = current_intent
        else:
            self.cooldown_counter = 3

        # Current Price extract karna SL/TP ke liye
        current_price = df["Close"].iloc[-1] if not df.empty else 0.0
        stop_loss, take_profit = self.calculate_fourier_sl_tp(df, current_price, current_intent)

        if current_intent == "LONG":
            final_score = abs(raw_score)
        elif current_intent == "SHORT":
            final_score = -abs(raw_score)
        else:
            final_score = 0.0

        # Signals dictionary ya tuple mein ab SL aur TP bhi return honge
        signal_payload = {
            "intent": current_intent,
            "score": final_score,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

        return results, signal_payload, self.dynamic_weights


class PowerTradingRiskEngine:

    def __init__(self):
        pass

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
        total_ltz = (
            np.sum(liquidation_volumes) if len(liquidation_volumes) > 0 else 0.0
        )
        max_ltz = np.max(liquidation_volumes) if len(liquidation_volumes) > 0 else 0.0
        ltz_score = (max_ltz / (total_ltz + 1e-8)) * 100

        spoof_ratio = cancelled_vol / (displayed_vol + 1e-8)
        persistence = min(max(time_exists / (obs_window + 1e-8), 0), 1)
        spoof_score = spoof_ratio * (1 - persistence)

        squeeze_risk = total_ltz * open_interest * leverage * volatility
        market_risk = ltz_score + spoof_score + squeeze_risk

        return {
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze_risk,
            "Market_Risk": market_risk,
        }
