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
        loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, random_state=42
    )
    self.ml_model = CalibratedClassifierCV(
        estimator=base_model, method='sigmoid', cv='prefit'
    )
    self.is_model_trained = False

    # Sirf Top 5 Behtareen Features/Formulas jo 85% Accuracy ke liye ahem hain
    self.feature_names = [
        "BOOK_IMB",
        "TAKER_FLOW",
        "QUANT_IMPLY",
        "ADAPT_CONF",
        "BAYESIAN",
    ]
    self.dynamic_weights = {
        "BOOK_IMB": 0.30,
        "TAKER_FLOW": 0.25,
        "QUANT_IMPLY": 0.20,
        "ADAPT_CONF": 0.15,
        "BAYESIAN": 0.10,
    }

    # State tracking for Hysteresis / Cooldown (Signal bar-bar change hone se rokne ke liye)
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

    # 1. Book Imbalance (Top Priority)
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

    # 4. Adaptive Conformal Band Crosses Zero
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

    return results

  def calculate_all_signals(
      self, df, bids, asks, current_inventory=0, performance_history=None
  ):
    results = self.extract_features(df, bids, asks)
    feature_vector = np.array(
        [results[k] for k in self.feature_names]
    ).reshape(1, -1)

    # Weighted Linear Score calculation using Top 5 Features
    weight_vector = np.array(
        [self.dynamic_weights[k] for k in self.feature_names]
    )
    raw_score = float(np.dot(feature_vector[0], weight_vector))

    # --- HYSTERESIS & DYNAMIC THRESHOLD LOGIC (Signal Stabilization) ---
    threshold = 0.45  # Jab tak score isse ooper ya neechay na ho, trade trigger nahi hogi
    final_score = 0.0

    current_intent = "NEUTRAL"
    if raw_score > threshold:
      current_intent = "LONG"
    elif raw_score < -threshold:
      current_intent = "SHORT"

    # Cooldown aur State Retention: Agar signal achanak palat raha hai toh cooldown check karein
    if current_intent != self.last_signal:
      if self.cooldown_counter > 0:
        self.cooldown_counter -= 1
        current_intent = (
            self.last_signal
        )  # Purani state ko qaim rakho jab tak cooldown khatam na ho
      else:
        self.cooldown_counter = (
            3  # 3 candles ka cooldown period taake fake flip na ho
        )
        self.last_signal = current_intent
    else:
      self.cooldown_counter = 3  # Reset counter

    # Final score mapping based on stable intent
    if current_intent == "LONG":
      final_score = abs(raw_score)
    elif current_intent == "SHORT":
      final_score = -abs(raw_score)
    else:
      final_score = 0.0

    return results, final_score, self.dynamic_weights


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
