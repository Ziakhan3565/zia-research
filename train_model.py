import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Global counter to track closed trades for automatic retraining
TRADE_COUNTER = 0


def train_trend_aligned_model():
  print(
      "🚀 Training Trend-Aligned Microstructure Model with Fourier Trend &"
      " Advanced Formulas...\n"
  )

  try:
    df = pd.read_csv("market_data_log.csv")
  except FileNotFoundError:
    print("❌ 'market_data_log.csv' nahi mili!")
    return

  # 1. Feature Engineering & Custom Formulas
  df["bid_ask_ratio"] = df["top20_bid_sum"] / (df["top20_ask_sum"] + 1e-5)
  df["total_depth"] = df["top20_bid_sum"] + df["top20_ask_sum"]

  df["sma_20"] = (
      df.groupby("symbol")["current_price"]
      .transform(lambda x: x.rolling(20, min_periods=1).mean())
  )
  df["trend_signal"] = df["current_price"] - df["sma_20"]

  # Fallback if Volume is missing
  if "Volume" not in df.columns:
    df["Volume"] = 1.0

  df["delta_p"] = df.groupby("symbol")["current_price"].diff().fillna(0)
  df["returns"] = df.groupby("symbol")["current_price"].pct_change().fillna(0)
  df["realized_vol"] = (
      df.groupby("symbol")["returns"]
      .transform(lambda x: x.rolling(15, min_periods=1).std())
      + 1e-8
  )

  # 5 Advanced Research Lab Formulas
  df["BOOK_IMB"] = (df["top20_bid_sum"] - df["top20_ask_sum"]) / (
      df["top20_bid_sum"] + df["top20_ask_sum"] + 1e-8
  )

  taker_buy = df["Volume"] * np.where(df["delta_p"] > 0, 1.0, 0.3)
  taker_sell = df["Volume"] * np.where(df["delta_p"] <= 0, 1.0, 0.3)
  df["TAKER_FLOW"] = (taker_buy - taker_sell) / (
      taker_buy + taker_sell + 1e-8
  )

  depth_skew = (df["top20_bid_sum"] - df["top20_ask_sum"]) / (
      df["top20_bid_sum"] + df["top20_ask_sum"] + 1e-8
  )
  df["QUANT_IMPLY"] = np.clip(depth_skew * 1.5, -1, 1)

  ma_fast = (
      df.groupby("symbol")["current_price"]
      .transform(lambda x: x.rolling(3, min_periods=1).mean())
  )
  ma_slow = (
      df.groupby("symbol")["current_price"]
      .transform(lambda x: x.rolling(10, min_periods=1).mean())
  )
  df["ADAPT_CONF"] = np.clip(
      (ma_fast - ma_slow)
      / (df["realized_vol"] * df["current_price"] + 1e-8),
      -1,
      1,
  )

  prior = 0.745
  likelihood = np.where(df["BOOK_IMB"] > 0, 1.0, 0.25)
  posterior = (likelihood * prior) / (
      (likelihood * prior) + ((1 - likelihood) * (1 - prior)) + 1e-8
  )
  df["BAYESIAN"] = np.clip((posterior - 0.5) * 2.0, -1, 1)

  # --- 2. Fourier Trend Feature Calculation per symbol ---
  def compute_fourier_trend(sub_df):
    prices = sub_df["current_price"].values
    if len(prices) < 15:
      return pd.Series(0.0, index=sub_df.index)
    xc = prices - np.mean(prices)
    fft_vals = np.fft.fft(xc)
    num_keep = max(1, int(len(fft_vals) * 0.15))
    fft_masked = np.zeros_like(fft_vals)
    fft_masked[:num_keep] = fft_vals[:num_keep]
    fft_masked[-num_keep:] = fft_vals[-num_keep:]
    trend_curve = np.real(np.fft.ifft(fft_masked))
    diffs = np.gradient(trend_curve)
    return pd.Series(diffs, index=sub_df.index)

  df["FOURIER_TREND"] = (
      df.groupby("symbol")
      .apply(compute_fourier_trend)
      .reset_index(level=0, drop=True)
  )

  features = [
      "top20_bid_sum",
      "top20_ask_sum",
      "obi_top20",
      "spread",
      "bid_ask_ratio",
      "total_depth",
      "trend_signal",
      "BOOK_IMB",
      "TAKER_FLOW",
      "QUANT_IMPLY",
      "ADAPT_CONF",
      "BAYESIAN",
      "FOURIER_TREND",  # Added Fourier Trend feature
  ]

  df = df.dropna(subset=features + ["current_price", "symbol"]).copy()

  # Target: Aglay 5 steps mein price move
  df["future_price"] = df.groupby("symbol")["current_price"].shift(-5)
  df = df.dropna(subset=["future_price"]).copy()

  df["target"] = (df["future_price"] > df["current_price"]).astype(int)

  X = df[features]
  y = df["target"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  model = XGBClassifier(
      n_estimators=100,
      learning_rate=0.03,
      max_depth=4,
      random_state=42,
      eval_metric="logloss",
  )
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)

  print(
      f"🎯 Model Accuracy with Fourier Trend & Advanced Formulas: {acc * 100:.2f}%"
  )

  joblib.dump(model, "xgboost_obi_model.pkl")
  print("💾 Model saved successfully as 'xgboost_obi_model.pkl'!")


def record_trade_outcome_and_check_retrain(outcome):
  """Yeh function har trade close hone par call hoga (outcome: "WIN" ya "LOSS").

  Jaise hi 20 trades poori hongi, yeh khud ba khud model ko retrain kar dega.
  """
  global TRADE_COUNTER

  if outcome in ["WIN", "LOSS"]:
    TRADE_COUNTER += 1
    print(f"📊 Closed Trades Tracked: {TRADE_COUNTER}/20")

    if TRADE_COUNTER >= 20:
      print(
          "🔄 20 Trades target reached! Automatically retraining model with"
          " fresh data..."
      )
      try:
        train_trend_aligned_model()
        print("✅ Model successfully retrained & updated after 20 trades!")
      except Exception as e:
        print(f"❌ Retraining mein error aaya: {e}")

      # Counter reset for the next 20 trades
      TRADE_COUNTER = 0


if __name__ == "__main__":
  train_trend_aligned_model()
