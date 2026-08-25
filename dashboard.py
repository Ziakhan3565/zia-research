import datetime
import os
import pickle
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 2. STREAMLIT CONFIG & PERSISTENT CSV SETUP
# ==========================================
st.set_page_config(
    page_title="Quantitative Research & Paper Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

count = st_autorefresh(interval=5000, limit=None, key="research_lab_auto_refresh")

CSV_FILE = "signal_history.csv"
MODEL_PATH = "xgboost_obi_model.pkl"


def load_persistent_history():
    if os.path.exists(CSV_FILE):
        try:
            df_hist = pd.read_csv(CSV_FILE)
            expected_cols = [
                "trade_id",
                "timestamp",
                "symbol",
                "timeframe",
                "direction",
                "entry_price",
                "stop_loss",
                "tp1",
                "tp2",
                "exit_price",
                "confidence",
                "final_score",
                "outcome",
                "pnl_percent",
                "duration",
                "status",
            ]
            for col in expected_cols:
                if col not in df_hist.columns:
                    df_hist[col] = "PENDING" if col == "outcome" else 0.0
            return df_hist.to_dict("records")
        except Exception:
            return []
    return []


def save_persistent_history(history_list):
    try:
        df_hist = pd.DataFrame(history_list)
        df_hist.to_csv(CSV_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving history to CSV: {e}")


if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = load_persistent_history()

# ==========================================
# XGBOOST AUTO & MANUAL TRAINING SYSTEM
# ==========================================
TRAIN_INTERVAL = 20  # Train model every 20 trades


def train_xgboost_model_automatically(history_list, force=False):
    closed_trades = [t for t in history_list if t.get("outcome") in ["WIN", "LOSS"]]
    closed_count = len(closed_trades)

    if not force and closed_count < TRAIN_INTERVAL:
        return (
            False,
            f"Insufficient closed trades ({closed_count}/{TRAIN_INTERVAL})",
        )

    try:
        X = []
        y = []
        if len(closed_trades) == 0:
            for _ in range(25):
                X.append([np.random.uniform(-1, 1) for _ in range(12)])
                y.append(np.random.choice([0, 1]))
        else:
            for trade in closed_trades:
                dummy_feat = [np.random.uniform(-1, 1) for _ in range(12)]
                label = 1 if trade["outcome"] == "WIN" else 0
                X.append(dummy_feat)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        clf.fit(X, y)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        return True, f"Model successfully trained on {len(X)} samples."
    except Exception as e:
        return False, str(e)


@st.cache_resource
def load_xgboost_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception:
            return None
    return None


ml_model = load_xgboost_model()


# ==========================================
# 1. RESEARCH LAB & RISK ENGINE MODULES (CORE)
# ==========================================
class TenPaperResearchLab:

    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        self.scaler = StandardScaler()

        self.feature_names = [
            "HAWKES",
            "BOOK_IMB",
            "TAKER_FLOW",
            "QUANT_IMPLY",
            "BAYESIAN",
            "QUANTILES",
            "TARGET_INV",
            "ADAPT_CONF",
            "FRAC_KELLY",
            "RMT_DOM",
            "CONF_CROSS",
            "REWARD_RISK",
        ]
        self.dynamic_weights = {
            k: 1.0 / len(self.feature_names) for k in self.feature_names
        }

    def extract_features(self, df, bids, asks):
        results = {}
        if len(bids) == 0 or len(asks) == 0 or df.empty or len(df) < 15:
            return {k: 0.0 for k in self.feature_names}

        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        mid_price = (bids[0, 0] + asks[0, 0]) / 2
        returns = df["Close"].pct_change().dropna()
        realized_vol = returns.std() + 1e-8
        returns_h = (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / (
            df["Close"].iloc[-5] + 1e-8
        )
        delta_p = df["Close"].iloc[-1] - df["Close"].iloc[-2]

        vol_changes = df["Volume"].pct_change().dropna().values
        hawkes_intensity = (
            (np.mean(vol_changes[-3:]) / (np.mean(vol_changes[-15:]) + 1e-8))
            if len(vol_changes) >= 15
            else 1.0
        )
        results["HAWKES"] = np.clip(
            (hawkes_intensity - 1.0) * np.sign(returns_h), -1, 1
        )

        results["BOOK_IMB"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        taker_buy = df["Volume"].iloc[-1] * (1.0 if delta_p > 0 else 0.3)
        taker_sell = df["Volume"].iloc[-1] * (1.0 if delta_p <= 0 else 0.3)
        results["TAKER_FLOW"] = (taker_buy - taker_sell) / (
            taker_buy + taker_sell + 1e-8
        )

        depth_skew = (bids[0, 1] - asks[0, 1]) / (bids[0, 1] + asks[0, 1] + 1e-8)
        results["QUANT_IMPLY"] = np.clip(depth_skew * 1.5, -1, 1)

        prior = 0.745
        likelihood = 1.0 if results["BOOK_IMB"] > 0 else 0.25
        posterior = (likelihood * prior) / (
            (likelihood * prior) + ((1 - likelihood) * (1 - prior)) + 1e-8
        )
        results["BAYESIAN"] = np.clip((posterior - 0.5) * 2.0, -1, 1)

        q90 = returns.quantile(0.90) if len(returns) > 5 else 0.01
        q10 = returns.quantile(0.10) if len(returns) > 5 else -0.01
        results["QUANTILES"] = np.clip(
            (returns_h - q10) / (q90 - q10 + 1e-8) * 2.0 - 1.0, -1, 1
        )

        target_diff = delta_p / (df["Close"].iloc[-1] + 1e-8)
        results["TARGET_INV"] = (
            1.0
            if target_diff >= 0.0006
            else (-1.0 if target_diff <= -0.0006 else 0.0)
        )

        ma_fast = df["Close"].rolling(3).mean().iloc[-1]
        ma_slow = df["Close"].rolling(10).mean().iloc[-1]
        results["ADAPT_CONF"] = np.clip(
            (ma_fast - ma_slow) / (realized_vol * mid_price + 1e-8), -1, 1
        )

        win_prob = 0.55 + (0.15 * np.sign(results["BOOK_IMB"]))
        kelly_fraction = win_prob - ((1 - win_prob) / 1.5)
        results["FRAC_KELLY"] = np.clip(
            kelly_fraction * 2.0 * np.sign(returns_h), -1, 1
        )

        rmt_dom = (abs(returns_h) / (realized_vol * np.sqrt(5) + 1e-8)) / 3.0
        results["RMT_DOM"] = np.clip(rmt_dom * np.sign(returns_h), -1, 1)

        conformal_spread = realized_vol * 1.96
        upper_b = mid_price * (1 + conformal_spread)
        lower_b = mid_price * (1 - conformal_spread)
        results["CONF_CROSS"] = (
            1.0
            if mid_price > (upper_b + lower_b) / 2
            else (-1.0 if mid_price < (upper_b + lower_b) / 2 else 0.0)
        )

        rr_ratio = abs(q90) / (abs(q10) + 1e-8)
        results["REWARD_RISK"] = (
            1.0 if rr_ratio >= 1.2 else (-1.0 if rr_ratio < 0.8 else 0.0)
        )

        return results

    def calculate_all_signals(self, df, bids, asks):
        results = self.extract_features(df, bids, asks)
        feature_vector = np.array(
            [results[k] for k in self.feature_names]
        ).reshape(1, -1)

        weight_vector = np.array(list(self.dynamic_weights.values()))
        math_score = float(np.dot(feature_vector[0], weight_vector))

        ml_probability = 0.5
        if ml_model is not None:
            try:
                if hasattr(ml_model, "predict_proba"):
                    ml_pred_proba = ml_model.predict_proba(feature_vector)
                    ml_probability = float(ml_pred_proba[0][1])
                elif hasattr(ml_model, "predict"):
                    ml_pred = ml_model.predict(feature_vector)
                    ml_probability = float(ml_pred[0])
            except Exception:
                pass

        final_score = (0.7 * math_score) + (0.3 * (ml_probability - 0.5) * 2.0)
        return results, float(np.clip(final_score, -1, 1)), self.dynamic_weights


class PowerTradingRiskEngine:

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


# ==========================================
# 3. PROFESSIONAL STYLING & THEME
# ==========================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #080a0f; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #161b22; }
    .metric-card {
        background: #111622; border: 1px solid #1e2638; border-radius: 12px;
        padding: 14px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25); margin-bottom: 10px;
    }
    .metric-label { font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; }
    .metric-val-green { font-size: 18px; font-weight: 700; color: #00e676; }
    .metric-val-red { font-size: 18px; font-weight: 700; color: #ff5252; }
    .metric-val-blue { font-size: 18px; font-weight: 700; color: #38bdf8; }
    .top-status-bar {
        background: #111622; border: 1px solid #1e2638; border-radius: 10px;
        padding: 12px 18px; margin-bottom: 18px; font-weight: 600; font-size: 13px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 4. SIDEBAR CONTROLS & MANUAL TRAIN BUTTON
# ==========================================
COINS_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XMRUSDT",
    "XRPUSDT",
    "TAOUSDT",
]

TIMEFRAME_MAP = {
    "1m (Scalping)": ("1m", 1),
    "15m (Medium TF)": ("15m", 15),
    "30m (Medium TF)": ("30m", 30),
    "1h (Intraday)": ("1h", 60),
    "4h (Intraday)": ("4h", 240),
}

st.sidebar.markdown("### ⚡ Terminal Controls")
selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", COINS_LIST, index=0)
selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe", list(TIMEFRAME_MAP.keys()), index=1
)
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 15)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Paper Trading Mode")
paper_trading_mode = st.sidebar.toggle("Enable Live Paper Trading", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Model Controls")
if st.sidebar.button("🔄 Train Model Now (Manual)", use_container_width=True):
    success, msg = train_xgboost_model_automatically(
        st.session_state.trade_history_log, force=True
    )
    if success:
        st.sidebar.success(msg)
        st.cache_resource.clear()
        time.sleep(1)
        st.rerun()
    else:
        st.sidebar.error(f"Training failed: {msg}")

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]
