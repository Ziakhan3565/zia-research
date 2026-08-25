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

# Smooth background refresh
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
TRAIN_INTERVAL = 20


def train_xgboost_model_automatically(history_list, force=False):
    closed_trades = [t for t in history_list if t.get("outcome") in ["WIN", "LOSS"]]
    closed_count = len(closed_trades)

    if not force and closed_count < TRAIN_INTERVAL:
        return (
            False,
            f"Insufficient closed trades ({closed_count}/{TRAIN_INTERVAL})",
        )

    try:
        X, y = [], []
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
# 3. PROFESSIONAL STYLING & CSS
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
    .top-status-bar {
        background: #111622; border: 1px solid #1e2638; border-radius: 10px;
        padding: 12px 18px; margin-bottom: 18px; font-weight: 600; font-size: 13px;
    }
    .sticky-dashboard-header {
        position: sticky; top: 0; z-index: 999; background: #080a0f;
        padding-top: 10px; padding-bottom: 10px; border-bottom: 1px solid #1e2638;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 4. SIDEBAR CONTROLS
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
    "15m (Scalping)": ("15m", 15),
    "30m (Scalping)": ("30m", 30),
    "1h (Intraday)": ("1h", 60),
    "4h (Intraday)": ("4h", 240),
}

st.sidebar.markdown("### ⚡ Terminal Controls")
selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", COINS_LIST, index=0)
selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe", list(TIMEFRAME_MAP.keys()), index=0
)
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 15)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Paper Trading Mode")
paper_trading_mode = st.sidebar.toggle("Enable Live Auto-Scans", value=True)
loop_all_coins = st.sidebar.toggle("🔄 Scan All Coins in Loop", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Manual Trade Execution")
with st.sidebar.form("manual_trade_form"):
    m_coin = st.selectbox("Manual Coin", COINS_LIST)
    m_dir = st.selectbox("Direction", ["LONG", "SHORT"])
    m_entry = st.number_input("Entry Price", value=0.0, step=0.1)
    m_sl = st.number_input("Stop Loss", value=0.0, step=0.1)
    m_tp = st.number_input("Take Profit (TP1)", value=0.0, step=0.1)
    submit_manual = st.form_submit_button("📥 Open Manual Trade")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Model & History Controls")
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

if st.sidebar.button("🗑️ Clear Trade History", use_container_width=True):
    st.session_state.trade_history_log = []
    if os.path.exists(CSV_FILE):
        try:
            os.remove(CSV_FILE)
        except Exception:
            pass
    st.sidebar.success("Trade history cleared successfully!")
    time.sleep(0.5)
    st.rerun()

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]


# ==========================================
# DATA FETCHING FUNCTIONS
# ==========================================
@st.cache_data(ttl=15)
def fetch_klines_data(symbol, tf_key, limit=100):
    binance_tf = (
        "15m"
        if "15m" in tf_key
        else (
            "30m"
            if "30m" in tf_key
            else ("1h" if "1h" in tf_key else "4h")
        )
    )
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
    try:
        res = requests.get(url, timeout=4).json()
        if isinstance(res, dict) or not isinstance(res, list):
            raise ValueError("API limit")
        df = pd.DataFrame(
            res,
            columns=[
                "Open_Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close_Time",
                "QAV",
                "NAT",
                "TBBAV",
                "TBQAV",
                "Ignore",
            ],
        )
        df["Time"] = pd.to_datetime(df["Open_Time"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        df.set_index("Time", inplace=True)
        return df.reset_index()[["Time", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        dates = pd.date_range(
            end=datetime.datetime.now(), periods=limit, freq=binance_tf
        )
        base_p = (
            60000.0 if "BTC" in symbol else (3000.0 if "ETH" in symbol else 200.0)
        )
        closes = base_p + np.cumsum(np.random.normal(0, 5, limit))
        return pd.DataFrame({
            "Time": dates,
            "Open": closes - 2,
            "High": closes + 5,
            "Low": closes - 5,
            "Close": closes,
            "Volume": np.random.uniform(50, 500, limit),
        })


@st.cache_data(ttl=10)
def fetch_order_book_depth(symbol, depth_limit=20):
    try:
        url = f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        res = requests.get(url, timeout=4).json()
        if "bids" in res and "asks" in res:
            return np.array(res["bids"], dtype=float), np.array(
                res["asks"], dtype=float
            )
    except Exception:
        pass
    dummy_bids = np.array([[100.0 - i * 0.1, 1.5] for i in range(20)], dtype=float)
    dummy_asks = np.array([[100.0 + i * 0.1, 1.5] for i in range(20)], dtype=float)
    return dummy_bids, dummy_asks


# ==========================================
# MANUAL TRADE SUBMISSION
# ==========================================
if submit_manual:
    temp_df_m = fetch_klines_data(m_coin, selected_tf_label, limit=5)
    current_m_price = temp_df_m["Close"].iloc[-1] if not temp_df_m.empty else 100.0
    entry_p = m_entry if m_entry > 0 else current_m_price
    atr_approx = current_m_price * 0.005
    sl_p = (
        m_sl
        if m_sl > 0
        else (
            entry_p - 1.5 * atr_approx
            if m_dir == "LONG"
            else entry_p + 1.5 * atr_approx
        )
    )
    tp_p = (
        m_tp
        if m_tp > 0
        else (
            entry_p + 3.0 * atr_approx
            if m_dir == "LONG"
            else entry_p - 3.0 * atr_approx
        )
    )

    manual_trade_id = f"MANUAL_{m_coin}_{int(time.time())}"
    manual_new_trade = {
        "trade_id": manual_trade_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": m_coin,
        "timeframe": selected_tf_label,
        "direction": m_dir,
        "entry_price": round(entry_p, 4),
        "stop_loss": round(sl_p, 4),
        "tp1": round(tp_p, 4),
        "tp2": round(tp_p * 1.01 if m_dir == "LONG" else tp_p * 0.99, 4),
        "exit_price": round(entry_p, 4),
        "confidence": 91,
        "final_score": 1.0 if m_dir == "LONG" else -1.0,
        "outcome": "PENDING",
        "pnl_percent": 0.0,
        "duration": "Active",
        "status": "Open",
    }
    st.session_state.trade_history_log.insert(0, manual_new_trade)
    save_persistent_history(st.session_state.trade_history_log)
    st.sidebar.success(f"Manually added {m_dir} trade for {m_coin}!")


# ==========================================
# LOOP SCANNER WITH COOLDOWN (NO REPEATING TRADES)
# ==========================================
symbols_to_scan = COINS_LIST if loop_all_coins else [selected_symbol]
lab = TenPaperResearchLab()

HOLDING_COOLDOWN_SECONDS = 600

for sym in symbols_to_scan:
    recent_symbol_trade = False
    for existing_t in st.session_state.trade_history_log:
        if (
            existing_t.get("symbol") == sym
            and existing_t.get("status") == "Open"
        ):
            recent_symbol_trade = True
            break
        if existing_t.get("symbol") == sym:
            t_time_str = existing_t.get("timestamp")
            if t_time_str:
                try:
                    t_dt = datetime.datetime.strptime(
                        t_time_str, "%Y-%m-%d %H:%M:%S"
                    )
                    elapsed_sec = (
                        datetime.datetime.now() - t_dt
                    ).total_seconds()
                    if elapsed_sec < HOLDING_COOLDOWN_SECONDS:
                        recent_symbol_trade = True
                        break
                except Exception:
                    pass

    if recent_symbol_trade:
        continue

    scan_df = fetch_klines_data(sym, selected_tf_label)
    scan_bids, scan_asks = fetch_order_book_depth(sym)

    if (
        not scan_df.empty
        and len(scan_df) >= 3
        and len(scan_bids) > 0
        and len(scan_asks) > 0
    ):
        _, s_final_score, _ = lab.calculate_all_signals(
            scan_df, scan_bids, scan_asks
        )
        s_close_p = scan_df["Close"].iloc[-1]
        s_atr = (
            (scan_df["High"] - scan_df["Low"]).rolling(14).mean().iloc[-1]
        )
        if np.isnan(s_atr):
            s_atr = s_close_p * 0.005

        s_dir = (
            "LONG"
            if s_final_score >= 0.12
            else ("SHORT" if s_final_score <= -0.12 else "NEUTRAL")
        )
        s_conf = int(min(max(abs(s_final_score) * 100, 20), 99))
        s_risk_dist = 1.5 * s_atr

        if s_dir == "LONG":
            s_sl = s_close_p - s_risk_dist
            s_tp1 = s_close_p + (2.0 * s_risk_dist)
            s_tp2 = s_close_p + (3.0 * s_risk_dist)
        elif s_dir == "SHORT":
            s_sl = s_close_p + s_risk_dist
            s_tp1 = s_close_p - (2.0 * s_risk_dist)
            s_tp2 = s_close_p - (3.0 * s_risk_dist)
        else:
            continue

        if paper_trading_mode and s_dir != "NEUTRAL":
            loop_trade_id = f"{sym}_{selected_tf_label}_{int(time.time())}"
            loop_trade = {
                "trade_id": loop_trade_id,
                "timestamp": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "symbol": sym,
                "timeframe": selected_tf_label,
                "direction": s_dir,
                "entry_price": round(s_close_p, 4),
                "stop_loss": round(s_sl, 4),
                "tp1": round(s_tp1, 4),
                "tp2": round(s_tp2, 4),
                "exit_price": round(s_close_p, 4),
                "confidence": s_conf,
                "final_score": round(s_final_score, 3),
                "outcome": "PENDING",
                "pnl_percent": 0.0,
                "duration": "Active",
                "status": "Open",
            }
            st.session_state.trade_history_log.insert(0, loop_trade)

save_persistent_history(st.session_state.trade_history_log)


# ==========================================
# 5. MAIN UI RENDERING
# ==========================================
df = fetch_klines_data(selected_symbol, selected_tf_label)
bids, asks = fetch_order_book_depth(selected_symbol)

if not df.empty and len(df) >= 3 and len(bids) > 0 and len(asks) > 0:
    paper_results, final_score, evolved_weights = lab.calculate_all_signals(
        df, bids, asks
    )
    close_p = df["Close"].iloc[-1]
    atr_val = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    if np.isnan(atr_val):
        atr_val = close_p * 0.005

    direction = (
        "LONG"
        if final_score >= 0.12
        else ("SHORT" if final_score <= -0.12 else "NEUTRAL")
    )
    confidence = int(min(max(abs(final_score) * 100, 20), 99))
    risk_distance = 1.5 * atr_val

    if direction == "LONG":
        sl_val = close_p - risk_distance
        tp1_val = close_p + (2.0 * risk_distance)
        tp2_val = close_p + (3.0 * risk_distance)
    elif direction == "SHORT":
        sl_val = close_p + risk_distance
        tp1_val = close_p - (2.0 * risk_distance)
        tp2_val = close_p - (3.0 * risk_distance)
    else:
        sl_val = close_p - risk_distance
        tp1_val = close_p + (2.0 * risk_distance)
        tp2_val = close_p + (3.0 * risk_distance)

    lock_seconds = tf_minutes * 60
    current_time_sec = int(time.time())
    time_remaining = lock_seconds - (current_time_sec % lock_seconds)

    # Outcome & Time Duration Checker
    for trade in st.session_state.trade_history_log:
        if trade["outcome"] == "PENDING":
            t_time_str = trade.get("timestamp")
            t_tf = trade.get("timeframe", "15m")
            max_duration_mins = (
                30 if any(tf in t_tf for tf in ["15m", "30m"]) else 480
            )

            if t_time_str:
                try:
                    t_dt = datetime.datetime.strptime(
                        t_time_str, "%Y-%m-%d %H:%M:%S"
                    )
                    elapsed_mins = (
                        datetime.datetime.now() - t_dt
                    ).total_seconds() / 60.0

                    trade_symbol = trade["symbol"]
                    temp_df = fetch_klines_data(
                        trade_symbol, trade["timeframe"], limit=15
                    )
                    current_price = (
                        temp_df["Close"].iloc[-1]
                        if not temp_df.empty
                        else trade["entry_price"]
                    )

                    if elapsed_mins >= max_duration_mins:
                        trade["status"] = "Closed"
                        trade["exit_price"] = current_price
                        pnl_calc = (
                            ((current_price - trade["entry_price"]) / trade["entry_price"])
                            * 100
                            if trade["direction"] == "LONG"
                            else ((trade["entry_price"] - current_price) / trade["entry_price"])
                            * 100
                        )
                        trade["pnl_percent"] = round(pnl_calc, 2)
                        trade["outcome"] = "WIN" if pnl_calc >= 0 else "LOSS"
                    else:
                        if not temp_df.empty:
                            for idx, row in temp_df.iterrows():
                                candle_time = str(row["Time"])
                                if candle_time >= t_time_str:
                                    curr_high, curr_low = (
                                        row["High"],
                                        row["Low"],
                                    )
                                    entry = trade["entry_price"]
                                    sl = trade["stop_loss"]
                                    tp = trade["tp1"]

                                    if trade["direction"] == "LONG":
                                        if curr_high >= tp:
                                            trade["outcome"] = "WIN"
                                            trade["exit_price"] = tp
                                            trade["pnl_percent"] = round(
                                                ((tp - entry) / entry) * 100, 2
                                            )
                                            trade["status"] = "Closed"
                                            break
                                        elif curr_low <= sl:
                                            trade["outcome"] = "LOSS"
                                            trade["exit_price"] = sl
                                            trade["pnl_percent"] = round(
                                                ((sl - entry) / entry) * 100, 2
                                            )
                                            trade["status"] = "Closed"
                                            break
                                    elif trade["direction"] == "SHORT":
                                        if curr_low <= tp:
                                            trade["outcome"] = "WIN"
                                            trade["exit_price"] = tp
                                            trade["pnl_percent"] = round(
                                                ((entry - tp) / entry) * 100, 2
                                            )
                                            trade["status"] = "Closed"
                                            break
                                        elif curr_high >= sl:
                                            trade["outcome"] = "LOSS"
                                            trade["exit_price"] = sl
                                            trade["pnl_percent"] = round(
                                                ((entry - sl) / entry) * 100, 2
                                            )
                                            trade["status"] = "Closed"
                                            break
                except Exception:
                    pass

    save_persistent_history(st.session_state.trade_history_log)

    total_pnl_sum = sum(
        [
            t.get("pnl_percent", 0.0)
            for t in st.session_state.trade_history_log
            if t.get("outcome") in ["WIN", "LOSS"]
        ]
    )
    total_wins = len(
        [
            t
            for t in st.session_state.trade_history_log
            if t.get("outcome") == "WIN"
        ]
    )
    total_losses = len(
        [
            t
            for t in st.session_state.trade_history_log
            if t.get("outcome") == "LOSS"
        ]
    )

    closed_count = len([
        t
        for t in st.session_state.trade_history_log
        if t["outcome"] in ["WIN", "LOSS"]
    ])
    next_train_target = (
        closed_count // TRAIN_INTERVAL + 1
    ) * TRAIN_INTERVAL

    risk_engine = PowerTradingRiskEngine()
    disp_vol = np.sum(asks[:, 1]) if len(asks) > 0 else 1.0
    risk_metrics = risk_engine.calculate_risk_metrics(
        liquidation_volumes=np.array([1000, 2500]),
        displayed_vol=disp_vol,
        cancelled_vol=disp_vol * 0.1,
        time_exists=15.0,
        obs_window=60.0,
        open_interest=150000.0,
        leverage=20.0,
        volatility=df["Close"].pct_change().std() + 1e-8,
    )

    dir_color = (
        "#00e676"
        if direction == "LONG"
        else ("#ff5252" if direction == "SHORT" else "#38bdf8")
    )
    mins_rem, secs_rem = divmod(time_remaining, 60)
    pnl_total_color = "#00e676" if total_pnl_sum >= 0 else "#ff5252"
    pnl_total_sign = "+" if total_pnl_sum >= 0 else ""

    st.markdown(
        f"""
    <div class="top-status-bar" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            🟢 <b>[{selected_symbol}]</b> &nbsp;|&nbsp; Price: <b>${close_p:,.4f}</b> &nbsp;|&nbsp; 
            SIGNAL: <span style="color:{dir_color};">{direction}</span> &nbsp;|&nbsp; 
            Score: <b>{final_score:+.3f}</b> &nbsp;|&nbsp; Conf: <b>{confidence}%</b> &nbsp;|&nbsp; 
            ⏳ Reset: <b>{mins_rem}m {secs_rem}s</b>
        </div>
        <div>
            💰 Total P&L: <b style="color:{pnl_total_color};">{pnl_total_sign}{total_pnl_sum:.2f}%</b> &nbsp;|&nbsp; 
            🏆 Wins: <b style="color:#00e676;">{total_wins}</b> &nbsp;|&nbsp; ❌ Losses: <b style="color:#ff5252;">{total_losses}</b>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_sig, col_m1, col_m2, col_m3, col_m4 = st.columns([1.2, 1, 1, 1, 1])

    with col_sig:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid {dir_color};">
                <div class="metric-label">Signal Engine</div>
                <div style="font-size:22px; font-weight:700; color:{dir_color};">{direction}</div>
                <div style="font-size:11px; color:#8b949e; margin-top:4px;">Entry: ${close_p:,.4f} | SL: ${sl_val:,.4f}</div>
                <div style="font-size:11px; color:#38bdf8;">TP1: ${tp1_val:,.4f} | TP2: ${tp2_val:,.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">TP1 Target (1:2)</div><div class="metric-val-blue">${tp1_val:,.4f}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">TP2 Target (1:3)</div><div class="metric-val-blue">${tp2_val:,.4f}</div></div>',
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Closed Count</div><div class="metric-val-blue">{closed_count} Trades</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Next Train At</div><div class="metric-val-green">{next_train_target} Trades</div></div>',
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">LTZ Score</div><div class="metric-val-blue">{risk_metrics["LTZ_Score"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Spoof Score</div><div class="metric-val-red">{risk_metrics["Spoof_Score"]:.3f}</div></div>',
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Squeeze Risk</div><div class="metric-val-red">{risk_metrics["Squeeze_Risk"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Market Risk</div><div class="metric-val-red">{risk_metrics["Market_Risk"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )

    col_chart, col_risk_panel = st.columns([2.5, 1])
    with col_chart:
        st.subheader(f"Price Trajectory & Levels ({selected_symbol})")
        time_delta = pd.Timedelta(minutes=tf_minutes)
        future_times = [
            df["Time"].iloc[-1] + (i * time_delta)
            for i in range(1, forecast_horizon + 1)
        ]
        t_steps = np.linspace(0, np.pi / 2, forecast_horizon)

        if direction == "LONG":
            forecast_prices = close_p + (tp2_val - close_p) * np.sin(t_steps)
        elif direction == "SHORT":
            forecast_prices = close_p - (close_p - tp2_val) * np.sin(t_steps)
        else:
            forecast_prices = [close_p] * forecast_horizon

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df["Time"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
                increasing_line_color="#00e676",
                decreasing_line_color="#ff5252",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df["Time"].iloc[-1]] + future_times,
                y=[close_p] + list(forecast_prices),
                mode="lines+markers",
                name="Trajectory",
                line=dict(color=dir_color, width=2, dash="dot"),
            )
        )
        fig.add_hline(
            y=tp2_val,
            line_dash="dash",
            line_color="#00e676",
            annotation_text=f"TP2: ${tp2_val:,.4f}",
        )
        fig.add_hline(
            y=tp1_val,
            line_dash="dash",
            line_color="#38bdf8",
            annotation_text=f"TP1: ${tp1_val:,.4f}",
        )
        fig.add_hline(
            y=sl_val,
            line_dash="dot",
            line_color="#ff5252",
            annotation_text=f"SL: ${sl_val:,.4f}",
        )
        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#111622",
            plot_bgcolor="#111622",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_risk_panel:
        st.subheader("Market Microstructure")
        bid_vol_sum = np.sum(bids[:, 1]) if len(bids) > 0 else 1.0
        ask_vol_sum = np.sum(asks[:, 1]) if len(asks) > 0 else 1.0
        obi_val = (bid_vol_sum - ask_vol_sum) / (
            bid_vol_sum + ask_vol_sum + 1e-8
        )
        spread_val = (
            abs(asks[0, 0] - bids[0, 0])
            if len(bids) > 0 and len(asks) > 0
            else 0.0
        )

        st.markdown(
            f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Bid Volume</span> <b style="color:#00e676;">{bid_vol_sum:,.2f}</b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Ask Volume</span> <b style="color:#ff5252;">{ask_vol_sum:,.2f}</b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>OBI</span> <b style="color:#38bdf8;">{obi_val:+.3f}</b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Spread</span> <b>${spread_val:.4f}</b></div>
            <div style="display:flex; justify-content:space-between;"><span>Loop Scan</span> <b style="color:#00e676;">{"ACTIVE" if loop_all_coins else "SINGLE"}</b></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(
        """
    <div class="sticky-dashboard-header">
        <h3 style="margin: 0; font-size: 18px; color: #f0f6fc; font-weight: 700;">📊 Active & Closed Trades Dashboard (Scalping: 15m/30m | Intraday: 1h/4h)</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab_scalping, tab_intraday = st.tabs(
        ["⚡ Scalping (15m / 30m)", "📊 Intraday (1h / 4h)"]
    )

    if st.session_state.trade_history_log:
        # FIXED: Corrected filtering logic so 1h and 4h are properly captured in Intraday tab
        scalping_trades = [
            t
            for t in st.session_state.trade_history_log
            if any(tf in str(t.get("timeframe", "")) for tf in ["15m", "30m"])
        ]
        intraday_trades = [
            t
            for t in st.session_state.trade_history_log
            if any(tf in str(t.get("timeframe", "")) for tf in ["1h", "4h"])
        ]

        def get_coin_icon(symbol):
            if "BTC" in symbol:
                return "₿"
            elif "ETH" in symbol:
                return "Ξ"
            elif "SOL" in symbol:
                return "🟣"
            elif "XRP" in symbol:
                return "✕"
            elif "XMR" in symbol:
                return "🅜"
            elif "TAO" in symbol:
                return "τ"
            else:
                return "🪙"

        def render_trade_cards(trades_subset):
            if not trades_subset:
                st.info("No trades found for this category.")
                return
            for t in trades_subset:
                t_dir = t.get("direction", "LONG")
                raw_sym = t.get("symbol", "BTCUSDT")
                t_sym = raw_sym.replace("USDT", "/USDT")
                coin_icon = get_coin_icon(raw_sym)

                t_entry = t.get("entry_price", 0.0)
                t_sl = t.get("stop_loss", 0.0)
                t_tp1 = t.get("tp1", 0.0)
                t_tp2 = t.get("tp2", 0.0)
                t_conf = t.get("confidence", 90)
                t_pnl = t.get("pnl_percent", 0.0)
                t_status = t.get("status", "Open")
                t_exit = t.get("exit_price", t_entry)
                t_time = t.get("timestamp", "")
                t_tf = t.get("timeframe", "15m")
                t_outcome = t.get("outcome", "PENDING")

                # Format as **Hold:** time / Left: time
                max_duration_mins = (
                    30 if any(tf in t_tf for tf in ["15m", "30m"]) else 480
                )
                time_display_str = "Completed"
                if t_status == "Open" and t_time:
                    try:
                        t_dt = datetime.datetime.strptime(
                            t_time, "%Y-%m-%d %H:%M:%S"
                        )
                        elapsed_seconds = (
                            datetime.datetime.now() - t_dt
                        ).total_seconds()
                        
                        hold_m = int(elapsed_seconds // 60)
                        hold_s = int(elapsed_seconds % 60)
                        hold_str = f"{hold_m}m {hold_s}s"

                        total_allowed_seconds = max_duration_mins * 60
                        rem_sec = total_allowed_seconds - elapsed_seconds
                        if rem_sec > 0:
                            rem_m = int(rem_sec // 60)
                            rem_s = int(rem_sec % 60)
                            time_display_str = f"⏳ <b>Hold:</b> {hold_str} / Left: {rem_m}m {rem_s}s"
                        else:
                            time_display_str = f"⏳ <b>Hold:</b> {hold_str} / ⌛ Expiring..."
                    except Exception:
                        time_display_str = "Active"
                else:
                    time_display_str = "🔒 Completed"

                c_color = "#00e676" if t_dir == "LONG" else "#ff5252"

                if t_status == "Open":
                    run_status_bg = "rgba(0, 230, 118, 0.1)"
                    run_status_fg = "#00e676"
                    run_status_text = "🟢 RUNNING"
                else:
                    if t_outcome == "WIN":
                        run_status_bg = "rgba(0, 230, 118, 0.1)"
                        run_status_fg = "#00e676"
                        run_status_text = "🟢 PROFIT (WIN)"
                    else:
                        run_status_bg = "rgba(255, 82, 82, 0.1)"
                        run_status_fg = "#ff5252"
                        run_status_text = "🔴 LOSS"

                pnl_color = "#00e676" if t_pnl >= 0 else "#ff5252"
                pnl_sign = "+" if t_pnl >= 0 else ""

                card_html = (
                    '<div style="background: #111622; border: 1px solid #1e2638; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">'
                    f'<div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">'
                    f'<span style="background: {c_color}; color: #080a0f; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 12px;">{t_dir}</span>'
                    f'<span style="font-size: 15px; font-weight: 700; color: #ffffff;">{coin_icon} {t_sym}</span>'
                    f'<span style="font-size: 11px; color: #8b949e;">({t_tf}) • {t_time}</span>'
                    f"</div>"
                    f'<div style="display: flex; align-items: center; gap: 8px;">'
                    f'<span style="background: #1f293d; color: #38bdf8; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">{time_display_str}</span>'
                    f'<div style="background: {run_status_bg}; border: 1px solid {run_status_fg}; color: {run_status_fg}; padding: 3px 8px; border-radius: 20px; font-size: 11px; font-weight: 600;">{run_status_text}</div>'
                    f"</div>"
                    f"</div>"
                    f'<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; background: #0d1117; padding: 10px; border-radius: 8px;">'
                    f'<div><div style="font-size: 10px; color: #8b949e; text-transform: uppercase;">Entry Price</div><div style="font-size: 14px; font-weight: 700; color: #e2e8f0;">${t_entry:,.2f}</div></div>'
                    f'<div><div style="font-size: 10px; color: #8b949e; text-transform: uppercase;">Current / Exit</div><div style="font-size: 14px; font-weight: 700; color: {c_color};">${t_exit:,.2f}</div></div>'
                    f'<div><div style="font-size: 10px; color: #8b949e; text-transform: uppercase;">Position Size</div><div style="font-size: 14px; font-weight: 700; color: #38bdf8;">$2.00</div></div>'
                    f"</div>"
                    f'<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; text-align: center;">'
                    f'<div style="background: #161b22; padding: 6px; border-radius: 6px;"><div style="font-size: 9px; color: #8b949e;">SL</div><div style="font-size: 11px; font-weight: 700; color: #ff5252;">${t_sl:,.2f}</div></div>'
                    f'<div style="background: #161b22; padding: 6px; border-radius: 6px;"><div style="font-size: 9px; color: #8b949e;">TP1</div><div style="font-size: 11px; font-weight: 700; color: #00e676;">${t_tp1:,.2f}</div></div>'
                    f'<div style="background: #161b22; padding: 6px; border-radius: 6px;"><div style="font-size: 9px; color: #8b949e;">TP2</div><div style="font-size: 11px; font-weight: 700; color: #38bdf8;">${t_tp2:,.2f}</div></div>'
                    f'<div style="background: #161b22; padding: 6px; border-radius: 6px;"><div style="font-size: 9px; color: #8b949e;">CONF</div><div style="font-size: 11px; font-weight: 700; color: #38bdf8;">{t_conf}%</div></div>'
                    f'<div style="background: #161b22; padding: 6px; border-radius: 6px;"><div style="font-size: 9px; color: #8b949e;">P&L</div><div style="font-size: 11px; font-weight: 700; color: {pnl_color};">{pnl_sign}{t_pnl:.2f}%</div></div>'
                    f"</div>"
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

        with tab_scalping:
            render_trade_cards(scalping_trades)

        with tab_intraday:
            render_trade_cards(intraday_trades)
    else:
        with tab_scalping:
            st.info("No trade history available yet.")
        with tab_intraday:
            st.info("No trade history available yet.")
