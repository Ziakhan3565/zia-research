import datetime
import os
import pickle
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ==========================================
# STREAMLIT CONFIG & PERSISTENT CSV SETUP
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
# XGBOOST MODEL LOADING
# ==========================================
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
# RESEARCH LAB & RISK ENGINE MODULES
# ==========================================
class TenPaperResearchLab:

    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        self.feature_names = [
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
            "FOURIER_TREND",
        ]
        self.dynamic_weights = {
            k: 1.0 / len(self.feature_names) for k in self.feature_names
        }

    def extract_features(self, df, bids, asks):
        results = {}
        if len(bids) == 0 or len(asks) == 0 or df.empty or len(df) < 15:
            return {k: 0.0 for k in self.feature_names}

        top20_bid_sum = np.sum(bids[:20, 1])
        top20_ask_sum = np.sum(asks[:20, 1])
        obi_top20 = (top20_bid_sum - top20_ask_sum) / (
            top20_bid_sum + top20_ask_sum + 1e-8
        )
        spread = abs(asks[0, 0] - bids[0, 0])
        bid_ask_ratio = top20_bid_sum / (top20_ask_sum + 1e-5)
        total_depth = top20_bid_sum + top20_ask_sum

        sma_20 = df["Close"].rolling(20, min_periods=1).mean().iloc[-1]
        trend_signal = df["Close"].iloc[-1] - sma_20

        delta_p = df["Close"].iloc[-1] - df["Close"].iloc[-2]
        returns_h = (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / (
            df["Close"].iloc[-5] + 1e-8
        )

        book_imb = obi_top20
        taker_buy = df["Volume"].iloc[-1] * (1.0 if delta_p > 0 else 0.3)
        taker_sell = df["Volume"].iloc[-1] * (1.0 if delta_p <= 0 else 0.3)
        taker_flow = (taker_buy - taker_sell) / (
            taker_buy + taker_sell + 1e-8
        )
        quant_imply = np.clip(book_imb * 1.5, -1, 1)

        ma_fast = df["Close"].rolling(3, min_periods=1).mean().iloc[-1]
        ma_slow = df["Close"].rolling(10, min_periods=1).mean().iloc[-1]
        realized_vol = df["Close"].pct_change().std() + 1e-8
        adapt_conf = np.clip(
            (ma_fast - ma_slow)
            / (realized_vol * df["Close"].iloc[-1] + 1e-8),
            -1,
            1,
        )

        prior = 0.745
        likelihood = 1.0 if book_imb > 0 else 0.25
        posterior = (likelihood * prior) / (
            (likelihood * prior) + ((1 - likelihood) * (1 - prior)) + 1e-8
        )
        bayesian = np.clip((posterior - 0.5) * 2.0, -1, 1)

        # Fourier trend calculation helper
        prices = df["Close"].values
        xc = prices - np.mean(prices)
        fft_vals = np.fft.fft(xc)
        num_keep = max(1, int(len(fft_vals) * 0.15))
        fft_masked = np.zeros_like(fft_vals)
        fft_masked[:num_keep] = fft_vals[:num_keep]
        fft_masked[-num_keep:] = fft_vals[-num_keep:]
        trend_curve = np.real(np.fft.ifft(fft_masked))
        diffs = np.gradient(trend_curve)
        fourier_trend = float(diffs[-1]) if len(diffs) > 0 else 0.0

        return {
            "top20_bid_sum": top20_bid_sum,
            "top20_ask_sum": top20_ask_sum,
            "obi_top20": obi_top20,
            "spread": spread,
            "bid_ask_ratio": bid_ask_ratio,
            "total_depth": total_depth,
            "trend_signal": trend_signal,
            "BOOK_IMB": book_imb,
            "TAKER_FLOW": taker_flow,
            "QUANT_IMPLY": quant_imply,
            "ADAPT_CONF": adapt_conf,
            "BAYESIAN": bayesian,
            "FOURIER_TREND": fourier_trend,
        }

    def calculate_all_signals(self, df, bids, asks):
        results = self.extract_features(df, bids, asks)
        feature_vector = np.array(
            [results[k] for k in self.feature_names]
        ).reshape(1, -1)

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

        final_score = (0.3 * results["BOOK_IMB"]) + (
            0.7 * (ml_probability - 0.5) * 2.0
        )
        return (
            results,
            float(np.clip(final_score, -1, 1)),
            self.dynamic_weights,
        )


# ==========================================
# STYLING & THEME
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
# SIDEBAR CONTROLS
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
    "15m (Medium TF)": ("15m", 15),
    "30m (Medium TF)": ("30m", 30),
    "1h (Intraday)": ("1h", 60),
    "4h (Intraday)": ("4h", 240),
}

st.sidebar.markdown("### ⚡ Multi-Coin Scanner Terminal")
selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe", list(TIMEFRAME_MAP.keys()), index=0
)
paper_trading_mode = st.sidebar.toggle("Enable Live Paper Trading", value=True)

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]


# ==========================================
# DATA FETCHING HELPERS
# ==========================================
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
            raise ValueError("Invalid format")
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


def fetch_order_book_depth(symbol, depth_limit=30):
    try:
        url = f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        res = requests.get(url, timeout=4).json()
        if "bids" in res and "asks" in res:
            return np.array(res["bids"], dtype=float), np.array(
                res["asks"], dtype=float
            )
    except Exception:
        pass
    dummy_bids = np.array([[100.0 - i * 0.1, 1.5] for i in range(30)], dtype=float)
    dummy_asks = np.array([[100.0 + i * 0.1, 1.5] for i in range(30)], dtype=float)
    return dummy_bids, dummy_asks


# ==========================================
# MULTI-COIN SCANNER & BACKGROUND EXECUTION
# ==========================================
st.markdown("### 🌐 Live Multi-Coin Microstructure Scanner")

lab = TenPaperResearchLab()
scanner_results = []
lock_seconds = tf_minutes * 60
current_time_sec = int(time.time())
time_bucket = current_time_sec - (current_time_sec % lock_seconds)

for coin in COINS_LIST:
    df_coin = fetch_klines_data(coin, selected_tf_label)
    bids_coin, asks_coin = fetch_order_book_depth(coin)

    if df_coin.empty or len(bids_coin) == 0 or len(asks_coin) == 0:
        continue

    _, final_score, _ = lab.calculate_all_signals(
        df_coin, bids_coin, asks_coin
    )
    close_p = df_coin["Close"].iloc[-1]
    atr_val = (df_coin["High"] - df_coin["Low"]).rolling(14).mean().iloc[-1]
    if np.isnan(atr_val):
        atr_val = close_p * 0.005

    direction = (
        "LONG"
        if final_score >= 0.10
        else ("SHORT" if final_score <= -0.10 else "NEUTRAL")
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

    # Auto trade logging for any strong signal in background
    trade_id = f"{coin}_{selected_tf_label}_{time_bucket}"
    if paper_trading_mode and direction != "NEUTRAL":
        existing_trade_ids = [
            item.get("trade_id") for item in st.session_state.trade_history_log
        ]
        if trade_id not in existing_trade_ids:
            new_trade = {
                "trade_id": trade_id,
                "timestamp": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "symbol": coin,
                "timeframe": selected_tf_label,
                "direction": direction,
                "entry_price": round(close_p, 4),
                "stop_loss": round(sl_val, 4),
                "tp1": round(tp1_val, 4),
                "tp2": round(tp2_val, 4),
                "exit_price": round(close_p, 4),
                "confidence": confidence,
                "final_score": round(final_score, 3),
                "outcome": "PENDING",
                "pnl_percent": 0.0,
                "duration": "Active",
                "status": "Open",
            }
            st.session_state.trade_history_log.insert(0, new_trade)
            save_persistent_history(st.session_state.trade_history_log)

    scanner_results.append({
        "Symbol": coin,
        "Price": round(close_p, 4),
        "Signal": direction,
        "Score": round(final_score, 3),
        "Confidence": f"{confidence}%",
    })

# Show live table of all scanned coins
df_scanner = pd.DataFrame(scanner_results)
st.dataframe(df_scanner, use_container_width=True)

# ==========================================
# PENDING TRADES OUTCOME CHECKER
# ==========================================
for trade in st.session_state.trade_history_log:
    if trade["outcome"] == "PENDING":
        trade_symbol = trade["symbol"]
        temp_df = fetch_klines_data(trade_symbol, trade["timeframe"], limit=15)
        if not temp_df.empty:
            entry_time_str = trade.get("timestamp")
            for idx, row in temp_df.iterrows():
                candle_time = str(row["Time"])
                if entry_time_str and candle_time >= entry_time_str:
                    curr_high = row["High"]
                    curr_low = row["Low"]
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

save_persistent_history(st.session_state.trade_history_log)

# ==========================================
# PERFORMANCE SUMMARY & HISTORY TABLE
# ==========================================
st.markdown("---")
st.subheader("📊 Performance Summary & Auto-Logged History")

if st.session_state.trade_history_log:
    df_log = pd.DataFrame(st.session_state.trade_history_log)
    total_signals = len(df_log)
    wins = len(df_log[df_log["outcome"] == "WIN"])
    losses = len(df_log[df_log["outcome"] == "LOSS"])
    pending = len(df_log[df_log["outcome"] == "PENDING"])
    closed_trades = wins + losses
    win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0.0

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Win Rate</div><div class="metric-val-green">{win_rate:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Closed Trades</div><div class="metric-val-blue">{closed_trades}</div></div>',
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Wins / Losses</div><div style="font-size:16px; font-weight:700; color:#00e676;">{wins}W / {losses}L</div></div>',
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Pending Active</div><div class="metric-val-blue">{pending}</div></div>',
            unsafe_allow_html=True,
        )

    st.dataframe(df_log, use_container_width=True)
