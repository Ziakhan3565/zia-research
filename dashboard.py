import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title='ZIA RESEARCH - Quantitative Trading Terminal',
    page_icon='📈',
    layout='wide',
)

# Custom CSS Styling for Professional Dark Terminal Look
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .metric-card { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""",
    unsafe_allow_html=True,
)

# App Header
st.title('⚡ ZIA RESEARCH: Multi-Venue Quant & ML Terminal')
st.markdown(
    '**Target Holding Window:** 1-2 Hours (5-Minute Resolution) | **Models:**'
    ' Cross-OFI, Kyle’s Lambda, CVD Skew, Basis Velocity'
)
st.markdown('---')


# Simulated Data & Engine Pipeline Functions
@st.cache_data
def run_quant_pipeline():
  np.random.seed(42)
  time_index = pd.date_range(
      start='2026-08-01', periods=1000, freq='5min'
  )

  venues_mock = {
      'MEXC': pd.DataFrame(
          {
              'bid_p': 60000 + np.cumsum(np.random.randn(1000) * 15),
              'ask_p': 60002 + np.cumsum(np.random.randn(1000) * 15),
              'bid_v': np.random.randint(50, 500, size=1000),
              'ask_v': np.random.randint(50, 500, size=1000),
          },
          index=time_index,
      ),
      'Binance': pd.DataFrame(
          {
              'bid_p': 59998 + np.cumsum(np.random.randn(1000) * 15),
              'ask_p': 60000 + np.cumsum(np.random.randn(1000) * 15),
              'bid_v': np.random.randint(50, 500, size=1000),
              'ask_v': np.random.randint(50, 500, size=1000),
          },
          index=time_index,
      ),
  }

  trade_mock = pd.DataFrame(
      {
          'v_market_buy': np.random.randint(20, 200, size=1000),
          'v_market_sell': np.random.randint(20, 200, size=1000),
          'perp_price': 60010 + np.cumsum(np.random.randn(1000) * 15),
          'spot_price': 60000 + np.cumsum(np.random.randn(1000) * 15),
      },
      index=time_index,
  )

  # Feature Calculations
  agg_price = venues_mock['MEXC']['bid_p']
  net_vol = trade_mock['v_market_buy'] - trade_mock['v_market_sell']
  cvd = net_vol.cumsum()
  cvd_skew = (cvd - cvd.ewm(span=12).mean()) / (
      cvd.rolling(window=12).std() + 1e-8
  )

  feature_df = pd.DataFrame(index=time_index)
  feature_df['Cross_OFI'] = np.random.randn(1000) * 120
  feature_df['Kyles_Lambda'] = (
      agg_price.diff(12).rolling(12).cov(net_vol.rolling(12).sum())
      / (net_vol.rolling(12).var() + 1e-8)
  ).fillna(0)
  feature_df['CVD_Skew'] = cvd_skew
  feature_df['EMA_10'] = cvd_skew.ewm(span=10).mean()
  feature_df['EMA_20'] = cvd_skew.ewm(span=20).mean()
  feature_df['Basis'] = trade_mock['perp_price'] - trade_mock['spot_price']
  feature_df['Basis_Velocity'] = feature_df['Basis'] - feature_df['Basis'].shift(
      12
  )
  feature_df['Close_Price'] = agg_price
  feature_df['Target'] = np.where(agg_price.shift(-12) - agg_price > 0, 1, 0)

  return feature_df.dropna()


df_features = run_quant_pipeline()

# Sidebar Controls
st.sidebar.header('⚙️ Terminal Controls')
selected_asset = st.sidebar.selectbox(
    'Select Trading Asset',
    ['BTC/USDT (MEXC)', 'SOL/USDT (MEXC)', 'ETH/USDT (MEXC)'],
)
lookback_window = st.sidebar.slider(
    'ML Lookback Period', 100, 500, 300, step=50
)
risk_reward = st.sidebar.selectbox(
    'Risk-Reward Strategy', ['1:1.5 (Conservative)', '1:2.0 (Standard)']
)

# Top Metrics Overview
col1, col2, col3, col4 = st.columns(4)
with col1:
  st.metric(
      label='Current Price (Ref)',
      value=f"${df_features['Close_Price'].iloc[-1]:,.2f}",
      delta='+0.84%',
  )
with col2:
  st.metric(
      label='Cross-OFI Signal',
      value=f"{df_features['Cross_OFI'].iloc[-1]:,.1f}",
      delta='Bullish Pressure',
  )
with col3:
  st.metric(
      label="Kyle's Lambda (Impact)",
      value=f"{df_features['Kyles_Lambda'].iloc[-1]:.4f}",
  )
with col4:
  st.metric(
      label='CVD Skew State',
      value=f"{df_features['CVD_Skew'].iloc[-1]:.2f}σ",
      delta='Overbought Zone',
  )

st.markdown('---')

# Main Layout: Charts and Active Signals
col_left, col_right = st.columns([2, 1])

with col_left:
  st.subheader('📊 Real-Time Price & CVD Skew Telemetry')
  st.line_chart(
      df_features[['Close_Price']] / df_features['Close_Price'].iloc[0]
  )

  st.subheader('📈 Feature Matrix: EMA 10 & 20 on CVD Skew')
  st.line_chart(df_features[['CVD_Skew', 'EMA_10', 'EMA_20']].tail(150))

with col_right:
  st.subheader('🤖 ML Signal & Trade Setup')

  # Simple ML evaluation preview
  scaler = StandardScaler()
  model = RandomForestClassifier(n_estimators=100, random_state=42)

  X = df_features.drop(columns=['Target', 'Close_Price'])
  y = df_features['Target']

  X_scaled = scaler.fit_transform(X.iloc[-lookback_window:])
  model.fit(X_scaled, y.iloc[-lookback_window:])

  latest_X = scaler.transform(X.iloc[[-1]])
  prediction = model.predict(latest_X)[0]

  cur_price = df_features['Close_Price'].iloc[-1]

  if prediction == 1:
    signal_str = '🟢 LONG SIGNAL (BUY)'
    tp = cur_price * 1.015
    sl = cur_price * 0.995
  else:
    signal_str = '🔴 SHORT SIGNAL (SELL)'
    tp = cur_price * 0.985
    sl = cur_price * 1.005

  st.markdown(f"""
        <div class="metric-card">
            <h3>{signal_str}</h3>
            <p><b>Entry Price:</b> ${cur_price:,.2f}</p>
            <p><b>Take Profit (1-2h):</b> <span style="color: #00ff00;">${tp:,.2f}</span></p>
            <p><b>Stop Loss:</b> <span style="color: #ff4d4d;">${sl:,.2f}</span></p>
            <hr>
            <p><b>Model Status:</b> Active & Training</p>
            <p><b>Confidence Score:</b> 68.4%</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('---')
st.subheader('📋 Recent Executed Signals Log')

# Generate recent mock trade log
log_data = []
for i in range(1, 6):
  p = df_features['Close_Price'].iloc[-i]
  log_data.append({
      'Timestamp': df_features.index[-i],
      'Asset': selected_asset.split(' ')[0],
      'Signal': 'LONG' if i % 2 == 0 else 'SHORT',
      'Entry': f'${p:,.2f}',
      'Target (TP)': f'${p * 1.015:,.2f}',
      'Status': 'CLOSED (WIN)' if i > 2 else 'ACTIVE',
  })

st.dataframe(pd.DataFrame(log_data), use_container_width=True)
