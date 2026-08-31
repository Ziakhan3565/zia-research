import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class SwingQuantTradingEngine:

  def __init__(self, T_window: int = 12, ml_lookback: int = 400):
    """Initialized for 1-2 Hour Holding Time (Using 5-minute bar resolution).

    T_window = 12 bars * 5 mins = 60 minutes (1 Hour lookback) ml_lookback =
    Training lookback window
    """
    self.T_window = T_window
    self.ml_lookback = ml_lookback
    self.scaler = StandardScaler()
    self.model = RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=42, n_jobs=-1
    )

  def compute_cross_ofi(
      self, venues_data: Dict[str, pd.DataFrame], beta_weights: Dict[str, float]
  ) -> pd.Series:
    timestamps = list(next(iter(venues_data.values())).index)
    cross_ofi_list = []

    for t in range(1, len(timestamps)):
      total_ofi = 0.0
      for m, df in venues_data.items():
        beta = beta_weights.get(m, 1.0)
        curr, prev = df.iloc[t], df.iloc[t - 1]

        dv_b = (
            curr['bid_v']
            if curr['bid_p'] > prev['bid_p']
            else (
                curr['bid_v'] - prev['bid_v']
                if curr['bid_p'] == prev['bid_p']
                else -prev['bid_v']
            )
        )
        dv_a = (
            -prev['ask_v']
            if curr['ask_p'] > prev['ask_p']
            else (
                curr['ask_v'] - prev['ask_v']
                if curr['ask_p'] == prev['ask_p']
                else curr['ask_v']
            )
        )

        ofi_m = dv_b - dv_a
        total_ofi += beta * ofi_m
      cross_ofi_list.append(total_ofi)

    return pd.Series(
        [0.0] + cross_ofi_list, index=timestamps, name='Cross_OFI'
    )

  def compute_kyles_lambda(
      self, net_vol: pd.Series, price_series: pd.Series
  ) -> pd.Series:
    delta_p = price_series.diff(self.T_window)
    sum_net_vol = net_vol.rolling(window=self.T_window).sum()
    cov = delta_p.rolling(window=self.T_window).cov(sum_net_vol)
    var = sum_net_vol.rolling(window=self.T_window).var()
    lam = cov / (var + 1e-8)
    return lam.rename('Kyles_Lambda').fillna(0)

  def compute_cvd_skew_features(
      self, trade_df: pd.DataFrame, n: int = 12
  ) -> pd.DataFrame:
    trade_df['net_trade_vol'] = (
        trade_df['v_market_buy'] - trade_df['v_market_sell']
    )
    trade_df['cvd'] = trade_df['net_trade_vol'].cumsum()

    ema_cvd = trade_df['cvd'].ewm(span=n, adjust=False).mean()
    std_cvd = trade_df['cvd'].rolling(window=n).std().fillna(1e-8)

    trade_df['cvd_skew'] = (trade_df['cvd'] - ema_cvd) / std_cvd
    trade_df['ema_10'] = trade_df['cvd_skew'].ewm(span=10, adjust=False).mean()
    trade_df['ema_20'] = trade_df['cvd_skew'].ewm(span=20, adjust=False).mean()

    return trade_df

  def compute_basis_velocity(
      self, perp_price: pd.Series, spot_price: pd.Series
  ) -> Tuple[pd.Series, pd.Series]:
    basis = perp_price - spot_price
    basis_velocity = basis - basis.shift(self.T_window)
    return (
        basis.rename('Basis'),
        basis_velocity.fillna(0).rename('Basis_Velocity'),
    )

  def feature_engineering_pipeline(
      self,
      venues_data: Dict[str, pd.DataFrame],
      trade_df: pd.DataFrame,
      beta_weights: Dict[str, float],
  ) -> pd.DataFrame:
    print(
        '[INFO] Processing features for 1-2 Hour Swing Horizon (5m Resolution)...'
    )

    cross_ofi = self.compute_cross_ofi(venues_data, beta_weights)
    trade_features = self.compute_cvd_skew_features(trade_df)
    agg_price = list(venues_data.values())[0]['bid_p']
    kyles_lam = self.compute_kyles_lambda(
        trade_features['net_trade_vol'], agg_price
    )

    perp_p = trade_df.get('perp_price', agg_price * 1.001)
    spot_p = trade_df.get('spot_price', agg_price)
    basis, basis_vel = self.compute_basis_velocity(perp_p, spot_p)

    feature_df = pd.DataFrame(index=trade_features.index)
    feature_df['Cross_OFI'] = cross_ofi
    feature_df['Kyles_Lambda'] = kyles_lam
    feature_df['CVD'] = trade_features['cvd']
    feature_df['CVD_Skew'] = trade_features['cvd_skew']
    feature_df['EMA_10'] = trade_features['ema_10']
    feature_df['EMA_20'] = trade_features['ema_20']
    feature_df['Basis'] = basis
    feature_df['Basis_Velocity'] = basis_vel
    feature_df['Close_Price'] = agg_price

    # Target definition for 1-2 hour horizon (checking price 12 bars ahead ~ 1 hour)
    horizon_bars = 12
    future_returns = agg_price.shift(-horizon_bars) - agg_price
    feature_df['Target'] = np.where(future_returns > 0, 1, 0)

    return feature_df.dropna()

  def walk_forward_training_with_targets(self, feature_df: pd.DataFrame):
    print('[INFO] Starting Walk-Forward Training & Target Generation Loop...')

    X = feature_df.drop(columns=['Target', 'Close_Price'])
    prices = feature_df['Close_Price']
    y = feature_df['Target']

    n_samples = len(X)
    predictions = []
    actuals = []
    trade_signals = []

    for i in range(self.ml_lookback, n_samples - 12):
      X_train = X.iloc[i - self.ml_lookback : i]
      y_train = y.iloc[i - self.ml_lookback : i]
      X_test = X.iloc[[i]]

      X_train_scaled = self.scaler.fit_transform(X_train)
      X_test_scaled = self.scaler.transform(X_test)

      self.model.fit(X_train_scaled, y_train)
      pred = self.model.predict(X_test_scaled)[0]

      current_price = prices.iloc[i]

      # Dynamic Target & Risk Management Setup (Risk-Reward Ratio 1:2)
      if pred == 1:  # Long Signal
        take_profit = current_price * 1.015  # +1.5% Target
        stop_loss = current_price * 0.995  # -0.5% Stop Loss
        signal_type = 'LONG'
      else:  # Short Signal
        take_profit = current_price * 0.985  # -1.5% Target
        stop_loss = current_price * 1.005  # +0.5% Stop Loss
        signal_type = 'SHORT'

      predictions.append(pred)
      actuals.append(y.iloc[i])
      trade_signals.append({
          'Timestamp': X.index[i],
          'Signal': signal_type,
          'Entry_Price': current_price,
          'Take_Profit': take_profit,
          'Stop_Loss': stop_loss,
      })

    acc = accuracy_score(actuals, predictions)
    print('\n================ SWING STRATEGY PERFORMANCE ================')
    print(f'Holding Window Target: 1 Hour (5m x 12 bars)')
    print(f'Model Accuracy: {acc * 100:.2f}%')
    print('------------------------------------------------------------')
    print(pd.DataFrame(trade_signals).tail(5))  print last 5 generated signals
    print('============================================================')

    return trade_signals


# ==========================================
# SIMULATION ENTRY POINT (5-Min Bars)
# ==========================================
if __name__ == '__main__':
  np.random.seed(42)
  # 5-minute frequency simulation for swing horizons
  time_index = pd.date_range(start='2026-08-01', periods=1500, freq='5min')

  venues_mock = {
      'MEXC': pd.DataFrame(
          {
              'bid_p': 60000 + np.cumsum(np.random.randn(1500) * 15),
              'ask_p': 60002 + np.cumsum(np.random.randn(1500) * 15),
              'bid_v': np.random.randint(50, 500, size=1500),
              'ask_v': np.random.randint(50, 500, size=1500),
          },
          index=time_index,
      ),
      'Binance': pd.DataFrame(
          {
              'bid_p': 59998 + np.cumsum(np.random.randn(1500) * 15),
              'ask_p': 60000 + np.cumsum(np.random.randn(1500) * 15),
              'bid_v': np.random.randint(50, 500, size=1500),
              'ask_v': np.random.randint(50, 500, size=1500),
          },
          index=time_index,
      ),
  }

  trade_mock = pd.DataFrame(
      {
          'v_market_buy': np.random.randint(20, 200, size=1500),
          'v_market_sell': np.random.randint(20, 200, size=1500),
          'perp_price': 60010 + np.cumsum(np.random.randn(1500) * 15),
          'spot_price': 60000 + np.cumsum(np.random.randn(1500) * 15),
      },
      index=time_index,
  )

  beta_weights_config = {'MEXC': 0.55, 'Binance': 0.45}

  engine = SwingQuantTradingEngine(T_window=12, ml_lookback=300)
  processed_features = engine.feature_engineering_pipeline(
      venues_mock, trade_mock, beta_weights_config
  )
  engine.walk_forward_training_with_targets(processed_features)
