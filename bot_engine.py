import json
import os
import time
import ccxt
import pandas as pd

# Exchange Setup
mexc = ccxt.mexc({
    'apiKey': os.getenv('MEXC_API_KEY', 'YOUR_MEXC_API_KEY'),
    'secret': os.getenv('MEXC_SECRET_KEY', 'YOUR_MEXC_SECRET_KEY'),
    'enableRateLimit': True,
})


def set_leverage_and_margin(symbol, leverage):
  """Leverage aur Isolated Margin mode set karna"""
  try:
    mexc.set_leverage(leverage, symbol)
    mexc.set_margin_mode('isolated', symbol)
    print(f'⚙️ Leverage set to {leverage}x (Isolated) for {symbol}')
  except Exception as e:
    print(f'⚠️ Leverage error: {e}')


def run_bot():
  processed_signals = set()
  print(
      '🤖 Bot Engine Ready with Fourier Dynamic SL/TP. Check your dashboard to'
      ' Start/Stop.'
  )

  while True:
    # 1. Config Load Karein
    if os.path.exists('config.json'):
      with open('config.json', 'r') as f:
        cfg = json.load(f)
    else:
      cfg = {'is_running': False, 'leverage': 5, 'selected_coins': []}

    # 2. Agar Bot OFF hai, toh wait karein
    if not cfg.get('is_running', False):
      time.sleep(5)
      continue

    # 3. Signals Monitor Karein
    if os.path.exists('signal_history.csv'):
      try:
        df = pd.read_csv('signal_history.csv')
        if not df.empty:
          latest = df.iloc[0]
          sig_id = f"{latest['timestamp']}_{latest['symbol']}"

          # Agar naya signal hai aur select kiye gaye coins mein se hai
          if sig_id not in processed_signals and latest['symbol'] in cfg.get(
              'selected_coins', []
          ):
            symbol = latest['symbol']
            leverage = cfg.get('leverage', 5)

            # Leverage & Margin Set Karein
            set_leverage_and_margin(symbol, leverage)

            # Signal details extract karein (LONG/SHORT + Dynamic SL/TP)
            intent = (
                str(latest.get('intent', 'LONG'))
                .upper()
                .strip()
            )  # ya ml_signal
            entry_price = float(latest.get('current_price', 0))
            stop_loss = float(
                latest.get('stop_loss', entry_price * 0.994)
            )  # Fourier SL
            take_profit = float(
                latest.get('take_profit', entry_price * 1.006)
            )  # Fourier TP

            side = 'buy' if intent == 'LONG' else 'sell'

            # Order Execution with Fourier Dynamic SL/TP Params
            print(f'🚀 Executing {intent} Trade on {symbol} at ~{entry_price}')
            print(
                f'   🎯 Fourier TP: {take_profit} | 🛡️ Fourier SL (w/ Hard'
                f' Cap): {stop_loss}'
            )

            try:
              # Market Order placement with attached TP/SL parameters for MEXC Futures
              # (CCXT params structure exchange ke mutabiq vary kar sakta hai)
              order_params = {
                  'stopLossPrice': stop_loss,
                  'takeProfitPrice': take_profit,
              }

              # Misal ke taur par 10$ ki trade amount ya config se size uthayein
              amount = float(cfg.get('trade_amount_usdt', 10.0)) / entry_price

              # Order create karein
              order = mexc.create_order(
                  symbol=symbol,
                  type='market',
                  side=side,
                  amount=amount,
                  params=order_params,
              )
              print(f'✅ Order Successfully Placed! ID: {order.get("id")}')

            except Exception as order_err:
              print(f'❌ Order Execution Error on Exchange: {order_err}')

            processed_signals.add(sig_id)

      except Exception as read_err:
        print(f'⚠️ Error reading signal_history.csv: {read_err}')

    time.sleep(5)


if __name__ == '__main__':
  run_bot()
