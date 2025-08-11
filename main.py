# main.py
import ccxt
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score

# --- Step 1: Get BTC 1-minute price data ---
exchange = ccxt.binance()
symbol = "BTC/USDT"
data = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1000)  # last ~16 hours

df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('ts', inplace=True)

# --- Step 2: Add Technical Indicators ---
df['log_close'] = np.log(df['close'])
df['ret_1'] = df['log_close'].diff()
df['ret_5'] = df['log_close'].diff(5)
df['sma_10'] = df['close'].rolling(10).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['sma_ratio'] = df['sma_10'] / df['sma_50']
df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

for lag in [1,2,3,5,10]:
    df[f'ret_lag_{lag}'] = df['ret_1'].shift(lag)

# --- Step 3: Target (10-min future log return) ---
HORIZON = 10
df['target'] = df['log_close'].shift(-HORIZON) - df['log_close']
df.dropna(inplace=True)

# --- Step 4: Train/Test Split ---
features = ['ret_5', 'sma_ratio', 'rsi_14', 'macd', 'macd_signal'] + [f'ret_lag_{l}' for l in [1,2,3,5,10]]
split = int(len(df) * 0.8)
X_train, X_test = df[features][:split], df[features][split:]
y_train, y_test = df['target'][:split], df['target'][split:]

# --- Step 5: Train Model ---
model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, verbosity=0)
model.fit(X_train, y_train)

# --- Step 6: Predictions ---
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
direction_acc = accuracy_score((y_test > 0), (preds > 0))

print(f"RMSE: {rmse:.6f}")
print(f"Directional Accuracy: {direction_acc:.2%}")

# --- Step 7: Show latest prediction ---
latest_features = df[features].iloc[-1].values.reshape(1, -1)
pred_return = model.predict(latest_features)[0]
latest_price = df['close'].iloc[-1]
pred_price = np.exp(np.log(latest_price) + pred_return)

print(f"Latest BTC Price: {latest_price:.2f} USDT")
print(f"Predicted BTC Price in {HORIZON} min: {pred_price:.2f} USDT")
print("Prediction Direction:", "UP" if pred_price > latest_price else "DOWN")
