import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 🎯 Entrada del usuario
ticker = input("📥 Ingresá el ticker (ej: AAPL, MSFT, TSLA): ").upper()
start_date = "2020-01-01"
end_date = "2024-12-31"

# 📦 Descargar datos
df = yf.download(ticker, start=start_date, end=end_date)
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()

# 🟢🔴 Señales de trading
df['Signal'] = 0
df.loc[df['SMA20'] > df['SMA50'], 'Signal'] = 1
df.loc[df['SMA20'] < df['SMA50'], 'Signal'] = -1
df['Position'] = df['Signal'].shift(1)

# 📈 Cálculo de retornos
df['Daily Return'] = df['Close'].pct_change()
df['Strategy Return'] = df['Daily Return'] * df['Position']

# 📍 Señales visuales
buy_signals = df[(df['Signal'] == 1) & (df['Signal'].shift(1) != 1)]
sell_signals = df[(df['Signal'] == -1) & (df['Signal'].shift(1) != -1)]

# 📊 Graficar precio, SMAs y señales
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Precio', linewidth=1)
plt.plot(df['SMA20'], label='SMA20', linestyle='--')
plt.plot(df['SMA50'], label='SMA50', linestyle='--')
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Compra', s=100)
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Venta', s=100)
plt.title(f'Señales de Trading SMA20/SMA50 en {ticker}')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 📊 Graficar performance acumulada
df[['Daily Return', 'Strategy Return']].cumsum().apply(np.exp).plot(figsize=(12, 6))
plt.title(f"Comparación Estrategia SMA20/SMA50 vs Buy & Hold en {ticker}")
plt.grid()
plt.show()

# 🧮 Métricas
total_return_strategy = df['Strategy Return'].cumsum().apply(np.exp).iloc[-1]
total_return_bh = df['Daily Return'].cumsum().apply(np.exp).iloc[-1]
sharpe_ratio = df['Strategy Return'].mean() / df['Strategy Return'].std() * (252 ** 0.5)

print("\n📊 Estadísticas de la Estrategia:")
print(f"• Retorno total estrategia: {total_return_strategy:.2f}x")
print(f"• Retorno total buy & hold: {total_return_bh:.2f}x")
print(f"• Sharpe Ratio estrategia: {sharpe_ratio:.2f}")