import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 🔧 Page config
st.set_page_config(
    page_title="Multi-Asset SMA Backtest | Matteo Bontempo",
    layout="wide"
)

# === SIDEBAR: ABOUT ME ===
with st.sidebar:
    st.image(
        "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
        width=30
    )
    st.markdown("### 👨‍💻 Matteo Vito Bontempo")
    st.markdown(
        "B.Sc. in Accounting & CIS – Dec 2025  \n"
        "North Carolina Wesleyan University  \n"
        "🇦🇷 From Argentina"
    )
    st.markdown(
        "[![GitHub]"
        "(https://img.shields.io/badge/GitHub-matteovbontempo-black?logo=github)]"
        "(https://github.com/matteovbontempo)"
    )
    st.markdown(
        "[![LinkedIn]"
        "(https://img.shields.io/badge/LinkedIn-Matteo--Bontempo-blue?logo=linkedin)]"
        "(https://www.linkedin.com/in/matteo-bontempo-)"
    )
    st.markdown(
        "[![Instagram]"
        "(https://img.shields.io/badge/Instagram-@vito.bontempo-purple?logo=instagram)]"
        "(https://www.instagram.com/vito.bontempo)"
    )
    st.markdown("---")
    st.markdown("### 🧾 About Me")
    st.write(
        """
I'm a student from Argentina pursuing a double major in Computer Information Systems (CIS)
and Accounting, with plans to graduate in December 2025.

My academic journey reflects my deep interest in combining technology with business to create
efficient, data-driven solutions.

I gained valuable experience during the COVID-19 pandemic working as a data entry assistant
in a medical lab. I uploaded COVID-19 test results to the national health system and sent lab
reports to patients, which taught me attention to detail and responsibility under pressure.

Currently, I’m interning at CoolGeeks, where I help repair electronic devices, build websites
and apps, and assist with networking.

I’m deeply interested in the financial world and stock market, and I aspire to create powerful
tools using programming to benefit finance by blending both fields.
        """
    )

# === PAGE TITLE ===
st.title("📈 Multi-Asset SMA Backtesting Tool")
st.markdown(
    "Backtest a moving average crossover strategy (SMA20 vs SMA50 vs SMA200) "
    "on multiple assets and compare performance."
)

# === INPUTS ===
tickers_input = st.text_input(
    "Enter multiple stock tickers separated by commas:",
    value="AAPL, MSFT, TSLA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.date_input(
    "Start Date",
    value=pd.to_datetime("2020-01-01")
)
end_date = st.date_input(
    "End Date",
    value=pd.to_datetime("today")
)

# === LIVE DATA PREVIEW ===
if tickers:
    st.markdown("### 🌐 Live Data Preview")
    cols = st.columns(len(tickers))
    for i, tkr in enumerate(tickers):
        try:
            info    = yf.Ticker(tkr).info
            current = info.get("currentPrice", np.nan)
            change  = info.get("regularMarketChangePercent", np.nan)
            high52  = info.get("fiftyTwoWeekHigh", np.nan)
            low52   = info.get("fiftyTwoWeekLow", np.nan)
            cols[i].metric(label=tkr, value=f"${current:.2f}", delta=f"{change:.2f}%")
            cols[i].write(f"52w High: ${high52:.2f}")
            cols[i].write(f"52w Low:  ${low52:.2f}")
        except Exception:
            cols[i].write(f"⚠️ Data unavailable for {tkr}")

# === DATA LOADER & SMA CALCS ===
@st.cache_data
def load_data(ticker, start, end, sma_short=20, sma_mid=50, sma_long=200):
    df = yf.download(ticker, start=start, end=end)
    df['SMA20'] = df['Close'].rolling(sma_short).mean()
    df['SMA50'] = df['Close'].rolling(sma_mid).mean()
    df['SMA200'] = df['Close'].rolling(sma_long).mean()
    df['Signal'] = 0
    df.loc[df['SMA20'] > df['SMA50'], 'Signal'] = 1
    df.loc[df['SMA20'] < df['SMA50'], 'Signal'] = -1
    df['Position'] = df['Signal'].shift(1)
    df['Daily Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Daily Return'] * df['Position']
    return df

# === RUN BACKTEST & COLLECT METRICS ===
results = []
all_data = {}
for tkr in tickers:
    try:
        df = load_data(tkr, start_date, end_date)
        all_data[tkr] = df
        strat_x = df['Strategy Return'].cumsum().apply(np.exp).iloc[-1]
        bh_x    = df['Daily Return'].cumsum().apply(np.exp).iloc[-1]
        sharpe  = df['Strategy Return'].mean() / df['Strategy Return'].std() * np.sqrt(252)
        results.append({
            "Ticker":                tkr,
            "SMA Strat Return (x)":  round(strat_x, 2),
            "Buy & Hold Return (x)": round(bh_x,    2),
            "Sharpe Ratio":          round(sharpe,  2),
        })
    except Exception:
        st.warning(f"⚠️ Could not process data for {tkr}")

# === SUMMARY TABLE & CSV DOWNLOAD ===
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    st.markdown("### 📊 Strategy Performance Summary")
    st.dataframe(summary_df)
    csv = summary_df.to_csv().encode()
    st.download_button(
        "📥 Download Summary as CSV",
        data=csv,
        file_name="sma_backtest_results.csv",
        mime="text/csv"
    )

# === CHART PICKER & DETAILED PLOTS ===
if all_data:
    selected = st.selectbox(
        "📌 Select a ticker to view detailed charts:",
        list(all_data.keys())
    )
    df = all_data[selected]

    # — Signals chart —
    st.markdown(f"### 📉 {selected} Price & SMA Signals")
    buys  = df[(df.Signal == 1) & (df.Signal.shift(1) != 1)]
    sells = df[(df.Signal == -1) & (df.Signal.shift(1) != -1)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.Close,    label='Close Price')
    ax.plot(df.SMA20,    '--', label='SMA20')
    ax.plot(df.SMA50,    '--', label='SMA50')
    ax.plot(df.SMA200,   '--', label='SMA200')
    ax.scatter(buys.index,  buys.Close,  marker='^', color='green',  s=100, label='Buy')
    ax.scatter(sells.index, sells.Close, marker='v', color='red',    s=100, label='Sell')
    ax.set_title(f"{selected} — SMA Crossover Signals")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(); ax.grid()
    ax.text(
        0.99, 0.01,
        "Created by Matteo Vito Bontempo\nmatteovbontempo@gmail.com",
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=8, color='gray', alpha=0.6
    )
    st.pyplot(fig)

    # — Cumulative returns chart —
    st.markdown("### 📈 Cumulative Return: Strategy vs Buy & Hold")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    df[['Daily Return','Strategy Return']].cumsum().apply(np.exp).plot(ax=ax2)
    ax2.set_title(f"{selected} — Growth of $1 Invested")
    ax2.set_ylabel("Growth (x)")
    ax2.grid()
    ax2.tick_params(axis='x', labelrotation=0)
    st.pyplot(fig2)
