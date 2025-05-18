import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ─── 0) PAGE CONFIG (must be first) ─────────────────────────────────
st.set_page_config(
    page_title="Multi-Asset SMA Backtesting Tool | Matteo Bontempo",
    layout="wide",
)

# ─── 0.5) GLOBAL CSS: force sans‐serif ───────────────────────────────
st.markdown(
    """
    <style>
      html, body, [class*="css"] {
        font-family: 'Helvetica', Arial, sans-serif !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── 1) SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Matteo Vito Bontempo")
    st.markdown(
        "B.S. in Accounting & CIS – Dec 2025  \n"
        "North Carolina Wesleyan University  \n"
        "From Argentina"
    )
    st.markdown("---")

    # 📱 Social icons (hosted on Icons8)
    social = {
        "https://github.com/matteovbontempo":             "https://img.icons8.com/ios-filled/50/000000/github.png",
        "https://www.linkedin.com/in/matteo-bontempo-/":  "https://img.icons8.com/ios-filled/50/000000/linkedin.png",
        "https://www.instagram.com/vito.bontempo/":       "https://img.icons8.com/ios-filled/50/000000/instagram-new.png",
        "mailto:matteovbontempo@gmail.com":               "https://img.icons8.com/ios-filled/50/000000/apple-mail.png",
    }
    cols = st.columns(len(social))
    for (link, icon), col in zip(social.items(), cols):
        col.markdown(
            f'[<img src="{icon}" width="30"/>]({link})',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### About Me")
    st.write(
        """
        I'm a student from Argentina pursuing a double major in Computer
        Information Systems (CIS) and Accounting, with plans to graduate in
        December 2025.

        My academic journey reflects my deep interest in combining technology
        with business to create efficient, data-driven solutions.

        I gained valuable experience during the COVID-19 pandemic working in
        a medical lab, uploading test results to the national health system
        and sending lab reports to patients.

        Currently, I’m interning at CoolGeeks: repairing electronics,
        building websites & apps, and assisting with networking tasks.

        I’m deeply interested in the financial world and aspire to build
        powerful, data-driven tools for investors.
        """
    )

# ─── 2) PAGE HEADER ─────────────────────────────────────────────────
st.title("Multi-Asset SMA Backtesting Tool")
st.markdown(
    "Backtest a moving average crossover strategy (SMA20 vs SMA50) on "
    "multiple assets and compare performance."
)

# ─── 3) USER INPUTS ─────────────────────────────────────────────────
tickers_input = st.text_input("Enter multiple tickers (comma-separated):", "AAPL, MSFT, TSLA")
tickers       = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date   = st.date_input("End Date",   value=pd.to_datetime("today"))

# ─── 4) LIVE DATA PREVIEW ───────────────────────────────────────────
if tickers:
    st.markdown("### Live Price Preview")
    preview_cols = st.columns(len(tickers))
    for col, tkr in zip(preview_cols, tickers):
        try:
            info    = yf.Ticker(tkr).info
            current = info.get("currentPrice", np.nan)
            pctchg  = info.get("regularMarketChangePercent", np.nan)
            hi52    = info.get("fiftyTwoWeekHigh", np.nan)
            lo52    = info.get("fiftyTwoWeekLow",  np.nan)
            col.metric(label=tkr, value=f"${current:.2f}", delta=f"{pctchg:.2f}%")
            col.write(f"52w High: ${hi52:.2f}")
            col.write(f"52w Low : ${lo52:.2f}")
        except:
            col.write(f"⚠️ Data unavailable for {tkr}")

# ─── 5) DATA LOADER (cached) ────────────────────────────────────────
@st.cache_data
def load_data(ticker, start, end, sma_short=20, sma_long=50):
    df = yf.download(ticker, start=start, end=end)
    df["SMA_Short"]       = df["Close"].rolling(sma_short).mean()
    df["SMA_Long"]        = df["Close"].rolling(sma_long).mean()
    df["Signal"]          = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1
    df["Position"]        = df["Signal"].shift(1)
    df["Daily Return"]    = df["Close"].pct_change()
    df["Strategy Return"] = df["Daily Return"] * df["Position"]
    return df

# ─── 6) BACKTEST LOOP ───────────────────────────────────────────────
results, all_data = [], {}
for tkr in tickers:
    try:
        df = load_data(tkr, start_date, end_date)
        all_data[tkr] = df

        strat = df["Strategy Return"].cumsum().apply(np.exp).iloc[-1]
        bh    = df["Daily Return"].cumsum().apply(np.exp).iloc[-1]
        sr    = df["Strategy Return"].mean() / df["Strategy Return"].std() * np.sqrt(252)

        results.append({
            "Ticker":                tkr,
            "SMA Return (×)":        round(strat, 2),
            "Buy & Hold (×)":        round(bh,    2),
            "Sharpe Ratio":          round(sr,    2),
        })
    except:
        st.warning(f"⚠️ Could not process {tkr}")

# ─── 7) SUMMARY TABLE ────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    st.markdown("### Performance Summary")
    st.dataframe(summary_df)
    csv = summary_df.to_csv().encode()
    st.download_button("Download CSV", data=csv, file_name="results.csv", mime="text/csv")

# ─── 8) DETAILED CHARTS ──────────────────────────────────────────────
if all_data:
    sel = st.selectbox("Select ticker to view charts:", list(all_data.keys()))
    df  = all_data[sel]

    # — Price + Signals
    st.markdown(f"### {sel} Price & SMA Signals")
    buy  = df[(df.Signal == 1) & (df.Signal.shift(1) != 1)]
    sell = df[(df.Signal == -1) & (df.Signal.shift(1) != -1)]
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df.Close, label="Close")
    ax.plot(df.SMA_Short, "--", label="SMA20")
    ax.plot(df.SMA_Long,  "--", label="SMA50")
    ax.scatter(buy.index,  buy.Close,  marker="^", color="green", label="Buy",  s=80)
    ax.scatter(sell.index, sell.Close, marker="v", color="red",   label="Sell", s=80)
    ax.set(title=f"{sel} – SMA Crossover", xlabel="Date", ylabel="Price ($)")
    ax.legend(); ax.grid()
    st.pyplot(fig)

    # — Cumulative Growth
    st.markdown("### Growth of $1 Invested")
    fig2, ax2 = plt.subplots(figsize=(14,5))
    df[["Daily Return","Strategy Return"]].cumsum().apply(np.exp).plot(ax=ax2)
    ax2.set(title=f"{sel} – Strategy vs Buy & Hold", ylabel="Growth (×)")
    ax2.grid(); ax2.tick_params(axis="x", rotation=0)
    st.pyplot(fig2)
