import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from datetime import datetime, timezone


# ============================================================
# NEXUS AI
# Educational Research & Simulation Only
# ============================================================

st.set_page_config(
    page_title="NEXUS AI",
    page_icon="📊",
    layout="wide"
)

SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "Gold Futures": "GC=F",
    "Bitcoin": "BTC-USD"
}

HTF_MAP = {
    "15m": "1h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1wk"
}

TF_DURATION = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1wk": pd.Timedelta(days=7)
}


# ============================================================
# TELEGRAM SECURITY
# ============================================================

def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")


def send_telegram_alert(message):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Telegram settings missing."

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=15
        )

        response.raise_for_status()

        if response.json().get("ok"):
            return True, "Message sent."

        return False, "Telegram rejected the message."

    except requests.RequestException as error:
        return False, str(error)


# ============================================================
# DATA DOWNLOAD ENGINE
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_raw(ticker, interval, period):

    try:

        data = yf.download(
            ticker,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.loc[
            :, ~data.columns.duplicated()
        ].copy()

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        if not all(
            column in data.columns
            for column in required
        ):
            return pd.DataFrame()

        if "Volume" not in data.columns:
            data["Volume"] = 0.0

        data = data[
            required + ["Volume"]
        ].copy()

        for column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        data = data.dropna(
            subset=required
        )

        data = data[
            ~data.index.duplicated(
                keep="last"
            )
        ]

        return data.sort_index()

    except Exception:
        return pd.DataFrame()


# ============================================================
# TIMEFRAME DATA
# ============================================================

def resample_ohlcv(data, rule):

    if data.empty:
        return data.copy()

    result = data.resample(
        rule,
        label="left",
        closed="left"
    ).agg({

        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"

    })

    return result.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )


def download_data(ticker, timeframe):

    if timeframe == "15m":
        return download_raw(
            ticker, "15m", "60d"
        )

    if timeframe == "1h":
        return download_raw(
            ticker, "1h", "730d"
        )

    if timeframe == "4h":

        hourly = download_raw(
            ticker, "1h", "730d"
        )

        return resample_ohlcv(
            hourly, "4h"
        )

    if timeframe == "1d":
        return download_raw(
            ticker, "1d", "5y"
        )

    if timeframe == "1wk":

        daily = download_raw(
            ticker, "1d", "5y"
        )

        return resample_ohlcv(
            daily, "W-FRI"
        )

    raise ValueError(
        f"Unsupported timeframe: {timeframe}"
    )


def remove_latest_incomplete_bar(data):

    if data.empty or len(data) < 2:
        return data.copy()

    return data.iloc[:-1].copy()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(data):

    df = data.copy()

    close = df["Close"]

    # EMA 20 / 50
    df["EMA20"] = close.ewm(
        span=20,
        adjust=False,
        min_periods=20
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False,
        min_periods=50
    ).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0, np.nan
    )

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "RSI"
    ] = 0

    df.loc[
        (avg_gain == 0) & (avg_loss == 0),
        "RSI"
    ] = np.nan

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False,
        min_periods=12
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
        min_periods=26
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df[
        "MACD"
    ].ewm(
        span=9,
        adjust=False,
        min_periods=9
    ).mean()

    # ATR 14
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    return df


# ============================================================
# SMC MARKET STRUCTURE
# ============================================================

def calculate_smc(data, left=3, right=3):

    df = data.copy()

    df["Swing_High_Plot"] = np.nan
    df["Swing_Low_Plot"] = np.nan
    df["Structure_Event"] = ""

    last_high = None
    last_low = None
    trend = None

    high_used = False
    low_used = False

    for i in range(left + right, len(df)):

        pivot = i - right

        high_window = df["High"].iloc[pivot-left:pivot+right+1]
        low_window = df["Low"].iloc[pivot-left:pivot+right+1]

        if df["High"].iloc[pivot] == high_window.max():
            last_high = float(df["High"].iloc[pivot])
            high_used = False
            df.iloc[pivot, df.columns.get_loc("Swing_High_Plot")] = last_high

        if df["Low"].iloc[pivot] == low_window.min():
            last_low = float(df["Low"].iloc[pivot])
            low_used = False
            df.iloc[pivot, df.columns.get_loc("Swing_Low_Plot")] = last_low

        close_price = float(df["Close"].iloc[i])
        event = ""

        if last_high is not None and not high_used and close_price > last_high:
            event = "Bullish CHoCH" if trend == "bearish" else "Bullish BOS"
            trend = "bullish"
            high_used = True

        elif last_low is not None and not low_used and close_price < last_low:
            event = "Bearish CHoCH" if trend == "bullish" else "Bearish BOS"
            trend = "bearish"
            low_used = True

        df.iloc[i, df.columns.get_loc("Structure_Event")] = event

    return df


# ============================================================
# TREND ENGINE
# ============================================================

def add_trend(data):

    df = data.copy()
    df["Trend"] = "NEUTRAL"

    valid = df[["Close", "EMA20", "EMA50"]].notna().all(axis=1)

    bullish = (
        valid
        & (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
    )

    bearish = (
        valid
        & (df["Close"] < df["EMA20"])
        & (df["EMA20"] < df["EMA50"])
    )

    df.loc[bullish, "Trend"] = "BULLISH"
    df.loc[bearish, "Trend"] = "BEARISH"

    return df


# ============================================================
# MULTI TIMEFRAME ALIGNMENT
# ============================================================

def normalize_datetime_index(index):

    idx = pd.DatetimeIndex(index)

    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)

    return idx


def align_htf_trend(primary, htf, primary_tf, higher_tf):

    if primary.empty or htf.empty:
        return pd.Series("UNKNOWN", index=primary.index)

    p_index = normalize_datetime_index(primary.index)
    h_index = normalize_datetime_index(htf.index)

    primary_data = pd.DataFrame({
        "available_at": p_index + TF_DURATION[primary_tf],
        "original_order": np.arange(len(primary))
    })

    higher_data = pd.DataFrame({
        "htf_available_at": h_index + TF_DURATION[higher_tf],
        "htf_trend": htf["Trend"].astype(str).to_numpy()
    })

    merged = pd.merge_asof(
        primary_data.sort_values("available_at"),
        higher_data.sort_values("htf_available_at"),
        left_on="available_at",
        right_on="htf_available_at",
        direction="backward",
        allow_exact_matches=True
    )

    merged = merged.sort_values("original_order")

    result = merged["htf_trend"].fillna("UNKNOWN")
    result.index = primary.index

    return result.astype(str)


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(row, htf_trend):

    required = ["RSI", "ATR", "EMA20", "EMA50", "MACD", "MACD_Signal"]

    if any(pd.isna(row.get(col)) for col in required):
        return "WAIT"

    bullish = (
        row["Close"] > row["EMA20"] > row["EMA50"]
        and row["RSI"] > 50
        and row["MACD"] > row["MACD_Signal"]
        and htf_trend == "BULLISH"
        and row["Structure_Event"] in ("Bullish BOS", "Bullish CHoCH")
    )

    bearish = (
        row["Close"] < row["EMA20"] < row["EMA50"]
        and row["RSI"] < 50
        and row["MACD"] < row["MACD_Signal"]
        and htf_trend == "BEARISH"
        and row["Structure_Event"] in ("Bearish BOS", "Bearish CHoCH")
    )

    if bullish:
        return "BUY"

    if bearish:
        return "SELL"

    return "WAIT"


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def run_backtest(df, htf_trends, daily_limit, max_losses, max_dd, spread_value, slippage_value, commission, start_index=1):

    trades = []
    curve = []

    position = None
    daily_pnl = {}

    consecutive_losses = 0
    paused_global = False

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    entries = 0
    forced = 0

    def entry_fill(raw_price, side):
        adjustment = (spread_value / 2.0 + slippage_value)
        return raw_price + adjustment if side == "BUY" else raw_price - adjustment

    def exit_fill(raw_price, side):
        adjustment = (spread_value / 2.0 + slippage_value)
        return raw_price - adjustment if side == "BUY" else raw_price + adjustment

    def close_position(timestamp, raw_exit, reason):
        nonlocal position, equity, peak, max_drawdown, consecutive_losses, forced

        side = position["side"]
        fill = exit_fill(float(raw_exit), side)
        pnl_price = (fill - position["entry"]) if side == "BUY" else (position["entry"] - fill)
        gross_r = pnl_price / position["risk_price"]
        net_r = gross_r - commission

        equity += net_r
        peak = max(peak, equity)
        dd = peak - equity
        max_drawdown = max(max_drawdown, dd)

        day_key = str(pd.Timestamp(timestamp).date())
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + net_r
        consecutive_losses = consecutive_losses + 1 if net_r < 0 else 0

        if reason == "FORCED_CLOSE":
            forced += 1

        trades.append({
            "Entry Time": position["entry_time"],
            "Exit Time": timestamp,
            "Side": side,
            "Entry": position["entry"],
            "Exit": fill,
            "Stop Loss": position["sl"],
            "Take Profit": position["tp"],
            "Gross Result (R)": gross_r,
            "Result (R)": net_r,
            "Exit Reason": reason,
            "Trade Type": "End-of-data" if reason == "FORCED_CLOSE" else "SL/TP"
        })

        curve.append({"Time": timestamp, "Equity (R)": equity, "Drawdown (R)": dd})
        position = None

    for i in range(max(start_index, 1), len(df)):
        bar = df.iloc[i]
        day_key = str(pd.Timestamp(df.index[i]).date())

        if position is not None:
            side = position["side"]
            exit_price, reason = None, None

            if side == "BUY":
                if bar["Open"] <= position["sl"]: exit_price, reason = float(bar["Open"]), "SL_GAP"
                elif bar["Open"] >= position["tp"]: exit_price, reason = float(position["tp"]), "TP_GAP"
            else:
                if bar["Open"] >= position["sl"]: exit_price, reason = float(bar["Open"]), "SL_GAP"
                elif bar["Open"] <= position["tp"]: exit_price, reason = float(position["tp"]), "TP_GAP"

            if exit_price is None:
                if side == "BUY":
                    if bar["Low"] <= position["sl"]: exit_price, reason = float(position["sl"]), "SL"
                    elif bar["High"] >= position["tp"]: exit_price, reason = float(position["tp"]), "TP"
                else:
                    if bar["High"] >= position["sl"]: exit_price, reason = float(position["sl"]), "SL"
                    elif bar["Low"] <= position["tp"]: exit_price, reason = float(position["tp"]), "TP"

            if exit_price is not None: close_position(df.index[i], exit_price, reason)

        if (peak - equity) >= max_dd: paused_global = True
        if consecutive_losses >= max_losses: paused_global = True
        if paused_global or daily_pnl.get(day_key, 0.0) <= -daily_limit: continue
        if i >= len(df) - 1 or position is not None: continue

        signal = generate_signal(df.iloc[i], htf_trends.iloc[i])
        if signal not in ("BUY", "SELL"): continue

        atr = df.iloc[i]["ATR"]
        if pd.isna(atr) or not np.isfinite(atr) or atr <= 0: continue

        entry = entry_fill(float(df.iloc[i + 1]["Open"]), signal)
        risk_price = 1.5 * float(atr)
        if risk_price <= 0: continue

        # TP = 2.0 ATR, SL = 1.5 ATR -> True R:R = 1:1.33
        if signal == "BUY": sl, tp = entry - risk_price, entry + 2.0 * float(atr)
        else: sl, tp = entry + risk_price, entry - 2.0 * float(atr)

        position = {"side": signal, "entry_time": df.index[i + 1], "entry": entry, "sl": sl, "tp": tp, "risk_price": risk_price}
        entries += 1

    if position is not None: close_position(df.index[-1], float(df.iloc[-1]["Close"]), "FORCED_CLOSE")

    result_df, curve_df = pd.DataFrame(trades), pd.DataFrame(curve)
    
    if result_df.empty:
        return result_df, curve_df, {"Entries": entries, "Closed": 0, "Win Rate (%)": 0.0, "Expectancy (R)": 0.0, "Profit Factor": np.nan, "Net Result (R)": 0.0, "Max Drawdown (R)": max_drawdown, "Paused": paused_global, "Forced": forced}

    wins, losses = result_df[result_df["Result (R)"] > 0], result_df[result_df["Result (R)"] < 0]
    gross_profit, gross_loss = wins["Result (R)"].sum(), abs(losses["Result (R)"].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)

    summary = {
        "Entries": entries, "Closed": len(result_df), 
        "Win Rate (%)": (len(wins) / len(result_df) * 100), 
        "Expectancy (R)": result_df["Result (R)"].mean(), 
        "Profit Factor": pf, "Net Result (R)": result_df["Result (R)"].sum(), 
        "Max Drawdown (R)": max_drawdown, "Paused": paused_global, "Forced": forced
    }
    return result_df, curve_df, summary


# ============================================================
# MAIN APPLICATION & UI
# ============================================================

st.title("📊 NEXUS AI")
st.caption("Market Research | MTF | SMC Structure | Backtesting | News Center")

tab_analysis, tab_news = st.tabs(["📈 Market Analysis & Backtest", "🌍 Economic News Center"])

with tab_analysis:
    st.sidebar.header("⚙️ System Settings")
    market_name = st.sidebar.selectbox("Market", list(SYMBOLS.keys()))
    symbol = SYMBOLS[market_name]

    timeframe = st.sidebar.selectbox("Primary Timeframe", list(HTF_MAP.keys()))
    htf_timeframe = HTF_MAP[timeframe]
    exclude_latest = st.sidebar.checkbox("Exclude latest candle", True)

    st.sidebar.divider()
    st.sidebar.subheader("Simulation Risk Controls")
    daily_loss_limit = st.sidebar.number_input("Daily Loss Limit (R)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
    max_losses = st.sidebar.number_input("Maximum Consecutive Losses", min_value=1, max_value=20, value=3)
    max_dd_limit = st.sidebar.number_input("Maximum Drawdown (R)", min_value=1.0, max_value=100.0, value=5.0, step=1.0)

    st.sidebar.divider()
    st.sidebar.subheader("Estimated Transaction Costs")
    spread = st.sidebar.number_input("Full Spread (Price Units)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    slippage = st.sidebar.number_input("Slippage Per Side", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    commission_r = st.sidebar.number_input("Commission Per Round Trip (R)", min_value=0.0, value=0.0, step=0.01)

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    try:
        with st.spinner("Downloading market history..."):
            primary_raw = download_data(symbol, timeframe)
            htf_raw = download_data(symbol, htf_timeframe)

        if primary_raw.empty or htf_raw.empty: 
            st.error("No data returned from Yahoo Finance."); st.stop()
            
        if exclude_latest:
            primary_raw = remove_latest_incomplete_bar(primary_raw)
            htf_raw = remove_latest_incomplete_bar(htf_raw)
            
        if len(primary_raw) < 100 or len(htf_raw) < 60: 
            st.error("Insufficient data."); st.stop()

        primary = add_trend(calculate_smc(calculate_indicators(primary_raw)))
        higher = add_trend(calculate_indicators(htf_raw))
        htf_trend_series = align_htf_trend(primary, higher, timeframe, htf_timeframe)
        
        primary["HTF_Trend"] = htf_trend_series
        primary["Signal"] = [generate_signal(primary.iloc[i], htf_trend_series.iloc[i]) for i in range(len(primary))]
    except Exception as exc:
        st.error(f"Data error: {exc}"); st.stop()

    latest = primary.iloc[-1]
    
    st.divider()
    st.header("📡 Live Research Dashboard")
    
    # Data Health
    last_candle_time = primary.index[-1]
    time_diff = datetime.now(timezone.utc) - last_candle_time
    st.caption(f"⏱️ Last Candle: {last_candle_time} | Data Freshness: {time_diff}")
    
    a, b, c, d = st.columns(4)
    a.metric("Market", market_name)
    b.metric("Last Close", f"{latest['Close']:.5f}")
    c.metric("Primary Trend", latest["Trend"])
    d.metric("HTF Trend", latest["HTF_Trend"])

    st.subheader("🧠 Signal Engine")
    sig = latest["Signal"]
    
    # 🟢 د تکراري پیغامونو مخنیوی (Anti-Duplication)
    if "last_sent_candle" not in st.session_state:
        st.session_state.last_sent_candle = None

    if sig == "BUY":
        st.success("Bullish conditions detected — Signal: BUY")
        if st.session_state.last_sent_candle != last_candle_time:
            success, msg = send_telegram_alert(f"🚀 **NEXUS AI ALERT**\n\nSymbol: {market_name}\nTimeframe: {timeframe}\nSignal: **BUY**\nPrice: {latest['Close']:.5f}")
            if success:
                st.session_state.last_sent_candle = last_candle_time
                st.toast("Telegram alert sent for this candle!")
    elif sig == "SELL":
        st.error("Bearish conditions detected — Signal: SELL")
        if st.session_state.last_sent_candle != last_candle_time:
            success, msg = send_telegram_alert(f"🔻 **NEXUS AI ALERT**\n\nSymbol: {market_name}\nTimeframe: {timeframe}\nSignal: **SELL**\nPrice: {latest['Close']:.5f}")
            if success:
                st.session_state.last_sent_candle = last_candle_time
                st.toast("Telegram alert sent for this candle!")
    else:
        st.warning("No valid signal — WAIT")

    st.divider()
    st.header("📈 Advanced Chart & Backtest")
    
    # Chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.60, 0.20, 0.20])
    fig.add_trace(go.Candlestick(x=primary.index, open=primary["Open"], high=primary["High"], low=primary["Low"], close=primary["Close"], name="Candles"), row=1, col=1)
    fig.add_trace(go.Scatter(x=primary.index, y=primary["EMA20"], name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=primary.index, y=primary["EMA50"], name="EMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=primary.index, y=primary["RSI"], name="RSI"), row=2, col=1)
    fig.add_trace(go.Scatter(x=primary.index, y=primary["MACD"], name="MACD"), row=3, col=1)
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Backtest
    res, curve, stats = run_backtest(primary, htf_trend_series, daily_loss_limit, int(max_losses), max_dd_limit, spread, slippage, commission_r)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries", stats["Entries"])
    c2.metric("Win Rate", f"{stats['Win Rate (%)']:.1f}%")
    c3.metric("Net Result", f"{stats['Net Result (R)']:.2f} R")
    c4.metric("Profit Factor", f"{stats['Profit Factor']:.2f}" if not pd.isna(stats["Profit Factor"]) and not np.isinf(stats["Profit Factor"]) else "N/A")
    
    st.info(f"ℹ️ **True Risk/Reward Ratio:** 1 : 1.33 (TP: 2.0 ATR / SL: 1.5 ATR)")

    if st.button("🚀 Run Multi-Market Scan", use_container_width=True):
        with st.spinner("Scanning all markets..."):
            results = []
            for m_name, m_symbol in SYMBOLS.items():
                try:
                    m_df_raw = download_data(m_symbol, timeframe)
                    m_htf_raw = download_data(m_symbol, htf_timeframe)
                    
                    if exclude_latest:
                        m_df_raw = remove_latest_incomplete_bar(m_df_raw)
                        m_htf_raw = remove_latest_incomplete_bar(m_htf_raw)
                        
                    m_df = add_trend(calculate_smc(calculate_indicators(m_df_raw)))
                    m_htf = add_trend(calculate_indicators(m_htf_raw))
                    m_htf_trends = align_htf_trend(m_df, m_htf, timeframe, htf_timeframe)
                    
                    _, _, m_stats = run_backtest(m_df, m_htf_trends, daily_loss_limit, int(max_losses), max_dd_limit, spread, slippage, commission_r)
                    results.append({
                        "Market": m_name, 
                        "Trades": m_stats["Closed"], 
                        "Win Rate (%)": round(m_stats["Win Rate (%)"], 1), 
                        "Net Result (R)": round(m_stats["Net Result (R)"], 2)
                    })
                except:
                    pass
            st.dataframe(pd.DataFrame(results), use_container_width=True)


# ============================================================
# ECONOMIC NEWS CENTER TAB
# ============================================================
with tab_news:
    st.header("🌍 Global Economic News & Calendar")
    st.write("دا برخه یوازې د معلوماتو او څیړنې لپاره ده. دا د اتوماتیک ټریډ کولو لپاره نه کارول کیږي.")
    
    st.markdown("---")
    st.subheader("📅 Upcoming High-Impact Events (Demo Data)")
    st.warning("⚠️ Attention: دا مهال ویش یوازې د نمونې (Demo) په توګه دی. د ژوندیو معلوماتو لپاره باید له رسمي API څخه کار واخلو.")
    
    calendar_data = [
        {"Date": "Dec 04, 2026", "Event": "Non-Farm Payrolls (NFP)", "Currency": "USD", "Importance": "High"},
        {"Date": "Dec 10, 2026", "Event": "Consumer Price Index (CPI)", "Currency": "USD", "Importance": "High"},
        {"Date": "Dec 18, 2026", "Event": "FOMC Meeting & Rate Decision", "Currency": "USD", "Importance": "High"},
        {"Date": "Dec 14, 2026", "Event": "ECB Interest Rate Decision", "Currency": "EUR", "Importance": "High"},
        {"Date": "Dec 19, 2026", "Event": "Bank of England (BoE) Rate Decision", "Currency": "GBP", "Importance": "High"}
    ]
    st.dataframe(pd.DataFrame(calendar_data), use_container_width=True)

    st.markdown("---")
    st.subheader("📰 Live News Feed (Yahoo Finance)")
    st.caption("سرچینه: Yahoo Finance | دا خبرونه ممکن لږ ځنډ ولري.")
    
    try:
        news_data = yf.Ticker(SYMBOLS["EUR/USD"]).news
        if news_data:
            for news in news_data[:5]:
                st.markdown(f"**{news.get('title', 'No Title')}**")
                st.write(f"Source: {news.get('publisher', 'Unknown')} | Published: {datetime.fromtimestamp(news.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')}")
                st.write(f"[Read More]({news.get('link', '#')})")
                st.markdown("---")
        else:
            st.warning("No recent news found for this market.")
    except Exception as e:
        st.error(f"Could not fetch news: {e}")

    st.markdown("---")
    st.caption("Official Sources: Bureau of Labor Statistics (BLS), Federal Reserve, European Central Bank, Bank of England.")
