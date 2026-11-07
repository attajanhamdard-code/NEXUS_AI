import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ============================================================
# NEXUS AI - FULL SYSTEM (Engines 1-6 + Telegram Alerts)
# ============================================================

st.set_page_config(page_title="NEXUS AI", page_icon="📊", layout="wide")

# 🟢 TELEGRAM ALERT SETTINGS
TELEGRAM_BOT_TOKEN = "9068532808:AAG0gZcP1V7Ebo0wQm_YWxvTzyoVuLBiKrM"
TELEGRAM_CHAT_ID = "6382027867"

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)
    except Exception as e:
        pass

st.title("📊 NEXUS AI")
st.caption("Market Research | MTF | SMC Structure | Backtesting | Validation")

# ============================================================
# 1. MARKET SETTINGS
# ============================================================
SYMBOLS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "Gold Futures": "GC=F", "Bitcoin": "BTC-USD"
}
HTF_MAP = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1wk"}
TF_DURATION = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1), "1wk": pd.Timedelta(days=7)}

# ============================================================
# 2. SIDEBAR
# ============================================================
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

st.sidebar.divider()
st.sidebar.subheader("Validation Settings")
train_fraction = st.sidebar.slider("Training Share", min_value=0.50, max_value=0.85, value=0.70, step=0.05)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# 3. DATA DOWNLOAD
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_raw(ticker, interval, period):
    try:
        data = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False, threads=False)
        if data is None or data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.loc[:, ~data.columns.duplicated()].copy()
        required = ["Open", "High", "Low", "Close"]
        if not all(c in data.columns for c in required): return pd.DataFrame()
        if "Volume" not in data.columns: data["Volume"] = 0.0
        data = data[required + ["Volume"]].copy()
        for col in data.columns: data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=required)
        data = data[~data.index.duplicated(keep="last")]
        return data.sort_index()
    except Exception:
        return pd.DataFrame()

def resample_ohlcv(data, rule):
    if data.empty: return data.copy()
    out = data.resample(rule, label="left", closed="left").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return out.dropna(subset=["Open", "High", "Low", "Close"])

def download_data(ticker, tf):
    if tf == "15m":
        data = download_raw(ticker, "15m", "60d")
        if data.empty or len(data) < 100: data = download_raw(ticker, "15m", "30d")
        return data
    if tf == "1h": return download_raw(ticker, "1h", "730d")
    if tf == "4h": return resample_ohlcv(download_raw(ticker, "1h", "730d"), "4h")
    if tf == "1d": return download_raw(ticker, "1d", "5y")
    if tf == "1wk": return resample_ohlcv(download_raw(ticker, "1d", "5y"), "W-FRI")
    raise ValueError(f"Unsupported timeframe: {tf}")

# ============================================================
# 4. INDICATORS & SMC
# ============================================================
def calculate_indicators(data):
    df = data.copy()
    close = df["Close"]
    df["EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), "RSI"] = 0

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    prev_close = close.shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return df

def calculate_smc(data, left=3, right=3):
    df = data.copy()
    df["Swing_High_Plot"] = np.nan
    df["Swing_Low_Plot"] = np.nan
    df["Structure_Event"] = ""
    last_high, last_low, trend = None, None, None
    high_used, low_used = False, False

    for i in range(left + right, len(df)):
        pivot = i - right
        hi = df["High"].iloc[pivot-left:pivot+right+1]
        lo = df["Low"].iloc[pivot-left:pivot+right+1]
        
        if df["High"].iloc[pivot] == hi.max():
            last_high = df["High"].iloc[pivot]
            high_used = False
            df.iloc[pivot, df.columns.get_loc("Swing_High_Plot")] = last_high
            
        if df["Low"].iloc[pivot] == lo.min():
            last_low = df["Low"].iloc[pivot]
            low_used = False
            df.iloc[pivot, df.columns.get_loc("Swing_Low_Plot")] = last_low

        close_i = df["Close"].iloc[i]
        event = ""
        if last_high is not None and not high_used and close_i > last_high:
            event = "Bullish CHoCH" if trend == "bearish" else "Bullish BOS"
            trend, high_used = "bullish", True
        elif last_low is not None and not low_used and close_i < last_low:
            event = "Bearish CHoCH" if trend == "bullish" else "Bearish BOS"
            trend, low_used = "bearish", True
            
        df.iloc[i, df.columns.get_loc("Structure_Event")] = event
    return df

# ============================================================
# 5. TREND & HTF ALIGNMENT
# ============================================================
def add_trend(data):
    df = data.copy()
    df["Trend"] = "NEUTRAL"
    valid = df[["Close", "EMA20", "EMA50"]].notna().all(axis=1)
    df.loc[valid & (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"]), "Trend"] = "BULLISH"
    df.loc[valid & (df["Close"] < df["EMA20"]) & (df["EMA20"] < df["EMA50"]), "Trend"] = "BEARISH"
    return df

def normalize_timezone(series):
    idx = pd.DatetimeIndex(series)
    if idx.tz is not None: idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

def align_htf_trend(primary, htf, primary_tf, higher_tf):
    if primary.empty or htf.empty: return pd.Series("UNKNOWN", index=primary.index)
    p_duration = TF_DURATION[primary_tf]
    h_duration = TF_DURATION[higher_tf]
    p_index = normalize_timezone(primary.index)
    h_index = normalize_timezone(htf.index)
    
    p = pd.DataFrame({"available_at": p_index + p_duration, "original_index": range(len(primary))})
    h = pd.DataFrame({"htf_available_at": h_index + h_duration, "htf_trend": htf["Trend"].to_numpy()})
    
    p, h = p.sort_values("available_at"), h.sort_values("htf_available_at")
    merged = pd.merge_asof(p, h, left_on="available_at", right_on="htf_available_at", direction="backward", allow_exact_matches=True)
    merged = merged.sort_values("original_index")
    
    result = merged["htf_trend"].fillna("UNKNOWN")
    result.index = primary.index
    return result.astype(str)

# ============================================================
# 6. SIGNAL & BACKTEST
# ============================================================
def generate_signal(row, htf_trend):
    needed = ["RSI", "ATR", "EMA20", "EMA50", "MACD", "MACD_Signal"]
    if any(pd.isna(row.get(c)) for c in needed): return "WAIT"
    bull = (row["Close"] > row["EMA20"] > row["EMA50"] and row["RSI"] > 50 and row["MACD"] > row["MACD_Signal"] and htf_trend == "BULLISH" and row["Structure_Event"] in ("Bullish BOS", "Bullish CHoCH"))
    bear = (row["Close"] < row["EMA20"] < row["EMA50"] and row["RSI"] < 50 and row["MACD"] < row["MACD_Signal"] and htf_trend == "BEARISH" and row["Structure_Event"] in ("Bearish BOS", "Bearish CHoCH"))
    if bull: return "BUY"
    if bear: return "SELL"
    return "WAIT"

def run_backtest(df, htf_trends, daily_limit, max_losses, max_dd, spread_value, slippage_value, commission, start_index=1):
    trades, curve = [], []
    position, daily_pnl = None, {}
    consecutive_losses, paused_global = 0, False
    equity, peak, max_drawdown = 0.0, 0.0, 0.0
    entries, forced = 0, 0

    def entry_fill(price, side):
        adj = (spread_value / 2 + slippage_value)
        return price + adj if side == "BUY" else price - adj

    def exit_fill(price, side):
        adj = (spread_value / 2 + slippage_value)
        return price - adj if side == "BUY" else price + adj

    def close_position(time, raw_exit, reason):
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
        
        day_key = str(pd.Timestamp(time).date())
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + net_r
        consecutive_losses = consecutive_losses + 1 if net_r < 0 else 0
        if reason == "FORCED_CLOSE": forced += 1
        
        trades.append({"Entry Time": position["entry_time"], "Exit Time": time, "Side": side, "Entry": position["entry"], "Exit": fill, "Stop Loss": position["sl"], "Take Profit": position["tp"], "Gross Result (R)": gross_r, "Result (R)": net_r, "Exit Reason": reason, "Trade Type": "End-of-data" if reason == "FORCED_CLOSE" else "SL/TP"})
        curve.append({"Time": time, "Equity (R)": equity, "Drawdown (R)": dd})
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

    summary = {"Entries": entries, "Closed": len(result_df), "Win Rate (%)": (len(wins) / len(result_df) * 100), "Expectancy (R)": result_df["Result (R)"].mean(), "Profit Factor": pf, "Net Result (R)": result_df["Result (R)"].sum(), "Max Drawdown (R)": max_drawdown, "Paused": paused_global, "Forced": forced}
    return result_df, curve_df, summary

# ============================================================
# 7. MAIN APP EXECUTION
# ============================================================
try:
    with st.spinner("Downloading market history..."):
        primary_raw = download_data(symbol, timeframe)
        htf_raw = download_data(symbol, htf_timeframe)

    if primary_raw.empty or htf_raw.empty: st.error("No data returned."); st.stop()

    if exclude_latest:
        if len(primary_raw) > 1: primary_raw = primary_raw.iloc[:-1].copy()
        if len(htf_raw) > 1: htf_raw = htf_raw.iloc[:-1].copy()

    if len(primary_raw) < 100 or len(htf_raw) < 60: st.error("Insufficient data."); st.stop()

    primary = calculate_indicators(primary_raw)
    primary = calculate_smc(primary)
    primary = add_trend(primary)
    higher = add_trend(calculate_indicators(htf_raw))
    
    htf_trend_series = align_htf_trend(primary, higher, timeframe, htf_timeframe)
    primary["HTF_Trend"] = htf_trend_series
    primary["Signal"] = [generate_signal(primary.iloc[i], htf_trend_series.iloc[i]) for i in range(len(primary))]

except Exception as exc:
    st.error(f"Data error: {exc}")
    st.stop()

# ============================================================
# 8. DASHBOARD & SIGNAL
# ============================================================
latest = primary.iloc[-1]
st.divider()
st.header("📡 Live Research Dashboard")

a, b, c, d = st.columns(4)
a.metric("Market", market_name)
b.metric("Last Close", f"{latest['Close']:.5f}")
c.metric("Primary Trend", latest["Trend"])
d.metric("HTF Trend", latest["HTF_Trend"])

st.subheader("🧠 Signal Engine")
sig = latest["Signal"]

if sig == "BUY":
    st.success("Bullish conditions detected — Signal: BUY")
    send_telegram_alert(f"🚀 **NEXUS AI ALERT**\n\nSymbol: {market_name}\nTimeframe: {timeframe}\nSignal: **BUY**\nPrice: {latest['Close']:.5f}")
elif sig == "SELL":
    st.error("Bearish conditions detected — Signal: SELL")
    send_telegram_alert(f"🔻 **NEXUS AI ALERT**\n\nSymbol: {market_name}\nTimeframe: {timeframe}\nSignal: **SELL**\nPrice: {latest['Close']:.5f}")
else:
    st.warning("No valid signal — WAIT")

# ============================================================
# 9. CHART & BACKTEST METRICS
# ============================================================
st.divider()
st.header("📈 Advanced Chart & Backtest")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.60, 0.20, 0.20])
fig.add_trace(go.Candlestick(x=primary.index, open=primary["Open"], high=primary["High"], low=primary["Low"], close=primary["Close"], name="Candles"), row=1, col=1)
fig.add_trace(go.Scatter(x=primary.index, y=primary["EMA20"], name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=primary.index, y=primary["EMA50"], name="EMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=primary.index, y=primary["RSI"], name="RSI"), row=2, col=1)
fig.add_trace(go.Scatter(x=primary.index, y=primary["MACD"], name="MACD"), row=3, col=1)
fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧪 Historical Backtest Results")
res, curve, stats = run_backtest(primary, htf_trend_series, daily_loss_limit, int(max_losses), max_dd_limit, spread, slippage, commission_r)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entries", stats["Entries"])
c2.metric("Win Rate", f"{stats['Win Rate (%)']:.1f}%")
# 🟢 دلته تېروتنه سمه شوه
c3.metric("Net Result", f"{stats['Net Result (R)']:.2f} R")
c4.metric("Profit Factor", f"{stats['Profit Factor']:.2f}" if not pd.isna(stats["Profit Factor"]) and not np.isinf(stats["Profit Factor"]) else "N/A")

# ============================================================
# 10. MULTI-MARKET SCANNER
# ============================================================
st.divider()
st.header("🌍 Multi-Market Scanner")
if st.button("🚀 Run Multi-Market Scan", use_container_width=True):
    with st.spinner("Scanning..."):
        results = []
        for m_name, m_symbol in SYMBOLS.items():
            try:
                m_df = add_trend(calculate_smc(calculate_indicators(download_data(m_symbol, timeframe))))
                m_htf = add_trend(calculate_indicators(download_data(m_symbol, htf_timeframe)))
                m_htf_trends = align_htf_trend(m_df, m_htf, timeframe, htf_timeframe)
                _, _, m_stats = run_backtest(m_df, m_htf_trends, daily_loss_limit, int(max_losses), max_dd_limit, spread, slippage, commission_r)
                results.append({"Market": m_name, "Trades": m_stats["Closed"], "Win Rate (%)": round(m_stats["Win Rate (%)"], 1), "Net Result (R)": round(m_stats["Net Result (R)"], 2)})
            except:
                pass
        st.dataframe(pd.DataFrame(results), use_container_width=True)
