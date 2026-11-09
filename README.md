import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import plotly.graph_objects as go
from datetime import datetime, timezone
import ta
import requests
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# =========================================================
# NEXUS AI - ULTIMATE SCALPER TERMINAL (Syntax Fixed)
# =========================================================

st.set_page_config(page_title="NEXUS AI Scalper", page_icon="⚡", layout="wide")

# 🟢 هره ۱۰ ثانیې اپلیکیشن تازه کیږي (ترڅو ټایمر کار وکړي)
st_autorefresh(interval=10 * 1000, key="timerexit")

# -----------------------------
# MT5 CONNECTION
# -----------------------------
@st.cache_resource
def initialize_mt5():
    if not mt5.initialize(): return False
    return True

if not initialize_mt5():
    st.error("❌ MT5 Connection Failed! Open MT5 and login to DEMO.")
    st.stop()

account_info = mt5.account_info()
if account_info is None:
    st.error("❌ Could not fetch account info.")
    st.stop()

# -----------------------------
# TELEGRAM SETTINGS
# -----------------------------
def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        if value: return str(value)
    except Exception: pass
    return os.getenv(name, default)

TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_TOKEN_HERE": return False, "Telegram not configured."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=60)
        return True, "Sent" if response.json().get("ok") else "Failed"
    except: return False, "Network Error"

# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.header("⚡ MT5 Account Info")
st.sidebar.success(f"👤 Account: {account_info.login}")
st.sidebar.info(f"💰 Balance: ${account_info.balance:.2f}")

st.sidebar.divider()
st.sidebar.header("⚙️ Scalper Settings")
SYMBOL = st.sidebar.selectbox("Market", ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"])
TIMEFRAME_STR = st.sidebar.selectbox("Timeframe", ["M15", "M5"])
TIMEFRAME_MAP = {"M15": mt5.TIMEFRAME_M15, "M5": mt5.TIMEFRAME_M5}
TIMEFRAME_MT5 = TIMEFRAME_MAP[TIMEFRAME_STR]

RISK_PERCENT = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.5)
AUTO_TRADING = st.sidebar.checkbox("Enable Auto-Trading", value=False)
SCALP_MODE = st.sidebar.checkbox("15-Minute Scalper Mode", value=True)

if SYMBOL == "USDJPY": PIP_VALUE = 0.01
elif SYMBOL == "XAUUSD": PIP_VALUE = 0.1
else: PIP_VALUE = 0.0001

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_mt5_data(symbol, timeframe, num_candles=1000):
    if not mt5.symbol_select(symbol, True): return pd.DataFrame()
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None or len(rates) == 0: return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return df

def calculate_indicators(df):
    df = df.copy()
    df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # Candle Psychology (Price Action)
    body = (df['Close'] - df['Open']).abs()
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    df['Price_Action'] = ""
    df.loc[(df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & 
           (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1)), 'Price_Action'] = "Bullish Engulfing"
    df.loc[(df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & 
           (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1)), 'Price_Action'] = "Bearish Engulfing"
    df.loc[(lower_wick > body * 2) & (upper_wick < body), 'Price_Action'] = "Hammer"
    df.loc[(upper_wick > body * 2) & (lower_wick < body), 'Price_Action'] = "Shooting Star"

    # SMC
    df['Swing_High'] = (df['High'] == df['High'].rolling(5, center=True).max()).shift(5).fillna(False)
    df['Swing_Low'] = (df['Low'] == df['Low'].rolling(5, center=True).min()).shift(5).fillna(False)
    df['Last_Swing_High'] = df['High'].where(df['Swing_High']).ffill()
    df['Last_Swing_Low'] = df['Low'].where(df['Swing_Low']).ffill()
    df['SMC_Event'] = ""
    df.loc[(df['Close'] > df['Last_Swing_High'].shift(1)), 'SMC_Event'] = "Bullish BOS"
    df.loc[(df['Close'] < df['Last_Swing_Low'].shift(1)), 'SMC_Event'] = "Bearish BOS"
    return df

def execute_trade(symbol, order_type, lot, price, sl, tp):
    order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot,
        "type": order_type_mt5, "price": price, "sl": sl, "tp": tp,
        "deviation": 20, "magic": 123456, "comment": "Nexus_Scalp",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

def close_position(ticket, symbol, pos_type, volume):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return False
    price = tick.bid if pos_type == mt5.POSITION_TYPE_BUY else tick.ask
    order_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "position": ticket, "symbol": symbol,
        "volume": volume, "type": order_type, "price": price,
        "deviation": 20, "magic": 123456, "comment": "Nexus_TimeExit",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE

def check_and_close_old_trades():
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            if time.time() - pos.time > 900:
                st.toast(f"⏱️ Trade #{pos.ticket} reached 15 mins. Closing now...")
                close_position(pos.ticket, pos.symbol, pos.type, pos.volume)

# =========================================================
# MAIN UI TABS
# =========================================================
st.title("⚡ NEXUS AI - Ultimate Scalper")

tab_live, tab_news, tab_tools = st.tabs(["⚡ Live Trading", "🌍 Economic News", "🧭 Research Tools"])

# =========================================================
# TAB 1: LIVE TRADING
# =========================================================
with tab_live:
    check_and_close_old_trades()
    df = get_mt5_data(SYMBOL, TIMEFRAME_MT5)
    if df.empty:
        st.error(f"Could not fetch data for {SYMBOL}.")
    else:
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        signal = "WAIT"
        entry_price = latest['Close']
        sl_price = 0.0
        tp_price = 0.0
        lot_size = 0.0
        risk_amount = 0.0

        if (latest['EMA20'] > latest['EMA50'] and latest['RSI'] > 50 and latest['MACD'] > latest['MACD_Signal'] and (latest['SMC_Event'] == "Bullish BOS" or previous['SMC_Event'] == "Bullish BOS")):
            signal = "BUY"
        elif (latest['EMA20'] < latest['EMA50'] and latest['RSI'] < 50 and latest['MACD'] < latest['MACD_Signal'] and (latest['SMC_Event'] == "Bearish BOS" or previous['SMC_Event'] == "Bearish BOS")):
            signal = "SELL"

        if signal in ["BUY", "SELL"]:
            if SCALP_MODE:
                if signal == "BUY":
                    sl_price = entry_price - (10 * PIP_VALUE)
                    tp_price = entry_price + (15 * PIP_VALUE)
                else:
                    sl_price = entry_price + (10 * PIP_VALUE)
                    tp_price = entry_price - (15 * PIP_VALUE)
            else:
                atr = latest['ATR']
                if signal == "BUY":
                    sl_price = entry_price - (1.5 * atr)
                    tp_price = entry_price + (2.0 * atr)
                else:
                    sl_price = entry_price + (1.5 * atr)
                    tp_price = entry_price - (2.0 * atr)
                
            balance = account_info.balance
            risk_amount = balance * (RISK_PERCENT / 100)
            sl_distance = abs(entry_price - sl_price)
            symbol_info = mt5.symbol_info(SYMBOL)
            
            if symbol_info:
                tick_value = symbol_info.trade_tick_value
                tick_size = symbol_info.trade_tick_size
                if tick_size > 0:
                    sl_ticks = sl_distance / tick_size
                    if sl_ticks > 0 and tick_value > 0:
                        lot_size = round(risk_amount / (sl_ticks * tick_value), 2)
                        if lot_size < symbol_info.volume_min: lot_size = symbol_info.volume_min
                        if lot_size > symbol_info.volume_max: lot_size = symbol_info.volume_max

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{entry_price:.5f}")
        col2.metric("Signal", signal)
        col3.metric("Risk Amount", f"${risk_amount:.2f}" if signal != "WAIT" else "N/A")
        col4.metric("Lot Size", lot_size if signal != "WAIT" else "N/A")

        if signal != "WAIT":
            st.success(f"🎯 **{signal} Setup Detected!**")
            st.write(f"**Entry:** {entry_price:.5f} | **SL:** {sl_price:.5f} | **TP:** {tp_price:.5f}")
            st.write(f"**Latest Price Action:** {latest.get('Price_Action', 'None')}")
            st.info("⏱️ **Scalper Mode:** دا ټریډ به په ۱۵ دقیقو کې په زور سره وتړل شي که هدف یا سټاپ ته ونه رسیږي.")
            
            if st.button(f"🚀 EXECUTE {signal} (15-Min Scalp)", use_container_width=True):
                if not AUTO_TRADING:
                    st.error("⚠️ Enable 'Auto-Trading' in the sidebar first.")
                else:
                    with st.spinner("Sending order..."):
                        result = execute_trade(SYMBOL, signal, lot_size, entry_price, sl_price, tp_price)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            st.balloons()
                            st.success(f"✅ Trade Executed! Ticket: {result.order}")
                            send_telegram_alert(f"NEXUS AI: {signal} Executed on {SYMBOL}\nEntry: {entry_price}\nSL: {sl_price}\nTP: {tp_price}")
                        else:
                            st.error(f"❌ Trade Failed: {result.comment}")
        else:
            st.warning("⏳ No Valid Signal. Waiting for conditions...")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='cyan'), name='EMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange'), name='EMA 50'))
        
        bos_bull = df[df['SMC_Event'] == "Bullish BOS"]
        bos_bear = df[df['SMC_Event'] == "Bearish BOS"]
        fig.add_trace(go.Scatter(x=bos_bull.index, y=bos_bull['Low'] - (df['ATR'].mean()*0.5), mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Bullish BOS'))
        fig.add_trace(go.Scatter(x=bos_bear.index, y=bos_bear['High'] + (df['ATR'].mean()*0.5), mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Bearish BOS'))

        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2: NEWS
# =========================================================
with tab_news:
    st.header("🌍 Economic News")
    news_data = yf.Ticker("EURUSD=X").news
    if news_data:
        for item in news_data[:10]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            publisher = content.get("provider", {}).get("displayName") or item.get("publisher") or "Unknown"
            link = content.get("canonicalUrl", {}).get("url") or item.get("link")
            if title and link:
                st.markdown(f"**{title}**")
                st.caption(f"Source: {publisher}")
                st.markdown(f"[Read More]({link})")
                st.divider()

# =========================================================
# TAB 3: TOOLS (Currency Strength & Sessions)
# =========================================================
with tab_tools:
    st.header("🧭 Currency Strength & Sessions")
    
    # Currency Strength
    st.subheader("💱 Currency Strength (MT5)")
    if st.button("Calculate Strength"):
        pairs = {"EUR": ["EURUSD"], "GBP": ["GBPUSD"], "JPY": ["USDJPY"], "CHF": ["USDCHF"], "CAD": ["USDCAD"], "AUD": ["AUDUSD"], "NZD": ["NZDUSD"]}
        strengths = {}
        for curr, p_list in pairs.items():
            p_data = get_mt5_data(p_list[0], mt5.TIMEFRAME_D1, 20)
            if not p_data.empty and len(p_data) > 15:
                change = ((p_data['Close'].iloc[-1] - p_data['Close'].iloc[0]) / p_data['Close'].iloc[0]) * 100
                strengths[curr] = change
        strengths["USD"] = -sum(strengths.values()) / len(strengths) if strengths else 0
        st.dataframe(pd.DataFrame(list(strengths.items()), columns=["Currency", "Strength (%)"]).sort_values("Strength (%)", ascending=False))
    
    st.divider()
    st.subheader("🌐 Market Sessions")
    now = datetime.now(timezone.utc)
    hour = now.hour
    sessions = {"Sydney": (22, 7), "Tokyo": (0, 9), "London": (8, 17), "New York": (13, 22)}
    
    # 🟢 دلته مو نحوي تېروتنه سمه کړه
    active = []
    for name, (s, e) in sessions.items():
        if s < e:
            if s <= hour < e:
                active.append(name)
        else:
            if hour >= s or hour < e:
                active.append(name)
                
    st.write(f"Current UTC: {now.strftime('%H:%M')}")
    st.success(f"Active Sessions: {', '.join(active) if active else 'None'}")
