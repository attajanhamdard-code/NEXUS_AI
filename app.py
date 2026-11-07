import os
import logging
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from datetime import datetime, timezone


# =========================================================
# NEXUS AI
# Educational Research & Historical Simulation Only
# No Broker Connection
# No Real-Money Execution
# =========================================================

st.set_page_config(
    page_title="NEXUS AI",
    page_icon="📊",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEXUS")


# =========================================================
# CONFIGURATION
# =========================================================

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

MIN_PRIMARY_BARS = 100
MIN_HTF_BARS = 60


# =========================================================
# TELEGRAM SETTINGS
# =========================================================

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
        return False, "Telegram is not configured."

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

        result = response.json()

        if result.get("ok"):
            return True, "Message sent."

        return False, "Telegram rejected the message."

    except requests.RequestException as error:
        logger.exception("Telegram error")
        return False, str(error)


# =========================================================
# DATETIME FUNCTIONS
# =========================================================

def normalize_datetime_index(index):

    idx = pd.DatetimeIndex(index)

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    return idx


def normalize_dataframe(data):

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    df.index = normalize_datetime_index(df.index)

    df = df.sort_index()

    df = df[~df.index.duplicated(keep="last")]

    return df


# =========================================================
# MARKET DATA
# =========================================================

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

        data = data.loc[:, ~data.columns.duplicated()]

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        if not all(col in data.columns for col in required):
            return pd.DataFrame()

        if "Volume" not in data.columns:
            data["Volume"] = 0.0

        data = data[
            required + ["Volume"]
        ].copy()

        for col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

        data = data.dropna(subset=required)

        data = normalize_dataframe(data)

        return data

    except Exception:

        logger.exception(
            "Download failed: %s",
            ticker
        )

        return pd.DataFrame()


# =========================================================
# RESAMPLING
# =========================================================

def resample_ohlcv(data, rule):

    if data.empty:
        return pd.DataFrame()

    df = normalize_dataframe(data)

    result = df.resample(
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

    result = result.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return result


def download_data(ticker, timeframe):

    if timeframe == "15m":
        return download_raw(ticker, "15m", "60d")

    if timeframe == "1h":
        return download_raw(ticker, "1h", "730d")

    if timeframe == "4h":

        hourly = download_raw(
            ticker,
            "1h",
            "730d"
        )

        return resample_ohlcv(hourly, "4h")

    if timeframe == "1d":
        return download_raw(ticker, "1d", "5y")

    if timeframe == "1wk":

        daily = download_raw(
            ticker,
            "1d",
            "5y"
        )

        return resample_ohlcv(daily, "W-FRI")

    raise ValueError(
        f"Unsupported timeframe: {timeframe}"
    )


# =========================================================
# COMPLETED CANDLES
# =========================================================

def remove_incomplete_bars(data, timeframe):

    if data.empty:
        return data.copy()

    df = normalize_dataframe(data)

    now = pd.Timestamp.now(tz="UTC")

    # Conservative handling for daily/weekly candles.
    # Exchange session calendars are not available here.

    if timeframe in ("1d", "1wk"):

        if len(df) > 1:
            return df.iloc[:-1].copy()

        return df.iloc[0:0].copy()

    duration = TF_DURATION.get(timeframe)

    if duration is None:
        raise ValueError("Unsupported timeframe")

    available_at = df.index + duration

    return df.loc[
        available_at <= now
    ].copy()


# =========================================================
# WILDER MOVING AVERAGE
# =========================================================

def wilder_rma(series, period):

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    if len(series) < period:
        return result

    first_value = series.iloc[:period].mean()

    result.iloc[period - 1] = first_value

    for i in range(period, len(series)):

        result.iloc[i] = (
            result.iloc[i - 1] * (period - 1)
            + series.iloc[i]
        ) / period

    return result


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(data):

    df = data.copy()

    close = df["Close"]

    # EMA

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

    # RSI

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = wilder_rma(
        gain.iloc[1:],
        14
    ).reindex(df.index)

    avg_loss = wilder_rma(
        loss.iloc[1:],
        14
    ).reindex(df.index)

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
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

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False,
        min_periods=9
    ).mean()

    df["MACD_Hist"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # ATR

    previous_close = close.shift(1)

    true_range = pd.concat([

        df["High"] - df["Low"],

        (df["High"] - previous_close).abs(),

        (df["Low"] - previous_close).abs()

    ], axis=1).max(axis=1)

    df["ATR"] = wilder_rma(
        true_range.iloc[1:],
        14
    ).reindex(df.index)

    return df


# =========================================================
# SMC STRUCTURE
# =========================================================

def calculate_smc(data, left=3, right=3):

    df = data.copy()

    df["Swing_High_Plot"] = np.nan
    df["Swing_Low_Plot"] = np.nan

    df["Structure_Event"] = ""
    df["Confirmed_Pivot_High"] = np.nan
    df["Confirmed_Pivot_Low"] = np.nan

    last_high = None
    last_low = None

    trend = None

    high_used = False
    low_used = False

    for i in range(left + right, len(df)):

        pivot = i - right

        high_window = df["High"].iloc[
            pivot-left:pivot+right+1
        ]

        low_window = df["Low"].iloc[
            pivot-left:pivot+right+1
        ]

        # Confirmed swing high

        if (
            df["High"].iloc[pivot]
            == high_window.max()
        ):

            last_high = float(
                df["High"].iloc[pivot]
            )

            high_used = False

            # Plot at confirmation candle,
            # not at the historical pivot candle.

            df.iloc[
                i,
                df.columns.get_loc(
                    "Swing_High_Plot"
                )
            ] = last_high

            df.iloc[
                i,
                df.columns.get_loc(
                    "Confirmed_Pivot_High"
                )
            ] = last_high

        # Confirmed swing low

        if (
            df["Low"].iloc[pivot]
            == low_window.min()
        ):

            last_low = float(
                df["Low"].iloc[pivot]
            )

            low_used = False

            df.iloc[
                i,
                df.columns.get_loc(
                    "Swing_Low_Plot"
                )
            ] = last_low

            df.iloc[
                i,
                df.columns.get_loc(
                    "Confirmed_Pivot_Low"
                )
            ] = last_low

        close_price = float(
            df["Close"].iloc[i]
        )

        event = ""

        if (
            last_high is not None
            and not high_used
            and close_price > last_high
        ):

            event = (
                "Bullish CHoCH"
                if trend == "bearish"
                else "Bullish BOS"
            )

            trend = "bullish"
            high_used = True

        elif (
            last_low is not None
            and not low_used
            and close_price < last_low
        ):

            event = (
                "Bearish CHoCH"
                if trend == "bullish"
                else "Bearish BOS"
            )

            trend = "bearish"
            low_used = True

        df.iloc[
            i,
            df.columns.get_loc(
                "Structure_Event"
            )
        ] = event

    return df


# =========================================================
# TREND ENGINE
# =========================================================

def add_trend(data):

    df = data.copy()

    df["Trend"] = "NEUTRAL"

    valid = df[
        ["Close", "EMA20", "EMA50"]
    ].notna().all(axis=1)

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


# =========================================================
# MULTI-TIMEFRAME ALIGNMENT
# =========================================================

def align_htf_trend(
    primary,
    htf,
    primary_tf,
    higher_tf
):

    if primary.empty or htf.empty:

        return pd.Series(
            "UNKNOWN",
            index=primary.index
        )

    primary = normalize_dataframe(primary)
    htf = normalize_dataframe(htf)

    primary_data = pd.DataFrame({

        "Time": primary.index,

        "Order": np.arange(
            len(primary)
        )

    })

    higher_data = pd.DataFrame({

        "HTF_Time": htf.index,

        "HTF_Trend": htf["Trend"].astype(
            str
        ).values

    })

    primary_data["Available"] = (
        primary_data["Time"]
        + TF_DURATION[primary_tf]
    )

    higher_data["Available"] = (
        higher_data["HTF_Time"]
        + TF_DURATION[higher_tf]
    )

    merged = pd.merge_asof(

        primary_data.sort_values("Available"),

        higher_data.sort_values("Available"),

        on="Available",

        direction="backward"

    )

    merged = merged.sort_values(
        "Order"
    )

    result = merged[
        "HTF_Trend"
    ].fillna("UNKNOWN")

    result.index = primary.index

    return result.astype(str)


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(row, htf_trend):

    required = [

        "RSI",
        "ATR",
        "EMA20",
        "EMA50",
        "MACD",
        "MACD_Signal"

    ]

    if any(
        pd.isna(row.get(col))
        for col in required
    ):

        return "WAIT"

    bullish = (

        row["Close"] > row["EMA20"]
        and row["EMA20"] > row["EMA50"]

        and row["RSI"] > 50

        and row["MACD"] > row["MACD_Signal"]

        and htf_trend == "BULLISH"

        and row["Structure_Event"]
        in (
            "Bullish BOS",
            "Bullish CHoCH"
        )

    )

    bearish = (

        row["Close"] < row["EMA20"]
        and row["EMA20"] < row["EMA50"]

        and row["RSI"] < 50

        and row["MACD"] < row["MACD_Signal"]

        and htf_trend == "BEARISH"

        and row["Structure_Event"]
        in (
            "Bearish BOS",
            "Bearish CHoCH"
        )

    )

    if bullish:
        return "BUY"

    if bearish:
        return "SELL"

    return "WAIT"


# =========================================================
# BACKTEST ENGINE
# =========================================================

def run_backtest(
    df,
    htf_trends,
    daily_limit,
    max_losses,
    max_dd,
    spread_value,
    slippage_value,
    commission,
    stop_atr=1.5,
    target_atr=2.0
):

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

    empty_summary = {

        "Entries": 0,
        "Closed": 0,
        "Win Rate (%)": 0.0,
        "Expectancy (R)": 0.0,
        "Profit Factor": np.nan,
        "Net Result (R)": 0.0,
        "Max Drawdown (R)": 0.0,
        "Paused": False,
        "Forced": 0

    }

    if df.empty or len(df) < 2:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            empty_summary
        )

    def entry_fill(price, side):

        adjustment = (
            spread_value / 2
            + slippage_value
        )

        if side == "BUY":
            return price + adjustment

        return price - adjustment

    def exit_fill(price, side):

        adjustment = (
            spread_value / 2
            + slippage_value
        )

        if side == "BUY":
            return price - adjustment

        return price + adjustment

    def close_position(
        timestamp,
        raw_exit,
        reason
    ):

        nonlocal position
        nonlocal equity
        nonlocal peak
        nonlocal max_drawdown
        nonlocal consecutive_losses
        nonlocal forced

        side = position["side"]

        fill = exit_fill(
            float(raw_exit),
            side
        )

        if side == "BUY":

            pnl_price = (
                fill - position["entry"]
            )

        else:

            pnl_price = (
                position["entry"] - fill
            )

        gross_r = (
            pnl_price / position["risk_price"]
        )

        net_r = gross_r - commission

        equity += net_r

        peak = max(peak, equity)

        drawdown = peak - equity

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

        day_key = str(
            pd.Timestamp(timestamp).date()
        )

        daily_pnl[day_key] = (
            daily_pnl.get(day_key, 0.0)
            + net_r
        )

        if net_r < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

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

            "Exit Reason": reason

        })

        curve.append({

            "Time": timestamp,
            "Equity (R)": equity,
            "Drawdown (R)": drawdown

        })

        position = None

    for i in range(1, len(df)):

        bar = df.iloc[i]

        timestamp = df.index[i]

        day_key = str(
            pd.Timestamp(timestamp).date()
        )

        # -------------------------------------
        # Manage existing position
        # -------------------------------------

        if position is not None:

            side = position["side"]

            exit_price = None
            reason = None

            # Gap handling

            if side == "BUY":

                if bar["Open"] <= position["sl"]:

                    exit_price = float(
                        bar["Open"]
                    )

                    reason = "SL_GAP"

                elif bar["Open"] >= position["tp"]:

                    exit_price = float(
                        position["tp"]
                    )

                    reason = "TP_GAP"

            else:

                if bar["Open"] >= position["sl"]:

                    exit_price = float(
                        bar["Open"]
                    )

                    reason = "SL_GAP"

                elif bar["Open"] <= position["tp"]:

                    exit_price = float(
                        position["tp"]
                    )

                    reason = "TP_GAP"

            # Intrabar handling
            # Conservative assumption:
            # SL is checked before TP.

            if exit_price is None:

                if side == "BUY":

                    if bar["Low"] <= position["sl"]:

                        exit_price = float(
                            position["sl"]
                        )

                        reason = "SL"

                    elif bar["High"] >= position["tp"]:

                        exit_price = float(
                            position["tp"]
                        )

                        reason = "TP"

                else:

                    if bar["High"] >= position["sl"]:

                        exit_price = float(
                            position["sl"]
                        )

                        reason = "SL"

                    elif bar["Low"] <= position["tp"]:

                        exit_price = float(
                            position["tp"]
                        )

                        reason = "TP"

            if exit_price is not None:

                close_position(
                    timestamp,
                    exit_price,
                    reason
                )

        # -------------------------------------
        # Risk controls
        # -------------------------------------

        current_drawdown = peak - equity

        if current_drawdown >= max_dd:
            paused_global = True

        if consecutive_losses >= max_losses:
            paused_global = True

        if paused_global:
            continue

        if daily_pnl.get(
            day_key,
            0.0
        ) <= -daily_limit:

            continue

        if position is not None:
            continue

        if i >= len(df) - 1:
            continue

        # -------------------------------------
        # Signal
        # -------------------------------------

        signal = generate_signal(
            df.iloc[i],
            htf_trends.iloc[i]
        )

        if signal not in ("BUY", "SELL"):
            continue

        atr = df.iloc[i]["ATR"]

        if (
            pd.isna(atr)
            or not np.isfinite(atr)
            or atr <= 0
        ):
            continue

        raw_entry = float(
            df.iloc[i + 1]["Open"]
        )

        entry = entry_fill(
            raw_entry,
            signal
        )

        risk_price = (
            stop_atr * float(atr)
        )

        if risk_price <= 0:
            continue

        if signal == "BUY":

            sl = entry - risk_price

            tp = (
                entry
                + target_atr * float(atr)
            )

        else:

            sl = entry + risk_price

            tp = (
                entry
                - target_atr * float(atr)
            )

        position = {

            "side": signal,

            "entry_time": df.index[i + 1],

            "entry": entry,

            "sl": sl,

            "tp": tp,

            "risk_price": risk_price

        }

        entries += 1

    # -------------------------------------
    # End of historical data
    # -------------------------------------

    if position is not None:

        close_position(
            df.index[-1],
            float(df.iloc[-1]["Close"]),
            "FORCED_CLOSE"
        )

    results = pd.DataFrame(trades)

    equity_curve = pd.DataFrame(curve)

    if results.empty:

        empty_summary["Entries"] = entries
        empty_summary["Paused"] = paused_global
        empty_summary["Forced"] = forced

        return (
            results,
            equity_curve,
            empty_summary
        )

    wins = results[
        results["Result (R)"] > 0
    ]

    losses = results[
        results["Result (R)"] < 0
    ]

    gross_profit = wins[
        "Result (R)"
    ].sum()

    gross_loss = abs(
        losses["Result (R)"].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = np.inf

    else:

        profit_factor = np.nan

    summary = {

        "Entries": entries,

        "Closed": len(results),

        "Win Rate (%)": (
            len(wins) / len(results) * 100
        ),

        "Expectancy (R)": (
            results["Result (R)"].mean()
        ),

        "Profit Factor": profit_factor,

        "Net Result (R)": (
            results["Result (R)"].sum()
        ),

        "Max Drawdown (R)": max_drawdown,

        "Paused": paused_global,

        "Forced": forced

    }

    return (
        results,
        equity_curve,
        summary
    )


# =========================================================
# NEWS ENGINE
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_yahoo_news(ticker):

    try:

        news_data = yf.Ticker(ticker).news

        valid_news = []

        if not news_data:
            return []

        for item in news_data[:15]:

            content = item.get(
                "content",
                item
            )

            title = (
                content.get("title")
                or item.get("title")
            )

            if not title:
                continue

            canonical = content.get(
                "canonicalUrl",
                {}
            )

            link = (
                canonical.get("url")
                or item.get("link")
            )

            if not link:
                continue

            publisher = (
                content.get(
                    "provider",
                    {}
                ).get("displayName")
                or item.get("publisher")
                or "Unknown"
            )

            pub_time = (
                content.get("pubDate")
                or item.get("providerPublishTime")
            )

            try:

                if isinstance(
                    pub_time,
                    (int, float)
                ):

                    published = datetime.fromtimestamp(
                        pub_time,
                        tz=timezone.utc
                    )

                else:

                    published = pd.to_datetime(
                        pub_time,
                        utc=True
                    ).to_pydatetime()

                published_text = published.strftime(
                    "%Y-%m-%d %H:%M UTC"
                )

            except Exception:

                published_text = "Time unavailable"

            valid_news.append({

                "Title": title,

                "Publisher": publisher,

                "Published": published_text,

                "Link": link

            })

        return valid_news

    except Exception:

        logger.exception(
            "News loading failed"
        )

        return []


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.title("📊 NEXUS AI")

st.caption(
    "Market Research | MTF | SMC | "
    "Backtesting | Economic News"
)

tab_analysis, tab_news, tab_system = st.tabs([

    "📈 Market Analysis",

    "🌍 Economic News",

    "⚙️ System Health"

])


# =========================================================
# MARKET ANALYSIS
# =========================================================

with tab_analysis:

    st.sidebar.header(
        "⚙️ System Settings"
    )

    market_name = st.sidebar.selectbox(
        "Market",
        list(SYMBOLS.keys())
    )

    symbol = SYMBOLS[market_name]

    timeframe = st.sidebar.selectbox(
        "Primary Timeframe",
        list(HTF_MAP.keys())
    )

    htf_timeframe = HTF_MAP[timeframe]

    exclude_latest = st.sidebar.checkbox(
        "Use completed candles only",
        True
    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "Simulation Risk Controls"
    )

    daily_loss_limit = st.sidebar.number_input(
        "Daily Loss Limit (R)",
        min_value=0.5,
        max_value=20.0,
        value=2.0,
        step=0.5
    )

    max_losses = st.sidebar.number_input(
        "Maximum Consecutive Losses",
        min_value=1,
        max_value=20,
        value=3
    )

    max_dd_limit = st.sidebar.number_input(
        "Maximum Drawdown (R)",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
        step=1.0
    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "Estimated Transaction Costs"
    )

    spread = st.sidebar.number_input(
        "Full Spread (Price Units)",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f"
    )

    slippage = st.sidebar.number_input(
        "Slippage Per Side",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f"
    )

    commission_r = st.sidebar.number_input(
        "Commission Per Round Trip (R)",
        min_value=0.0,
        value=0.0,
        step=0.01
    )

    if st.sidebar.button(
        "🔄 Refresh Data"
    ):

        st.cache_data.clear()
        st.rerun()

    try:

        with st.spinner(
            "Downloading market data..."
        ):

            primary_raw = download_data(
                symbol,
                timeframe
            )

            htf_raw = download_data(
                symbol,
                htf_timeframe
            )

        if (
            primary_raw.empty
            or htf_raw.empty
        ):

            st.error(
                "No market data available."
            )

            st.stop()

        if exclude_latest:

            primary_raw = remove_incomplete_bars(
                primary_raw,
                timeframe
            )

            htf_raw = remove_incomplete_bars(
                htf_raw,
                htf_timeframe
            )

        if (
            len(primary_raw) < MIN_PRIMARY_BARS
            or len(htf_raw) < MIN_HTF_BARS
        ):

            st.error(
                "Insufficient historical data."
            )

            st.stop()

        # Primary analysis

        primary = calculate_indicators(
            primary_raw
        )

        primary = calculate_smc(
            primary
        )

        primary = add_trend(
            primary
        )

        # Higher timeframe

        higher = calculate_indicators(
            htf_raw
        )

        higher = add_trend(
            higher
        )

        # MTF alignment

        htf_trends = align_htf_trend(

            primary,
            higher,
            timeframe,
            htf_timeframe

        )

        primary["HTF_Trend"] = htf_trends

        primary["Signal"] = [

            generate_signal(
                primary.iloc[i],
                htf_trends.iloc[i]
            )

            for i in range(len(primary))

        ]

    except Exception as error:

        logger.exception(
            "Analysis error"
        )

        st.error(
            f"Analysis error: {error}"
        )

        st.stop()

    if primary.empty:
        st.error("No completed candles available.")
        st.stop()

    latest = primary.iloc[-1]

    last_candle_time = primary.index[-1]

    now = pd.Timestamp.now(tz="UTC")

    freshness = now - last_candle_time

    st.divider()

    st.header(
        "📡 Live Research Dashboard"
    )

    st.caption(
        f"Last Candle: {last_candle_time} | "
        f"Data Age: {freshness}"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Market",
        market_name
    )

    b.metric(
        "Last Close",
        f"{latest['Close']:.5f}"
    )

    c.metric(
        "Primary Trend",
        latest["Trend"]
    )

    d.metric(
        "HTF Trend",
        latest["HTF_Trend"]
    )

    st.subheader(
        "🧠 Signal Engine"
    )

    sig = latest["Signal"]

    if sig == "BUY":

        st.success(
            "Bullish conditions detected — "
            "Research Signal: BUY"
        )

    elif sig == "SELL":

        st.error(
            "Bearish conditions detected — "
            "Research Signal: SELL"
        )

    else:

        st.warning(
            "No valid signal — WAIT"
        )

    st.caption(
        "Signals are educational research outputs "
        "and are not trading recommendations."
    )

    # -----------------------------------------
    # Manual Telegram alert
    # -----------------------------------------

    st.subheader(
        "📨 Telegram Research Notification"
    )

    if st.button(
        "Send Current Research Status to Telegram"
    ):

        message = (

            "NEXUS AI RESEARCH STATUS\n"

            f"Market: {market_name}\n"

            f"Timeframe: {timeframe}\n"

            f"Signal: {sig}\n"

            f"Price: {latest['Close']:.5f}\n"

            "Educational research only."

        )

        success, status = send_telegram_alert(
            message
        )

        if success:
            st.success(status)
        else:
            st.warning(status)

    # -----------------------------------------
    # Advanced Chart
    # -----------------------------------------

    st.divider()

    st.header(
        "📈 Advanced Market Chart"
    )

    fig = make_subplots(

        rows=3,

        cols=1,

        shared_xaxes=True,

        vertical_spacing=0.04,

        row_heights=[
            0.60,
            0.20,
            0.20
        ]

    )

    fig.add_trace(

        go.Candlestick(

            x=primary.index,

            open=primary["Open"],

            high=primary["High"],

            low=primary["Low"],

            close=primary["Close"],

            name="Candles"

        ),

        row=1,
        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=primary.index,

            y=primary["EMA20"],

            name="EMA20"

        ),

        row=1,
        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=primary.index,

            y=primary["EMA50"],

            name="EMA50"

        ),

        row=1,
        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=primary.index,

            y=primary["Swing_High_Plot"],

            mode="markers",

            name="Confirmed Swing High"

        ),

        row=1,
        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=primary.index,

            y=primary["Swing_Low_Plot"],

            mode="markers",

            name="Confirmed Swing Low"

        ),

        row=1,
        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=primary.index,

            y=primary["RSI"],

            name="RSI"

        ),

        row=2,
        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=primary.index,

            y=primary["MACD"],

            name="MACD"

        ),

        row=3,
        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=primary.index,

            y=primary["MACD_Signal"],

            name="MACD Signal"

        ),

        row=3,
        col=1

    )

    fig.add_trace(

        go.Bar(

            x=primary.index,

            y=primary["MACD_Hist"],

            name="MACD Histogram"

        ),

        row=3,
        col=1

    )

    fig.update_layout(

        height=750,

        template="plotly_dark",

        xaxis_rangeslider_visible=False,

        hovermode="x unified"

    )

    fig.update_yaxes(
        range=[0, 100],
        row=2,
        col=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------
    # Backtesting
    # -----------------------------------------

    st.divider()

    st.header(
        "🧪 Historical Backtesting"
    )

    results, equity_curve, stats = run_backtest(

        primary,

        htf_trends,

        daily_loss_limit,

        int(max_losses),

        max_dd_limit,

        spread,

        slippage,

        commission_r

    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Entries",
        stats["Entries"]
    )

    c2.metric(
        "Win Rate",
        f"{stats['Win Rate (%)']:.1f}%"
    )

    c3.metric(
        "Net Result",
        f"{stats['Net Result (R)']:.2f} R"
    )

    pf = stats["Profit Factor"]

    if pd.notna(pf) and np.isfinite(pf):
        pf_text = f"{pf:.2f}"
    else:
        pf_text = "N/A"

    c4.metric(
        "Profit Factor",
        pf_text
    )

    st.info(
        "Simulation assumptions: "
        "TP = 2 ATR / SL = 1.5 ATR. "
        "Nominal Reward/Risk = 1.33."
    )

    st.write(
        "Expectancy:",
        f"{stats['Expectancy (R)']:.3f} R"
    )

    st.write(
        "Maximum Drawdown:",
        f"{stats['Max Drawdown (R)']:.2f} R"
    )

    st.write(
        "Forced Closed Positions:",
        stats["Forced"]
    )

    st.write(
        "Risk Pause Activated:",
        stats["Paused"]
    )

    if not equity_curve.empty:

        st.subheader(
            "Equity Curve"
        )

        st.line_chart(

            equity_curve.set_index(
                "Time"
            )["Equity (R)"]

        )

        st.subheader(
            "Drawdown History"
        )

        st.line_chart(

            equity_curve.set_index(
                "Time"
            )["Drawdown (R)"]

        )

    if not results.empty:

        st.subheader(
            "Trade History"
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "Download Backtest CSV",

            csv_data,

            file_name="nexus_backtest.csv",

            mime="text/csv"

        )

    else:

        st.info(
            "No completed historical trades were found."
        )

    # -----------------------------------------
    # Multi-market Scanner
    # -----------------------------------------

    st.divider()

    st.header(
        "🌐 Multi-Market Research Scanner"
    )

    if st.button(
        "Run Multi-Market Scan",
        use_container_width=True
    ):

        scan_results = []
        scan_errors = []

        with st.spinner(
            "Scanning markets..."
        ):

            for name, ticker in SYMBOLS.items():

                try:

                    p_raw = download_data(
                        ticker,
                        timeframe
                    )

                    h_raw = download_data(
                        ticker,
                        htf_timeframe
                    )

                    if exclude_latest:

                        p_raw = remove_incomplete_bars(
                            p_raw,
                            timeframe
                        )

                        h_raw = remove_incomplete_bars(
                            h_raw,
                            htf_timeframe
                        )

                    if (
                        len(p_raw) < MIN_PRIMARY_BARS
                        or len(h_raw) < MIN_HTF_BARS
                    ):

                        scan_errors.append({

                            "Market": name,

                            "Error": "Insufficient data"

                        })

                        continue

                    p_df = calculate_indicators(
                        p_raw
                    )

                    p_df = calculate_smc(
                        p_df
                    )

                    p_df = add_trend(
                        p_df
                    )

                    h_df = calculate_indicators(
                        h_raw
                    )

                    h_df = add_trend(
                        h_df
                    )

                    h_trend = align_htf_trend(

                        p_df,

                        h_df,

                        timeframe,

                        htf_timeframe

                    )

                    latest_row = p_df.iloc[-1]

                    current_signal = generate_signal(

                        latest_row,

                        h_trend.iloc[-1]

                    )

                    _, _, scan_stats = run_backtest(

                        p_df,

                        h_trend,

                        daily_loss_limit,

                        int(max_losses),

                        max_dd_limit,

                        spread,

                        slippage,

                        commission_r

                    )

                    scan_results.append({

                        "Market": name,

                        "Trend": latest_row["Trend"],

                        "HTF Trend": h_trend.iloc[-1],

                        "Research Signal": current_signal,

                        "Trades": scan_stats["Closed"],

                        "Win Rate (%)": round(
                            scan_stats["Win Rate (%)"],
                            1
                        ),

                        "Net Result (R)": round(
                            scan_stats["Net Result (R)"],
                            2
                        )

                    })

                except Exception as error:

                    logger.exception(
                        "Scan failed: %s",
                        name
                    )

                    scan_errors.append({

                        "Market": name,

                        "Error": str(error)

                    })

        if scan_results:

            st.subheader(
                "Scan Results"
            )

            st.dataframe(

                pd.DataFrame(scan_results),

                use_container_width=True

            )

        if scan_errors:

            st.subheader(
                "Markets With Errors"
            )

            st.dataframe(

                pd.DataFrame(scan_errors),

                use_container_width=True

            )


# =========================================================
# ECONOMIC NEWS
# =========================================================

with tab_news:

    st.header(
        "🌍 Economic News Center"
    )

    st.info(
        "دا برخه د معلوماتو او څېړنې لپاره ده. "
        "دا Live Economic Calendar نه دی."
    )

    st.subheader(
        "Official Economic Sources"
    )

    official_sources = [

        (
            "US Employment / NFP",
            "https://www.bls.gov/news.release/empsit.htm"
        ),

        (
            "US CPI",
            "https://www.bls.gov/cpi/"
        ),

        (
            "Federal Reserve / FOMC",
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        ),

        (
            "European Central Bank",
            "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
        ),

        (
            "Bank of England",
            "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"
        )

    ]

    for title, link in official_sources:

        st.markdown(
            f"- [{title}]({link})"
        )

    st.divider()

    st.subheader(
        "📰 Yahoo Finance News Feed"
    )

    news_market = st.selectbox(

        "Choose News Market",

        list(SYMBOLS.keys())

    )

    if st.button(
        "Refresh News"
    ):

        get_yahoo_news.clear()
        st.rerun()

    news = get_yahoo_news(
        SYMBOLS[news_market]
    )

    if news:

        for item in news:

            st.markdown(
                f"### {item['Title']}"
            )

            st.write(
                f"Source: {item['Publisher']}"
            )

            st.caption(
                f"Published: {item['Published']}"
            )

            st.markdown(
                f"[Read More]({item['Link']})"
            )

            st.divider()

    else:

        st.info(
            "اوس مهال خبرونه ترلاسه نه شول."
        )


# =========================================================
# SYSTEM HEALTH
# =========================================================

with tab_system:

    st.header(
        "⚙️ NEXUS AI System Health"
    )

    st.write(
        "System Mode: Educational Research Only"
    )

    st.write(
        "Market Data: Yahoo Finance"
    )

    st.write(
        "Indicators: EMA / RSI / MACD / ATR"
    )

    st.write(
        "Structure: SMC / BOS / CHoCH"
    )

    st.write(
        "Multi-Timeframe: Enabled"
    )

    st.write(
        "Backtesting: Enabled"
    )

    telegram_status = (

        "Configured"

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

        else "Not Configured"

    )

    st.write(
        f"Telegram: {telegram_status}"
    )

    st.write(
        "Broker Connection: Disabled"
    )

    st.write(
        "Real-Money Execution: Disabled"
    )

    st.write(
        "Automated Trading: Disabled"
    )

    st.divider()

    st.caption(
        "NEXUS AI | Research and Simulation Platform"
    )

    st.caption(
        "Historical results do not guarantee future outcomes."
    )
