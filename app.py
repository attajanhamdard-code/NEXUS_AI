import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# NEXUS AI
# Educational research and historical paper simulation only.
# No broker connection. No live orders. No financial advice.
# ============================================================

st.set_page_config(
    page_title="NEXUS AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NEXUS AI")
st.caption(
    "Market Research | MTF | SMC Structure | Backtesting | Validation"
)

# ============================================================
# 1. MARKET SETTINGS
# ============================================================

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
# 2. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ System Settings")

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
    "Exclude latest candle",
    True
)

st.sidebar.divider()

st.sidebar.subheader("Simulation Risk Controls")

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

st.sidebar.subheader("Estimated Transaction Costs")

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

st.sidebar.divider()

st.sidebar.subheader("Validation Settings")

train_fraction = st.sidebar.slider(
    "Training Share",
    min_value=0.50,
    max_value=0.85,
    value=0.70,
    step=0.05
)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# 3. DATA DOWNLOAD
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

        if not all(c in data.columns for c in required):
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

        data = data.dropna(
            subset=required
        )

        data = data[
            ~data.index.duplicated(keep="last")
        ]

        return data.sort_index()

    except Exception:
        return pd.DataFrame()


def resample_ohlcv(data, rule):

    if data.empty:
        return data.copy()

    out = data.resample(
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

    return out.dropna(
        subset=["Open", "High", "Low", "Close"]
    )


def download_data(ticker, tf):

    if tf == "15m":

        data = download_raw(
            ticker,
            "15m",
            "60d"
        )

        if data.empty or len(data) < 100:

            data = download_raw(
                ticker,
                "15m",
                "30d"
            )

        return data

    if tf == "1h":

        return download_raw(
            ticker,
            "1h",
            "730d"
        )

    if tf == "4h":

        base = download_raw(
            ticker,
            "1h",
            "730d"
        )

        return resample_ohlcv(
            base,
            "4h"
        )

    if tf == "1d":

        return download_raw(
            ticker,
            "1d",
            "5y"
        )

    if tf == "1wk":

        daily = download_raw(
            ticker,
            "1d",
            "5y"
        )

        return resample_ohlcv(
            daily,
            "W-FRI"
        )

    raise ValueError(
        f"Unsupported timeframe: {tf}"
    )


# ============================================================
# 4. INDICATORS
# ============================================================

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

    df["MACD_Histogram"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # ATR
    prev_close = close.shift(1)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    return df


# ============================================================
# 5. SMC STRUCTURE
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

        hi = df["High"].iloc[
            pivot-left:pivot+right+1
        ]

        lo = df["Low"].iloc[
            pivot-left:pivot+right+1
        ]

        ph = df["High"].iloc[pivot]
        pl = df["Low"].iloc[pivot]

        # Pivot becomes known only now.
        if ph == hi.max():

            last_high = ph
            high_used = False

            # Retrospective plotting only.
            df.iloc[
                pivot,
                df.columns.get_loc("Swing_High_Plot")
            ] = ph

        if pl == lo.min():

            last_low = pl
            low_used = False

            df.iloc[
                pivot,
                df.columns.get_loc("Swing_Low_Plot")
            ] = pl

        close_i = df["Close"].iloc[i]

        event = ""

        bullish_break = (
            last_high is not None
            and not high_used
            and close_i > last_high
        )

        bearish_break = (
            last_low is not None
            and not low_used
            and close_i < last_low
        )

        if bullish_break:

            event = (
                "Bullish CHoCH"
                if trend == "bearish"
                else "Bullish BOS"
            )

            trend = "bullish"
            high_used = True

        elif bearish_break:

            event = (
                "Bearish CHoCH"
                if trend == "bullish"
                else "Bearish BOS"
            )

            trend = "bearish"
            low_used = True

        df.iloc[
            i,
            df.columns.get_loc("Structure_Event")
        ] = event

    return df


# ============================================================
# 6. TREND
# ============================================================

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


# ============================================================
# 7. HIGHER TIMEFRAME ALIGNMENT
# ============================================================

def normalize_timezone(series):

    idx = pd.DatetimeIndex(series)

    if idx.tz is not None:

        idx = idx.tz_convert(
            "UTC"
        ).tz_localize(None)

    return idx


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

    p_duration = TF_DURATION[primary_tf]
    h_duration = TF_DURATION[higher_tf]

    p_index = normalize_timezone(
        primary.index
    )

    h_index = normalize_timezone(
        htf.index
    )

    p = pd.DataFrame({
        "available_at": p_index + p_duration,
        "original_index": range(len(primary))
    })

    h = pd.DataFrame({
        "htf_available_at": h_index + h_duration,
        "htf_trend": htf["Trend"].to_numpy()
    })

    p = p.sort_values(
        "available_at"
    )

    h = h.sort_values(
        "htf_available_at"
    )

    merged = pd.merge_asof(
        p,
        h,
        left_on="available_at",
        right_on="htf_available_at",
        direction="backward",
        allow_exact_matches=True
    )

    merged = merged.sort_values(
        "original_index"
    )

    result = merged[
        "htf_trend"
    ].fillna("UNKNOWN")

    result.index = primary.index

    return result.astype(str)


# ============================================================
# 8. SIGNAL ENGINE
# ============================================================

def generate_signal(row, htf_trend):

    needed = [
        "RSI",
        "ATR",
        "EMA20",
        "EMA50",
        "MACD",
        "MACD_Signal"
    ]

    if any(
        pd.isna(row.get(c))
        for c in needed
    ):

        return "WAIT"

    bull = (
        row["Close"] > row["EMA20"] > row["EMA50"]
        and row["RSI"] > 50
        and row["MACD"] > row["MACD_Signal"]
        and htf_trend == "BULLISH"
        and row["Structure_Event"] in (
            "Bullish BOS",
            "Bullish CHoCH"
        )
    )

    bear = (
        row["Close"] < row["EMA20"] < row["EMA50"]
        and row["RSI"] < 50
        and row["MACD"] < row["MACD_Signal"]
        and htf_trend == "BEARISH"
        and row["Structure_Event"] in (
            "Bearish BOS",
            "Bearish CHoCH"
        )
    )

    if bull:
        return "BUY"

    if bear:
        return "SELL"

    return "WAIT"


# ============================================================
# 9. BACKTEST ENGINE
# ============================================================

def run_backtest(
    df,
    htf_trends,
    daily_limit,
    max_losses,
    max_dd,
    spread_value,
    slippage_value,
    commission,
    start_index=1
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

    start = max(start_index, 1)

    def entry_fill(raw_price, side):

        adjustment = (
            spread_value / 2
            + slippage_value
        )

        if side == "BUY":
            return raw_price + adjustment

        return raw_price - adjustment

    def exit_fill(raw_price, side):

        adjustment = (
            spread_value / 2
            + slippage_value
        )

        if side == "BUY":
            return raw_price - adjustment

        return raw_price + adjustment

    def close_position(timestamp, raw_exit, reason):

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

        peak = max(
            peak,
            equity
        )

        dd = peak - equity

        max_drawdown = max(
            max_drawdown,
            dd
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

            "Exit Reason": reason,

            "Trade Type": (
                "End-of-data mark"
                if reason == "FORCED_CLOSE"
                else "SL/TP"
            )

        })

        curve.append({

            "Time": timestamp,

            "Equity (R)": equity,

            "Drawdown (R)": dd

        })

        position = None

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for i in range(start, len(df)):

        bar = df.iloc[i]

        day_key = str(
            pd.Timestamp(df.index[i]).date()
        )

        # Manage existing position.
        if position is not None:

            side = position["side"]

            exit_price = None
            reason = None

            # Opening gap checks.
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

            # Intrabar checks.
            if exit_price is None:

                if side == "BUY":

                    hit_sl = (
                        bar["Low"] <= position["sl"]
                    )

                    hit_tp = (
                        bar["High"] >= position["tp"]
                    )

                else:

                    hit_sl = (
                        bar["High"] >= position["sl"]
                    )

                    hit_tp = (
                        bar["Low"] <= position["tp"]
                    )

                # Conservative assumption:
                # If both hit in one candle, SL first.
                if hit_sl:

                    exit_price = float(
                        position["sl"]
                    )

                    reason = "SL"

                elif hit_tp:

                    exit_price = float(
                        position["tp"]
                    )

                    reason = "TP"

            if exit_price is not None:

                close_position(
                    df.index[i],
                    exit_price,
                    reason
                )

        # ----------------------------------------------------
        # Risk checks
        # ----------------------------------------------------

        current_dd = peak - equity

        if current_dd >= max_dd:

            paused_global = True

        if consecutive_losses >= max_losses:

            paused_global = True

        daily_loss_reached = (
            daily_pnl.get(day_key, 0.0)
            <= -daily_limit
        )

        if paused_global or daily_loss_reached:

            continue

        # Need next candle for entry.
        if i >= len(df) - 1:

            continue

        if position is not None:

            continue

        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        signal_row = df.iloc[i]

        signal = generate_signal(
            signal_row,
            htf_trends.iloc[i]
        )

        if signal not in ("BUY", "SELL"):

            continue

        atr = signal_row["ATR"]

        if (
            pd.isna(atr)
            or not np.isfinite(atr)
            or atr <= 0
        ):

            continue

        next_bar = df.iloc[i + 1]

        raw_entry = float(
            next_bar["Open"]
        )

        entry = entry_fill(
            raw_entry,
            signal
        )

        # Risk model.
        risk_price = 1.5 * float(atr)

        if risk_price <= 0:

            continue

        reward_price = 2.0 * float(atr)

        if signal == "BUY":

            sl = entry - risk_price
            tp = entry + reward_price

        else:

            sl = entry + risk_price
            tp = entry - reward_price

        position = {

            "side": signal,

            "entry_time": df.index[i + 1],

            "entry": entry,

            "sl": sl,

            "tp": tp,

            "risk_price": risk_price

        }

        entries += 1

    # --------------------------------------------------------
    # End of available history
    # --------------------------------------------------------

    if position is not None:

        close_position(
            df.index[-1],
            float(df.iloc[-1]["Close"]),
            "FORCED_CLOSE"
        )

    result_df = pd.DataFrame(trades)

    curve_df = pd.DataFrame(curve)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    if result_df.empty:

        summary = {

            "Entries": entries,

            "Closed": 0,

            "TP/SL": 0,

            "Forced": forced,

            "Win Rate (%)": 0.0,

            "Expectancy (R)": 0.0,

            "Average Win (R)": 0.0,

            "Average Loss (R)": 0.0,

            "Profit Factor": np.nan,

            "Net Result (R)": 0.0,

            "Max Drawdown (R)": max_drawdown,

            "Paused": paused_global,

            "Daily Results": daily_pnl

        }

        return result_df, curve_df, summary

    wins = result_df[
        result_df["Result (R)"] > 0
    ]

    losses = result_df[
        result_df["Result (R)"] < 0
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

        "Closed": len(result_df),

        "TP/SL": int(
            (result_df["Trade Type"] == "SL/TP").sum()
        ),

        "Forced": forced,

        "Win Rate (%)": (
            len(wins) / len(result_df) * 100
        ),

        "Expectancy (R)": (
            result_df["Result (R)"].mean()
        ),

        "Average Win (R)": (
            wins["Result (R)"].mean()
            if not wins.empty else 0.0
        ),

        "Average Loss (R)": (
            abs(losses["Result (R)"].mean())
            if not losses.empty else 0.0
        ),

        "Profit Factor": profit_factor,

        "Net Result (R)": (
            result_df["Result (R)"].sum()
        ),

        "Max Drawdown (R)": max_drawdown,

        "Paused": paused_global,

        "Daily Results": daily_pnl

    }

    return result_df, curve_df, summary


# ============================================================
# 10. LOAD DATA
# ============================================================

try:

    with st.spinner("Downloading market history..."):

        primary_raw = download_data(
            symbol,
            timeframe
        )

        htf_raw = download_data(
            symbol,
            htf_timeframe
        )

    if primary_raw.empty or htf_raw.empty:

        st.error(
            "Yahoo Finance returned no data. Try another market or timeframe."
        )

        st.stop()

    if exclude_latest:

        if len(primary_raw) > 1:

            primary_raw = primary_raw.iloc[:-1].copy()

        if len(htf_raw) > 1:

            htf_raw = htf_raw.iloc[:-1].copy()

    if len(primary_raw) < 100:

        st.error(
            f"Primary data is insufficient: {len(primary_raw)} bars."
        )

        st.stop()

    if len(htf_raw) < 60:

        st.error(
            f"Higher timeframe data is insufficient: {len(htf_raw)} bars."
        )

        st.stop()

    # Primary indicators and structure.
    primary = calculate_indicators(
        primary_raw
    )

    primary = calculate_smc(
        primary
    )

    primary = add_trend(
        primary
    )

    # Higher timeframe indicators.
    higher = calculate_indicators(
        htf_raw
    )

    higher = add_trend(
        higher
    )

    # Historical MTF alignment.
    htf_trend_series = align_htf_trend(
        primary,
        higher,
        timeframe,
        htf_timeframe
    )

    primary["HTF_Trend"] = (
        htf_trend_series
    )

    primary["Signal"] = [

        generate_signal(
            primary.iloc[i],
            htf_trend_series.iloc[i]
        )

        for i in range(len(primary))

    ]

except Exception as exc:

    st.error(
        f"Data or calculation error: {exc}"
    )

    st.stop()


# ============================================================
# 11. LIVE DASHBOARD
# ============================================================

latest = primary.iloc[-1]

st.divider()

st.header("📡 Live Research Dashboard")

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

# Signal.
st.subheader("🧠 Signal Engine")

signal = latest["Signal"]

if signal == "BUY":

    st.success(
        "Bullish conditions detected — Historical research signal: BUY"
    )

elif signal == "SELL":

    st.error(
        "Bearish conditions detected — Historical research signal: SELL"
    )

else:

    st.warning(
        "No valid signal — WAIT"
    )

st.caption(
    "Research and paper simulation only. Not a live trading instruction."
)


# ============================================================
# 12. INDICATOR CHECKLIST
# ============================================================

st.subheader("📋 Indicator Checklist")

checks = {

    "Close above EMA20": (
        latest["Close"] > latest["EMA20"]
        if pd.notna(latest["EMA20"]) else False
    ),

    "EMA20 above EMA50": (
        latest["EMA20"] > latest["EMA50"]
        if pd.notna(latest["EMA50"]) else False
    ),

    "RSI above 50": (
        latest["RSI"] > 50
        if pd.notna(latest["RSI"]) else False
    ),

    "MACD above signal": (
        latest["MACD"] > latest["MACD_Signal"]
        if pd.notna(latest["MACD_Signal"]) else False
    ),

    "Higher timeframe bullish": (
        latest["HTF_Trend"] == "BULLISH"
    ),

    "Bullish structure event": (
        latest["Structure_Event"] in (
            "Bullish BOS",
            "Bullish CHoCH"
        )
    )

}

for label, passed in checks.items():

    st.write(
        f"{'✅' if passed else '❌'} {label}"
    )


# ============================================================
# 13. ADVANCED CHART
# ============================================================

st.divider()

st.header("📈 Advanced Market Chart")

chart = primary.tail(250)

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

        x=chart.index,

        open=chart["Open"],

        high=chart["High"],

        low=chart["Low"],

        close=chart["Close"],

        name="Candles"

    ),

    row=1,

    col=1

)

fig.add_trace(

    go.Scatter(

        x=chart.index,

        y=chart["EMA20"],

        name="EMA20"

    ),

    row=1,

    col=1

)

fig.add_trace(

    go.Scatter(

        x=chart.index,

        y=chart["EMA50"],

        name="EMA50"

    ),

    row=1,

    col=1

)

# Swing highs.
sh = chart.dropna(
    subset=["Swing_High_Plot"]
)

fig.add_trace(

    go.Scatter(

        x=sh.index,

        y=sh["Swing_High_Plot"],

        mode="markers",

        marker_symbol="triangle-down",

        marker_size=9,

        name="Swing High (retrospective)"

    ),

    row=1,

    col=1

)

# Swing lows.
sl = chart.dropna(
    subset=["Swing_Low_Plot"]
)

fig.add_trace(

    go.Scatter(

        x=sl.index,

        y=sl["Swing_Low_Plot"],

        mode="markers",

        marker_symbol="triangle-up",

        marker_size=9,

        name="Swing Low (retrospective)"

    ),

    row=1,

    col=1

)

# RSI.
fig.add_trace(

    go.Scatter(

        x=chart.index,

        y=chart["RSI"],

        name="RSI"

    ),

    row=2,

    col=1

)

fig.add_hline(
    y=50,
    line_dash="dot",
    row=2,
    col=1
)

# MACD.
fig.add_trace(

    go.Scatter(

        x=chart.index,

        y=chart["MACD"],

        name="MACD"

    ),

    row=3,

    col=1

)

fig.add_trace(

    go.Scatter(

        x=chart.index,

        y=chart["MACD_Signal"],

        name="MACD Signal"

    ),

    row=3,

    col=1

)

fig.update_layout(

    height=850,

    template="plotly_dark",

    xaxis_rangeslider_visible=False

)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 14. STATISTICS DISPLAY
# ============================================================

def show_stats(stats, prefix=""):

    x1, x2, x3, x4 = st.columns(4)

    x1.metric(
        f"{prefix} Entries",
        stats["Entries"]
    )

    x2.metric(
        f"{prefix} Closed Trades",
        stats["Closed"]
    )

    x3.metric(
        f"{prefix} Win Rate",
        f"{stats['Win Rate (%)']:.1f}%"
    )

    x4.metric(
        f"{prefix} Net Result",
        f"{stats['Net Result (R)']:.2f} R"
    )

    y1, y2, y3, y4 = st.columns(4)

    y1.metric(
        f"{prefix} Expectancy",
        f"{stats['Expectancy (R)']:.3f} R"
    )

    pf = stats["Profit Factor"]

    if np.isinf(pf):

        pf_text = "∞"

    elif pd.isna(pf):

        pf_text = "N/A"

    else:

        pf_text = f"{pf:.2f}"

    y2.metric(
        f"{prefix} Profit Factor",
        pf_text
    )

    y3.metric(
        f"{prefix} Max Drawdown",
        f"{stats['Max Drawdown (R)']:.2f} R"
    )

    y4.metric(
        f"{prefix} Forced Closes",
        stats["Forced"]
    )


# ============================================================
# 15. FULL HISTORICAL BACKTEST
# ============================================================

st.divider()

st.header("🧪 Historical Backtest")

full_results, full_curve, full_stats = run_backtest(

    primary,

    htf_trend_series,

    daily_loss_limit,

    int(max_losses),

    max_dd_limit,

    spread,

    slippage,

    commission_r

)

show_stats(
    full_stats
)

# Equity chart.
if not full_curve.empty:

    eq = go.Figure()

    eq.add_trace(

        go.Scatter(

            x=full_curve["Time"],

            y=full_curve["Equity (R)"],

            mode="lines",

            name="Equity"

        )

    )

    eq.add_trace(

        go.Scatter(

            x=full_curve["Time"],

            y=-full_curve["Drawdown (R)"],

            mode="lines",

            name="Negative Drawdown"

        )

    )

    eq.update_layout(

        template="plotly_dark",

        height=400,

        xaxis_title="Date",

        yaxis_title="R"

    )

    st.plotly_chart(
        eq,
        use_container_width=True
    )


# Trade history.
if not full_results.empty:

    st.subheader("📑 Trade History")

    st.dataframe(
        full_results,
        use_container_width=True
    )

    st.download_button(

        "⬇️ Download Trade History CSV",

        full_results.to_csv(
            index=False
        ).encode("utf-8"),

        "nexus_ai_trade_history.csv",

        "text/csv"

    )

else:

    st.info(
        "No simulated trades were produced by the current rules and data."
    )


if full_stats["Paused"]:

    st.warning(
        "A global simulation risk limit was reached. Later entries were blocked."
    )

else:

    st.success(
        "No global simulation stop condition was reached."
    )


# ============================================================
# 16. CHRONOLOGICAL TRAIN / TEST
# ============================================================

st.divider()

st.header(
    "🧪 Engine 6 — Chronological Train/Test"
)

st.write(
    "د معلوماتو لومړۍ برخه Train او وروستۍ برخه Test ده. "
    "دواړه برخې جلا ازمویل کېږي."
)

split = int(
    len(primary) * train_fraction
)

if split < 100 or len(primary) - split < 100:

    st.warning(
        "د Train/Test لپاره معلومات کم دي. اوږده تاریخي ډیټا وکاروه."
    )

else:

    train_df = primary.iloc[
        :split
    ].copy()

    train_htf = htf_trend_series.iloc[
        :split
    ].copy()

    test_df = primary.iloc[
        split:
    ].copy()

    test_htf = htf_trend_series.iloc[
        split:
    ].copy()

    # Each segment starts flat.
    # Indicators were calculated chronologically on historical data.
    # No open position is carried across the split.

    train_res, train_eq, train_stats = run_backtest(

        train_df,

        train_htf,

        daily_loss_limit,

        int(max_losses),

        max_dd_limit,

        spread,

        slippage,

        commission_r

    )

    test_res, test_eq, test_stats = run_backtest(

        test_df,

        test_htf,

        daily_loss_limit,

        int(max_losses),

        max_dd_limit,

        spread,

        slippage,

        commission_r

    )

    st.subheader("📘 Training Results")

    show_stats(
        train_stats,
        "Train"
    )

    st.subheader("📕 Out-of-Sample Test Results")

    show_stats(
        test_stats,
        "Test"
    )

    # Comparison.
    st.subheader("⚖️ Train vs Test")

    compare = pd.DataFrame([

        {
            "Metric": "Closed Trades",

            "Train": train_stats["Closed"],

            "Test": test_stats["Closed"]
        },

        {
            "Metric": "Win Rate (%)",

            "Train": train_stats["Win Rate (%)"],

            "Test": test_stats["Win Rate (%)"]
        },

        {
            "Metric": "Expectancy (R)",

            "Train": train_stats["Expectancy (R)"],

            "Test": test_stats["Expectancy (R)"]
        },

        {
            "Metric": "Profit Factor",

            "Train": train_stats["Profit Factor"],

            "Test": test_stats["Profit Factor"]
        },

        {
            "Metric": "Max Drawdown (R)",

            "Train": train_stats["Max Drawdown (R)"],

            "Test": test_stats["Max Drawdown (R)"]
        }

    ])

    st.dataframe(
        compare,
        use_container_width=True
    )

    # Overfitting warning.
    if (
        train_stats["Closed"] < 30
        or test_stats["Closed"] < 30
    ):

        st.info(
            "معاملې کمې دي. د Overfitting په اړه باوري نتیجه نشو اخیستلای."
        )

    elif (
        train_stats["Expectancy (R)"] > 0
        and test_stats["Expectancy (R)"] <= 0
    ):

        st.warning(
            "Train مثبت دی خو Test مثبت نه دی. "
            "ممکن Overfitting یا د بازار د شرایطو بدلون وي."
        )

    elif (
        train_stats["Expectancy (R)"] <= 0
        and test_stats["Expectancy (R)"] <= 0
    ):

        st.warning(
            "په دواړو برخو کې Expectancy مثبت نه دی."
        )

    elif (
        train_stats["Expectancy (R)"] > 0
        and test_stats["Expectancy (R)"] > 0
    ):

        st.success(
            "دواړو برخو کې Expectancy مثبت دی. "
            "خو دا د راتلونکي فعالیت ضمانت نه دی."
        )

    # Separate equity curves.
    eqfig = go.Figure()

    if not train_eq.empty:

        eqfig.add_trace(

            go.Scatter(

                x=train_eq["Time"],

                y=train_eq["Equity (R)"],

                mode="lines",

                name="Train Equity"

            )

        )

    if not test_eq.empty:

        eqfig.add_trace(

            go.Scatter(

                x=test_eq["Time"],

                y=test_eq["Equity (R)"],

                mode="lines",

                name="Test Equity"

            )

        )

    eqfig.update_layout(

        template="plotly_dark",

        height=400,

        xaxis_title="Date",

        yaxis_title="Cumulative R"

    )

    st.plotly_chart(
        eqfig,
        use_container_width=True
    )


# ============================================================
# 17. FINAL INFORMATION
# ============================================================

st.divider()

st.caption(
    "NEXUS AI is an educational research and historical simulation tool. "
    "Historical results do not guarantee future performance. "
    "Yahoo Finance data may be delayed, incomplete, or approximate. "
    "Spread, slippage, commissions, and market session boundaries "
    "may differ from actual conditions. "
    "OHLC candles cannot reveal the exact intrabar price sequence. "
    "No live trading or broker connection is provided."
)# ============================================================
# 18. NEXUS AI — MULTI-MARKET BACKTEST SCANNER
# Educational historical simulation only.
# ============================================================

st.divider()

st.header("🌍 Multi-Market Backtest Scanner")

st.write(
    "دا برخه ټول ټاکل شوي بازارونه په جلا توګه ازموي "
    "او د تاریخي پایلو پرتله وړاندې کوي."
)

# Separate estimated costs for each market.
st.subheader("⚙️ Market-specific estimated costs")

market_costs = {}

with st.expander("Configure individual market costs"):

    for m_name, m_symbol in SYMBOLS.items():

        st.markdown(f"**{m_name}**")

        c1, c2 = st.columns(2)

        market_spread = c1.number_input(
            f"{m_name} full spread",
            min_value=0.0,
            value=0.0,
            step=0.0001,
            format="%.6f",
            key=f"spread_{m_symbol}"
        )

        market_slippage = c2.number_input(
            f"{m_name} slippage per side",
            min_value=0.0,
            value=0.0,
            step=0.0001,
            format="%.6f",
            key=f"slippage_{m_symbol}"
        )

        market_costs[m_name] = {
            "spread": market_spread,
            "slippage": market_slippage
        }

st.caption(
    "یادونه: د هر بازار لګښتونه باید د مناسبو تاریخي یا "
    "معتبره اټکلي معلوماتو له مخې تنظیم شي. صفر لګښتونه "
    "د واقعي معاملاتي شرایطو استازیتوب نه کوي."
)

# Scan button.
if st.button(
    "🚀 Run Multi-Market Historical Scan",
    use_container_width=True
):

    multi_results = []
    scan_errors = []

    progress = st.progress(0)

    status_text = st.empty()

    total_markets = len(SYMBOLS)

    with st.spinner("Scanning historical market data..."):

        for number, (m_name, m_symbol) in enumerate(
            SYMBOLS.items(),
            start=1
        ):

            status_text.write(
                f"Testing {m_name} ({number}/{total_markets})..."
            )

            try:

                # ----------------------------------------
                # 1. Download data
                # ----------------------------------------

                m_primary_raw = download_data(
                    m_symbol,
                    timeframe
                )

                m_htf_raw = download_data(
                    m_symbol,
                    htf_timeframe
                )

                if (
                    m_primary_raw.empty
                    or m_htf_raw.empty
                ):

                    raise ValueError(
                        "No historical data returned."
                    )

                if exclude_latest:

                    if len(m_primary_raw) > 1:

                        m_primary_raw = (
                            m_primary_raw.iloc[:-1].copy()
                        )

                    if len(m_htf_raw) > 1:

                        m_htf_raw = (
                            m_htf_raw.iloc[:-1].copy()
                        )

                if len(m_primary_raw) < 100:

                    raise ValueError(
                        "Insufficient primary timeframe data."
                    )

                if len(m_htf_raw) < 60:

                    raise ValueError(
                        "Insufficient higher timeframe data."
                    )

                # ----------------------------------------
                # 2. Calculate primary indicators
                # ----------------------------------------

                m_primary = calculate_indicators(
                    m_primary_raw
                )

                m_primary = calculate_smc(
                    m_primary
                )

                m_primary = add_trend(
                    m_primary
                )

                # ----------------------------------------
                # 3. Calculate HTF indicators
                # ----------------------------------------

                m_higher = calculate_indicators(
                    m_htf_raw
                )

                m_higher = add_trend(
                    m_higher
                )

                # ----------------------------------------
                # 4. Historical MTF alignment
                # ----------------------------------------

                m_htf_trends = align_htf_trend(
                    m_primary,
                    m_higher,
                    timeframe,
                    htf_timeframe
                )

                m_primary["HTF_Trend"] = (
                    m_htf_trends
                )

                # ----------------------------------------
                # 5. Run historical simulation
                # ----------------------------------------

                m_spread = market_costs[m_name]["spread"]

                m_slippage = market_costs[m_name]["slippage"]

                (
                    m_trades,
                    m_curve,
                    m_stats
                ) = run_backtest(

                    m_primary,

                    m_htf_trends,

                    daily_loss_limit,

                    int(max_losses),

                    max_dd_limit,

                    m_spread,

                    m_slippage,

                    commission_r

                )

                # ----------------------------------------
                # 6. Prepare metrics
                # ----------------------------------------

                pf = m_stats["Profit Factor"]

                if pd.isna(pf):

                    pf_display = "N/A"

                elif np.isinf(pf):

                    pf_display = "Infinity"

                else:

                    pf_display = round(
                        float(pf),
                        2
                    )

                trades_count = m_stats["Closed"]

                expectancy = m_stats["Expectancy (R)"]

                # Minimum sample warning.
                if trades_count < 30:

                    reliability = "LOW SAMPLE"

                elif expectancy > 0:

                    reliability = "POSITIVE HISTORICAL"

                else:

                    reliability = "NEGATIVE HISTORICAL"

                # ----------------------------------------
                # 7. Store results
                # ----------------------------------------

                multi_results.append({

                    "Market": m_name,

                    "Primary TF": timeframe,

                    "HTF": htf_timeframe,

                    "Closed Trades": trades_count,

                    "Win Rate (%)": round(
                        m_stats["Win Rate (%)"],
                        2
                    ),

                    "Expectancy (R)": round(
                        expectancy,
                        3
                    ),

                    "Net Result (R)": round(
                        m_stats["Net Result (R)"],
                        2
                    ),

                    "Profit Factor": pf_display,

                    "Max Drawdown (R)": round(
                        m_stats["Max Drawdown (R)"],
                        2
                    ),

                    "Forced Closes": m_stats["Forced"],

                    "Risk Paused": m_stats["Paused"],

                    "Sample Status": reliability

                })

            except Exception as exc:

                scan_errors.append({

                    "Market": m_name,

                    "Error": str(exc)

                })

            progress.progress(
                number / total_markets
            )

    status_text.empty()

    progress.empty()

    # ========================================================
    # 19. DISPLAY RESULTS
    # ========================================================

    if multi_results:

        results_df = pd.DataFrame(
            multi_results
        )

        st.success(
            "Multi-Market Historical Scan Completed."
        )

        st.subheader("📊 Market Comparison")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # ----------------------------------------
        # Export CSV
        # ----------------------------------------

        st.download_button(

            "⬇️ Download Multi-Market Results",

            results_df.to_csv(
                index=False
            ).encode("utf-8"),

            "nexus_multi_market_results.csv",

            "text/csv"

        )

        # ----------------------------------------
        # Expectancy comparison
        # ----------------------------------------

        st.subheader("📈 Expectancy Comparison")

        chart_df = results_df.copy()

        chart_df["Expectancy (R)"] = pd.to_numeric(
            chart_df["Expectancy (R)"],
            errors="coerce"
        )

        chart = go.Figure()

        chart.add_trace(

            go.Bar(

                x=chart_df["Market"],

                y=chart_df["Expectancy (R)"],

                name="Expectancy"

            )

        )

        chart.add_hline(
            y=0,
            line_dash="dash"
        )

        chart.update_layout(

            template="plotly_dark",

            height=450,

            title="Historical Expectancy by Market",

            xaxis_title="Market",

            yaxis_title="Expectancy (R)"

        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        # ----------------------------------------
        # Sample size warning
        # ----------------------------------------

        low_sample = results_df[
            results_df["Closed Trades"] < 30
        ]

        if not low_sample.empty:

            st.warning(
                "ځینو بازارونو کې د معاملاتو شمېر کم دی. "
                "د دغو بازارونو پایلې لا باوري نه دي."
            )

        # ----------------------------------------
        # Research interpretation
        # ----------------------------------------

        st.subheader("🧠 Research Summary")

        valid = results_df[
            results_df["Closed Trades"] >= 30
        ].copy()

        if valid.empty:

            st.info(
                "تر اوسه په هېڅ بازار کې د ۳۰ یا زیاتو "
                "بشپړو معاملاتو نمونه نشته."
            )

        else:

            valid["Expectancy (R)"] = pd.to_numeric(
                valid["Expectancy (R)"],
                errors="coerce"
            )

            positive = valid[
                valid["Expectancy (R)"] > 0
            ]

            if not positive.empty:

                st.write(
                    "په لاندې بازارونو کې تاریخي Expectancy مثبت دی:"
                )

                st.dataframe(
                    positive[
                        [
                            "Market",
                            "Closed Trades",
                            "Expectancy (R)",
                            "Profit Factor"
                        ]
                    ],
                    use_container_width=True
                )

            else:

                st.warning(
                    "په هغو بازارونو کې چې کافي معاملې لري، "
                    "مثبت Expectancy ونه موندل شو."
                )

    else:

        st.error(
            "No market produced a valid backtest result."
        )

    # ========================================================
    # 20. SCAN ERRORS
    # ========================================================

    if scan_errors:

        st.subheader("⚠️ Market Data Errors")

        st.dataframe(
            pd.DataFrame(scan_errors),
            use_container_width=True
        )

# ============================================================
# END MULTI-MARKET SCANNER
# ============================================================