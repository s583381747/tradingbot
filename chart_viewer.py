"""
Interactive Chart Viewer — TradingView-style visualization.

Shows:
  - Candlestick chart
  - EMA20 (orange) + EMA50 (blue)
  - Chop Box zones (gray shading with box high/low lines)
  - Pullback touches (yellow dots on EMA)
  - Bounce entries (green/red arrows)
  - Volume bars (bottom pane)
  - Range/ATR ratio (second pane, for chop box calibration)

Usage: python3 chart_viewer.py [--date 2025-01-15] [--days 3]
  Opens in browser. Scroll/zoom like TradingView.
"""

import argparse
import functools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State

DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

print = functools.partial(print, flush=True)


def load_data():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(14).mean()
    slope5 = (df["ema20"] - df["ema20"].shift(5)) / 5 / df["Close"] * 100
    df["slope"] = slope5
    rh = df["High"].rolling(20).max()
    rl = df["Low"].rolling(20).min()
    df["range_atr"] = (rh - rl) / df["atr"]
    df["range_high"] = rh
    df["range_low"] = rl
    df["vol_ma"] = df["Volume"].rolling(20).mean()
    return df


def detect_signals(df):
    """Detect chop zones, pullback touches, and bounce entries."""
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    slope_v = df["slope"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    n = len(df)

    # ═══ Chop Box: tight sideways consolidation ═══
    # Use short lookback (10 bars), strict threshold (range/ATR < 2.5)
    # This captures only genuine tight ranges, not broad trending zones
    rh10 = pd.Series(high).rolling(10).max().values
    rl10 = pd.Series(low).rolling(10).min().values
    range10 = rh10 - rl10
    range_atr_10 = np.where(atr_v > 0, range10 / atr_v, 999)
    df["range_atr_10"] = range_atr_10
    df["in_chop"] = range_atr_10 < 2.5  # tight consolidation only
    df["chop_high"] = rh10
    df["chop_low"] = rl10

    # ═══ Touch: wick must actually reach EMA20 line ═══
    # Tolerance = 0.15 * ATR (very tight — wick must nearly kiss the EMA)
    TOUCH_TOL = 0.15
    touches = np.zeros(n, dtype=bool)
    bounces_long = np.zeros(n, dtype=bool)
    bounces_short = np.zeros(n, dtype=bool)

    # Trend: simple EMA alignment
    trend_arr = np.zeros(n, dtype=int)
    for i in range(50, n):
        if atr_v[i] <= 0 or np.isnan(atr_v[i]):
            continue
        if close[i] > ema[i] and ema[i] > ema_s[i]:
            trend_arr[i] = 1
        elif close[i] < ema[i] and ema[i] < ema_s[i]:
            trend_arr[i] = -1

    for i in range(50, n - 1):
        a = atr_v[i]
        if a <= 0 or np.isnan(a):
            continue
        t = trend_arr[i]
        tol = a * TOUCH_TOL

        if t == 1:
            # Long: wick low dips to within tol of EMA20 (or crosses it slightly)
            if low[i] <= ema[i] + tol and low[i] >= ema[i] - a * 0.5:
                touches[i] = True
                if i + 1 < n and close[i + 1] > high[i]:
                    bounces_long[i] = True
        elif t == -1:
            # Short: wick high reaches up to within tol of EMA20
            if high[i] >= ema[i] - tol and high[i] <= ema[i] + a * 0.5:
                touches[i] = True
                if i + 1 < n and close[i + 1] < low[i]:
                    bounces_short[i] = True

    df["trend"] = trend_arr
    df["touch"] = touches
    df["bounce_long"] = bounces_long
    df["bounce_short"] = bounces_short
    return df


def build_chart(df, start_date, n_days):
    """Build plotly figure for date range."""
    mask = (df.index.date >= start_date.date()) & \
           (df.index.date < (start_date + timedelta(days=n_days)).date())
    sub = df[mask].copy()

    if len(sub) == 0:
        return go.Figure().add_annotation(text="No data for this date range", showarrow=False)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.15, 0.20],
        subplot_titles=["QQQ 1min — Chop Box Viewer", "Volume", "Range/ATR"]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=sub.index, open=sub["Open"], high=sub["High"],
        low=sub["Low"], close=sub["Close"],
        name="QQQ", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # EMA20
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["ema20"], mode="lines",
        line=dict(color="orange", width=2), name="EMA20"
    ), row=1, col=1)

    # EMA50
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["ema50"], mode="lines",
        line=dict(color="dodgerblue", width=1, dash="dash"), name="EMA50"
    ), row=1, col=1)

    # Chop Box zones (shading)
    chop_starts = []
    chop_ends = []
    in_chop = False
    for i, (ts, row) in enumerate(sub.iterrows()):
        if row["in_chop"] and not in_chop:
            chop_starts.append(ts)
            in_chop = True
        elif not row["in_chop"] and in_chop:
            chop_ends.append(ts)
            in_chop = False
    if in_chop:
        chop_ends.append(sub.index[-1])

    for s, e in zip(chop_starts, chop_ends):
        chunk = sub[s:e]
        if len(chunk) < 3:
            continue
        # Use the actual tight range from the chop period
        box_h = chunk["chop_high"].iloc[-1] if "chop_high" in chunk else chunk["High"].max()
        box_l = chunk["chop_low"].iloc[-1] if "chop_low" in chunk else chunk["Low"].min()

        # Box shading
        fig.add_shape(
            type="rect", x0=s, x1=e, y0=box_l, y1=box_h,
            fillcolor="rgba(128,128,128,0.15)", line=dict(color="gray", width=1, dash="dot"),
            row=1, col=1
        )
        # Box high/low lines
        fig.add_trace(go.Scatter(
            x=[s, e], y=[box_h, box_h], mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[s, e], y=[box_l, box_l], mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip"
        ), row=1, col=1)

    # Pullback touches (yellow dots)
    touch_mask = sub["touch"]
    if touch_mask.any():
        fig.add_trace(go.Scatter(
            x=sub.index[touch_mask], y=sub["ema20"][touch_mask],
            mode="markers", marker=dict(color="yellow", size=5, symbol="circle",
                                         line=dict(color="orange", width=1)),
            name="EMA20 Touch"
        ), row=1, col=1)

    # Bounce entries
    bl = sub["bounce_long"]
    if bl.any():
        fig.add_trace(go.Scatter(
            x=sub.index[bl], y=sub["Low"][bl] - sub["atr"][bl] * 0.3,
            mode="markers", marker=dict(color="lime", size=10, symbol="triangle-up"),
            name="Long Bounce"
        ), row=1, col=1)

    bs = sub["bounce_short"]
    if bs.any():
        fig.add_trace(go.Scatter(
            x=sub.index[bs], y=sub["High"][bs] + sub["atr"][bs] * 0.3,
            mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"),
            name="Short Bounce"
        ), row=1, col=1)

    # Volume (colored by direction)
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(sub["Close"], sub["Open"])]
    fig.add_trace(go.Bar(
        x=sub.index, y=sub["Volume"], marker_color=colors,
        name="Volume", showlegend=False
    ), row=2, col=1)

    # Volume MA
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["vol_ma"], mode="lines",
        line=dict(color="orange", width=1), name="Vol MA20", showlegend=False
    ), row=2, col=1)

    # Range/ATR ratio (10-bar tight)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["range_atr_10"], mode="lines",
        line=dict(color="purple", width=1), name="Range/ATR(10)"
    ), row=3, col=1)

    # Also show 20-bar for reference
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["range_atr"], mode="lines",
        line=dict(color="gray", width=1, dash="dot"), name="Range/ATR(20)"
    ), row=3, col=1)

    # Chop threshold line
    fig.add_hline(y=2.5, line_dash="dash", line_color="red",
                  annotation_text="Chop < 2.5", row=3, col=1)

    # Layout
    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=60, r=20, t=40, b=20),
    )

    # Hide gaps (weekends, overnight)
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour"),
        ]
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2025-01-15", help="Start date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=3, help="Number of days to show")
    parser.add_argument("--port", type=int, default=8050, help="Server port")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data()
    print("Detecting signals...")
    df = detect_signals(df)
    print(f"Ready: {len(df)} bars")

    available_dates = sorted(set(df.index.date))
    date_options = [{"label": str(d), "value": str(d)} for d in available_dates]

    app = Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.Label("Date: ", style={"color": "white", "marginRight": "10px"}),
            dcc.Dropdown(
                id="date-picker",
                options=date_options,
                value=args.date,
                style={"width": "200px", "display": "inline-block"}
            ),
            html.Label(" Days: ", style={"color": "white", "marginLeft": "20px", "marginRight": "10px"}),
            dcc.Dropdown(
                id="days-picker",
                options=[{"label": str(d), "value": d} for d in [1, 2, 3, 5, 10]],
                value=args.days,
                style={"width": "80px", "display": "inline-block"}
            ),
            html.Button("Prev", id="prev-btn", n_clicks=0,
                         style={"marginLeft": "20px", "padding": "5px 15px"}),
            html.Button("Next", id="next-btn", n_clicks=0,
                         style={"marginLeft": "5px", "padding": "5px 15px"}),
        ], style={"display": "flex", "alignItems": "center", "padding": "10px",
                   "backgroundColor": "#1e1e1e"}),

        dcc.Graph(id="chart", style={"height": "90vh"}),
    ], style={"backgroundColor": "#1e1e1e"})

    @app.callback(
        [Output("chart", "figure"), Output("date-picker", "value")],
        [Input("date-picker", "value"), Input("days-picker", "value"),
         Input("prev-btn", "n_clicks"), Input("next-btn", "n_clicks")],
        [State("date-picker", "value")]
    )
    def update_chart(selected_date, n_days, prev_clicks, next_clicks, current_date):
        from dash import ctx
        date_str = selected_date or current_date or args.date

        if ctx.triggered_id == "prev-btn":
            dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=n_days)
            date_str = dt.strftime("%Y-%m-%d")
        elif ctx.triggered_id == "next-btn":
            dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=n_days)
            date_str = dt.strftime("%Y-%m-%d")

        start = datetime.strptime(date_str, "%Y-%m-%d")
        fig = build_chart(df, start, n_days)
        return fig, date_str

    print(f"\nStarting server at http://localhost:{args.port}")
    print("Open in browser. Use dropdown to navigate dates, Prev/Next to scroll.")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
