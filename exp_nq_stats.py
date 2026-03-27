"""
Full stats: Max trailing DD in R, Sharpe, APR, Sortino, Calmar — NQ real data.
Both V8 and gate=0.0, with $150 risk dynamic sizing.
"""
from __future__ import annotations
import functools, datetime as dt, math
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"
MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00
STARTING_EQUITY = 50_000

V8 = {
    "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
}
V8G0 = {**V8, "gate_tighten": 0.0}


def resample(df, m):
    if m <= 1: return df
    return df.resample(f"{m}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def add_ind(df, s):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=s["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(s["atr_period"]).mean()
    return df


def run_full(df_1min, s, risk_per_trade=150):
    df = resample(df_1min, s["tf_minutes"])
    df = add_ind(df, s)
    H = df["High"].values; L = df["Low"].values; C = df["Close"].values
    ef = df["ema_f"].values; es = df["ema_s"].values; atr = df["atr"].values
    T = df.index.time; D = df.index.date; n = len(df)
    tf = max(1, s["tf_minutes"])
    mh = max(20, s["max_hold_bars"] // tf)
    cb = max(5, s["chand_bars"] // tf)
    gb = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0

    equity = STARTING_EQUITY; peak_eq = equity
    trades = []; daily_pnl = {}
    bar = max(s["ema_slow"], s["atr_period"]) + 5
    dlr = 0.0; cd = None; sk = 0

    while bar < n - mh - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ef[bar]) or np.isnan(es[bar]):
            bar += 1; continue
        if T[bar] >= s["no_entry_after"]: bar += 1; continue
        d = D[bar]
        if cd != d: cd = d; dlr = 0.0
        if dlr >= s["daily_loss_r"]: bar += 1; continue
        c = C[bar]
        if c > ef[bar] and ef[bar] > es[bar]: tr = 1
        elif c < ef[bar] and ef[bar] < es[bar]: tr = -1
        else: bar += 1; continue
        tol = a * s["touch_tol"]
        if tr == 1: touch = L[bar] <= ef[bar] + tol and L[bar] >= ef[bar] - a * s["touch_below_max"]
        else: touch = H[bar] >= ef[bar] - tol and H[bar] <= ef[bar] + a * s["touch_below_max"]
        if not touch: bar += 1; continue
        if sk > 0: sk -= 1; bar += 1; continue

        entry = C[bar]
        stop = L[bar] - s["stop_buffer"] * a if tr == 1 else H[bar] + s["stop_buffer"] * a
        rp = abs(entry - stop)
        if rp <= 0: bar += 1; continue

        risk_per_c = rp * MNQ_PER_POINT
        nc = max(1, min(10, int(risk_per_trade / risk_per_c)))
        rm = rp * MNQ_PER_POINT * nc
        ec = COMM_RT * nc / 2 + SPREAD

        eb = bar; rs = stop; bt = False; mfe = 0.0; r = 0.0; endb = bar; ex = "timeout"
        for k in range(1, mh + 1):
            bi = eb + k
            if bi >= n: break
            h = H[bi]; l = L[bi]; ca = atr[bi] if not np.isnan(atr[bi]) else a
            if tr == 1: mfe = max(mfe, (h - entry) / rp)
            else: mfe = max(mfe, (entry - l) / rp)
            if T[bi] >= s["force_close_at"]:
                r = (C[bi] - entry) / rp * tr; endb = bi; ex = "close"; break
            if gb > 0 and k == gb and not bt:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * rp * tr
                    if tr == 1: rs = max(rs, ns)
                    else: rs = min(rs, ns)
            st = (tr == 1 and l <= rs) or (tr == -1 and h >= rs)
            if st:
                r = (rs - entry) / rp * tr; endb = bi
                if bt:
                    ref = entry + s["be_stop_r"] * rp * tr
                    ex = "be" if abs(rs - ref) < 0.05 * rp else "trail"
                else: ex = "stop"
                break
            if not bt and s["be_trigger_r"] > 0:
                tp = entry + s["be_trigger_r"] * rp * tr
                if (tr == 1 and h >= tp) or (tr == -1 and l <= tp):
                    bt = True; bl = entry + s["be_stop_r"] * rp * tr
                    if tr == 1: rs = max(rs, bl)
                    else: rs = min(rs, bl)
            if bt and k >= cb:
                skk = max(1, k - cb + 1)
                hv = [H[eb + kk] for kk in range(skk, k) if eb + kk < n]
                lv = [L[eb + kk] for kk in range(skk, k) if eb + kk < n]
                if hv and lv:
                    if tr == 1: rs = max(rs, max(hv) - s["chand_mult"] * ca)
                    else: rs = min(rs, min(lv) + s["chand_mult"] * ca)
        else:
            r = (C[min(eb + mh, n-1)] - entry) / rp * tr; endb = min(eb + mh, n-1)

        raw = r * rm
        xc = COMM_RT * nc / 2
        xs = STOP_SLIP if ex in ("stop", "trail") else 0
        bs = BE_SLIP if ex == "be" else 0
        net = raw - (ec + xc + xs + bs)

        equity += net; peak_eq = max(peak_eq, equity)
        d_str = str(d)
        daily_pnl[d_str] = daily_pnl.get(d_str, 0) + net

        trades.append({"pnl": net, "r": r, "ex": ex, "nc": nc,
                        "risk_total": rm, "date": d_str, "equity": equity})
        if r < 0: dlr += abs(r)
        if r > 0: sk = s.get("skip_after_win", 0)
        bar = endb + 1

    return pd.DataFrame(trades), daily_pnl


def compute_stats(tdf, daily_pnl, label=""):
    if len(tdf) == 0:
        print(f"  [{label}] No trades"); return

    # Basic
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    pf = gw / gl if gl > 0 else 0
    total_pnl = tdf["pnl"].sum()
    n_trades = len(tdf)
    wr = (tdf["r"] > 0).mean() * 100

    # Trailing DD in $ and R
    cum_pnl = tdf["pnl"].cumsum()
    dd_dollar = (cum_pnl.cummax() - cum_pnl).max()

    # R-based trailing DD: track cumulative R
    cum_r = tdf["r"].cumsum()
    dd_r = (cum_r.cummax() - cum_r).max()

    # Max trailing DD from equity peak
    eq = tdf["equity"].values
    peak = np.maximum.accumulate(eq)
    trail_dd_eq = (peak - eq).max()

    # Daily returns for Sharpe/Sortino
    daily_df = pd.Series(daily_pnl)
    daily_returns = daily_df.values
    n_days = len(daily_returns)

    # Trading days per year
    dates = pd.to_datetime(list(daily_pnl.keys()))
    years = (dates.max() - dates.min()).days / 365.25
    trading_days_per_year = n_days / years if years > 0 else 252

    # APR (Annual Percentage Return)
    apr = total_pnl / STARTING_EQUITY / years * 100 if years > 0 else 0

    # Sharpe (annualized, daily returns)
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe = (daily_mean / daily_std) * math.sqrt(trading_days_per_year) if daily_std > 0 else 0

    # Sortino (only downside deviation)
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.001
    sortino = (daily_mean / downside_std) * math.sqrt(trading_days_per_year) if downside_std > 0 else 0

    # Calmar (APR / Max DD%)
    max_dd_pct = trail_dd_eq / STARTING_EQUITY * 100
    calmar = (apr / max_dd_pct) if max_dd_pct > 0 else 0

    # R stats
    avg_r = tdf["r"].mean()
    avg_win_r = tdf.loc[tdf["r"] > 0, "r"].mean() if (tdf["r"] > 0).any() else 0
    avg_loss_r = tdf.loc[tdf["r"] <= 0, "r"].mean() if (tdf["r"] <= 0).any() else 0
    total_r = tdf["r"].sum()
    b5 = (tdf["r"] >= 5).sum()

    # Equity curve stats
    losing_days = (daily_df < 0).sum()
    winning_days = (daily_df > 0).sum()
    best_day = daily_df.max()
    worst_day = daily_df.min()

    # Average contracts
    avg_nc = tdf["nc"].mean()
    avg_risk = tdf["risk_total"].mean()

    # Max consecutive losses
    is_loss = (tdf["r"] <= 0).values
    max_consec_loss = 0; cur = 0
    for x in is_loss:
        if x: cur += 1; max_consec_loss = max(max_consec_loss, cur)
        else: cur = 0

    # DD distribution in R
    # Rolling peak R and DD from it
    dd_r_series = cum_r.cummax() - cum_r
    dd_r_p90 = dd_r_series.quantile(0.90)
    dd_r_p95 = dd_r_series.quantile(0.95)
    dd_r_p99 = dd_r_series.quantile(0.99)

    print(f"\n  ╔══ {label} ═══════════════════════════════════════════════╗")
    print(f"  ║")
    print(f"  ║  Return & Risk")
    print(f"  ║  ──────────────────────────────────────────────────")
    print(f"  ║  Net PF:            {pf:.3f}")
    print(f"  ║  Total PnL:         ${total_pnl:+,.0f}")
    print(f"  ║  APR:               {apr:+.1f}%")
    print(f"  ║  Sharpe:            {sharpe:.2f}")
    print(f"  ║  Sortino:           {sortino:.2f}")
    print(f"  ║  Calmar:            {calmar:.2f}")
    print(f"  ║")
    print(f"  ║  Drawdown")
    print(f"  ║  ──────────────────────────────────────────────────")
    print(f"  ║  Max Trailing DD $:  ${trail_dd_eq:,.0f}")
    print(f"  ║  Max Trailing DD R:  {dd_r:.2f}R")
    print(f"  ║  DD R P90:           {dd_r_p90:.2f}R")
    print(f"  ║  DD R P95:           {dd_r_p95:.2f}R")
    print(f"  ║  DD R P99:           {dd_r_p99:.2f}R")
    print(f"  ║  Max DD % of equity: {max_dd_pct:.2f}%")
    print(f"  ║")
    print(f"  ║  Trade Stats")
    print(f"  ║  ──────────────────────────────────────────────────")
    print(f"  ║  Trades:            {n_trades}")
    print(f"  ║  Win Rate:          {wr:.1f}%")
    print(f"  ║  Avg R:             {avg_r:+.3f}")
    print(f"  ║  Avg Win R:         {avg_win_r:+.3f}")
    print(f"  ║  Avg Loss R:        {avg_loss_r:+.3f}")
    print(f"  ║  Total R:           {total_r:+.1f}R")
    print(f"  ║  5R+ trades:        {b5}")
    print(f"  ║  Max consec loss:   {max_consec_loss}")
    print(f"  ║  Avg contracts:     {avg_nc:.1f}")
    print(f"  ║  Avg risk/trade $:  ${avg_risk:.0f}")
    print(f"  ║")
    print(f"  ║  Daily Stats ({n_days} trading days, {years:.1f} years)")
    print(f"  ║  ──────────────────────────────────────────────────")
    print(f"  ║  Avg $/day:         ${daily_mean:+.1f}")
    print(f"  ║  Win days:          {winning_days}/{n_days} ({winning_days/n_days*100:.0f}%)")
    print(f"  ║  Best day:          ${best_day:+,.0f}")
    print(f"  ║  Worst day:         ${worst_day:+,.0f}")
    print(f"  ║  Final equity:      ${STARTING_EQUITY + total_pnl:,.0f}")
    print(f"  ║")
    print(f"  ╚═══════════════════════════════════════════════════════════╝")


def main():
    print("=" * 70)
    print("FULL STATS — NQ Real Data, $150 Risk, Dynamic Sizing")
    print("=" * 70)

    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)
    nq = nq[nq.index >= "2022-01-01"]

    for name, s in [("V8 gate=-0.1 (4.3Y NQ)", V8),
                     ("V8 gate=0.0 (4.3Y NQ)", V8G0)]:
        tdf, daily_pnl = run_full(nq, s, risk_per_trade=150)
        compute_stats(tdf, daily_pnl, name)


if __name__ == "__main__":
    main()
