"""
MNQ Strategy Design — solve the cost problem by widening timeframe.

Problem: 1-min bars → 8 NQ point stop → $16 risk → $4.80 cost = 30% of 1R
Solution: wider bars → wider stops → cost becomes small fraction of 1R

Test: resample 1-min QQQ data to 3/5/10/15 min, run touch close strategy.
MNQ cost model: $4.80/RT fixed cost per contract.

For each timeframe:
  - What's the typical stop distance (in $ and NQ points)?
  - What's cost as % of 1R?
  - How many trades/day?
  - PF after costs?
  - Net daily R after all costs?
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import detect_trend, check_touch

print = functools.partial(print, flush=True)
IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

MNQ_COST_RT = 4.80  # total round-trip cost per contract
QQQ_TO_NQ_MULT = 4.0  # rough: NQ ≈ QQQ × 4 in point terms
NQ_DOLLAR_PER_POINT = 2.0  # MNQ $2/point


def resample_ohlcv(df, minutes):
    """Resample 1-min OHLCV to N-min bars."""
    if minutes == 1:
        return df
    rule = f"{minutes}min"
    resampled = df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    return resampled


def add_indicators(df, params):
    """Add EMA and ATR."""
    df = df.copy()
    df["ema20"] = df["Close"].ewm(span=params["ema_fast"], adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=params["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                     np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                                (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(params["atr_period"]).mean()
    return df


def run_mnq(df_1min, tf_minutes=1, capital_r=10000, lock_rr=0.1, lock_pct=0.05,
            chand_bars=40, chand_mult=0.5, gate_bars=3, gate_mfe=0.3, gate_tighten=-0.3,
            daily_limit=2.5, max_hold_bars=180, touch_tol=0.15, stop_buffer=0.3):
    """
    Run strategy on resampled timeframe, tracking R-multiples with MNQ cost model.
    """
    df = resample_ohlcv(df_1min, tf_minutes)
    params = {"ema_fast": 20, "ema_slow": 50, "atr_period": 14,
              "touch_tol": touch_tol, "touch_below_max": 0.5}
    df = add_indicators(df, params)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)

    # Time cutoff adjusted for timeframe
    no_entry_after = dt.time(14, 0)
    force_close_at = dt.time(15, 58)

    # Adjust max_hold and chand for timeframe
    tf_scale = tf_minutes  # e.g., 5-min bars: 40 bars = 200 min
    adj_max_hold = max(30, max_hold_bars // tf_minutes)
    adj_chand = max(8, chand_bars // tf_minutes)
    adj_gate = max(1, gate_bars // tf_minutes) if gate_bars > 0 else 0

    r_list = []; cum_r = 0; peak_r = 0; max_dd_r = 0
    bar = max(50, 14) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    risk_values = []  # track actual risk in $ for cost analysis

    while bar < n - adj_max_hold - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= no_entry_after: bar += 1; continue
        d = dates[bar]
        if current_date != d: current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= daily_limit: bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0: bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           touch_tol, 0.5):
            bar += 1; continue
        if skip_count > 0: skip_count -= 1; bar += 1; continue

        actual_entry = close[bar]
        stop = low[bar] - stop_buffer * a if trend == 1 else high[bar] + stop_buffer * a
        risk_qqq = abs(actual_entry - stop)  # QQQ dollars
        if risk_qqq <= 0: bar += 1; continue

        # Convert to NQ risk
        risk_nq_points = risk_qqq * QQQ_TO_NQ_MULT  # NQ points
        risk_nq_dollars = risk_nq_points * NQ_DOLLAR_PER_POINT  # $ per MNQ contract
        risk_values.append(risk_nq_dollars)

        # Cost as fraction of 1R
        cost_fraction = MNQ_COST_RT / risk_nq_dollars if risk_nq_dollars > 0 else 1.0

        entry_bar = bar
        runner_stop = stop; lock_done = False; mfe = 0
        trade_r = 0; end_bar = bar

        for k in range(1, adj_max_hold + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if trend == 1: mfe = max(mfe, (h - actual_entry) / risk_qqq)
            else: mfe = max(mfe, (actual_entry - l) / risk_qqq)

            if times[bi] >= force_close_at:
                trade_r = (close[bi] - actual_entry) / risk_qqq * trend
                end_bar = bi; break

            # MFE gate
            if adj_gate > 0 and k == adj_gate and not lock_done and mfe < gate_mfe:
                ns = actual_entry + gate_tighten * risk_qqq * trend
                if trend == 1: runner_stop = max(runner_stop, ns)
                else: runner_stop = min(runner_stop, ns)

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_r = (runner_stop - actual_entry) / risk_qqq * trend
                end_bar = bi; break

            if not lock_done and lock_rr > 0:
                tgt = actual_entry + lock_rr * risk_qqq * trend
                if (trend == 1 and h >= tgt) or (trend == -1 and l <= tgt):
                    lock_done = True
                    if trend == 1: runner_stop = max(runner_stop, actual_entry)
                    else: runner_stop = min(runner_stop, actual_entry)

            if lock_done and k >= adj_chand:
                sk = max(1, k - adj_chand + 1)
                hv = [high[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                lv = [low[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - chand_mult * ca)
                    else: runner_stop = min(runner_stop, min(lv) + chand_mult * ca)
        else:
            trade_r = (close[min(entry_bar + adj_max_hold, n-1)] - actual_entry) / risk_qqq * trend
            end_bar = min(entry_bar + adj_max_hold, n-1)

        # Subtract execution cost from trade R
        trade_r_after_cost = trade_r - cost_fraction

        cum_r += trade_r_after_cost
        peak_r = max(peak_r, cum_r)
        dd = peak_r - cum_r; max_dd_r = max(max_dd_r, dd)
        r_list.append({"r_raw": trade_r, "r_net": trade_r_after_cost, "cost_frac": cost_fraction})

        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = 1
        bar = end_bar + 1

    # Stats
    if not r_list:
        return {}
    rdf = pd.DataFrame(r_list)
    total = len(rdf)
    days = len(set(dates))

    raw_gw = rdf.loc[rdf["r_raw"]>0, "r_raw"].sum()
    raw_gl = abs(rdf.loc[rdf["r_raw"]<=0, "r_raw"].sum())
    raw_pf = raw_gw / raw_gl if raw_gl > 0 else 0

    net_gw = rdf.loc[rdf["r_net"]>0, "r_net"].sum()
    net_gl = abs(rdf.loc[rdf["r_net"]<=0, "r_net"].sum())
    net_pf = net_gw / net_gl if net_gl > 0 else 0

    avg_risk = np.mean(risk_values) if risk_values else 0
    avg_cost_frac = rdf["cost_frac"].mean()

    return {
        "tf": tf_minutes,
        "raw_pf": round(raw_pf, 3), "net_pf": round(net_pf, 3),
        "total_r_raw": round(rdf["r_raw"].sum(), 1),
        "total_r_net": round(rdf["r_net"].sum(), 1),
        "daily_r_net": round(rdf["r_net"].sum() / max(days, 1), 3),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "max_dd": round(max_dd_r, 1),
        "big5": int((rdf["r_raw"] >= 5).sum()),
        "avg_risk_nq": round(avg_risk, 2),
        "cost_pct": round(avg_cost_frac * 100, 1),
    }


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    print(f"{'='*120}")
    print(f"  MNQ STRATEGY DESIGN — Timeframe vs Cost Analysis")
    print(f"{'='*120}")

    hdr = (f"  {'TF':>4} {'Raw PF':>7} {'Net PF':>7} {'Raw R':>8} {'Net R':>8} {'R/day':>7}"
           f" {'Trades':>7} {'T/d':>5} {'MaxDD':>7} {'5R+':>5}"
           f" {'Avg Risk$':>10} {'Cost%':>7}")
    sep = f"  {'-'*105}"

    for label, df_data in [("IN-SAMPLE", df_is), ("OUT-OF-SAMPLE", df_oos)]:
        print(f"\n  === {label} ===")
        print(hdr); print(sep)

        for tf in [1, 2, 3, 5, 7, 10, 15]:
            r = run_mnq(df_data, tf_minutes=tf)
            if not r: continue
            mark = " ★" if r["net_pf"] > 1.3 and r["daily_r_net"] > 0.5 else ""
            print(f"  {tf:>3}m {r['raw_pf']:>6.3f} {r['net_pf']:>6.3f} {r['total_r_raw']:>+7.1f}"
                  f" {r['total_r_net']:>+7.1f} {r['daily_r_net']:>+6.3f}"
                  f" {r['trades']:>6} {r['tpd']:>4.1f} {r['max_dd']:>6.1f}R {r['big5']:>4}"
                  f" ${r['avg_risk_nq']:>8.2f} {r['cost_pct']:>6.1f}%{mark}")

    # ═══ Best TF: deep optimization ═══
    print(f"\n{'='*120}")
    print(f"  DEEP OPTIMIZATION ON BEST TIMEFRAME")
    print(f"{'='*120}")

    # Test various exit configs on best TF
    for tf in [3, 5]:
        print(f"\n  === {tf}-min bar optimization ===")
        print(f"  {'Config':<45} {'IS Net PF':>9} {'IS R/day':>8} {'OOS Net PF':>10} {'OOS R/day':>9}")
        print(f"  {'-'*85}")

        configs = [
            ("Baseline (lock 0.1R/5%)", {"lock_rr": 0.1, "lock_pct": 0.05}),
            ("No lock, no BE", {"lock_rr": 0, "lock_pct": 0}),
            ("Lock 0.3R/10% (cover costs)", {"lock_rr": 0.3, "lock_pct": 0.10}),
            ("Lock 0.5R/15%", {"lock_rr": 0.5, "lock_pct": 0.15}),
            ("Lock 1.0R/20%", {"lock_rr": 1.0, "lock_pct": 0.20}),
            ("Wider stop (0.5 ATR buffer)", {"stop_buffer": 0.5}),
            ("Wider stop (0.8 ATR buffer)", {"stop_buffer": 0.8}),
            ("Tighter chandelier (0.3)", {"chand_mult": 0.3}),
            ("Wider chandelier (1.0)", {"chand_mult": 1.0}),
            ("No gate (no tighten)", {"gate_bars": 0}),
            ("No lock + wider stop 0.5", {"lock_rr": 0, "lock_pct": 0, "stop_buffer": 0.5}),
            ("No lock + wider stop 0.8", {"lock_rr": 0, "lock_pct": 0, "stop_buffer": 0.8}),
            ("Lock 0.5R + wider stop 0.5", {"lock_rr": 0.5, "lock_pct": 0.15, "stop_buffer": 0.5}),
            ("Daily limit 1.5R", {"daily_limit": 1.5}),
            ("Touch tol 0.20", {"touch_tol": 0.20}),
            ("Touch tol 0.25", {"touch_tol": 0.25}),
        ]

        for config_label, extra_params in configs:
            params = {"tf_minutes": tf, **extra_params}
            r_is = run_mnq(df_is, **params)
            r_oos = run_mnq(df_oos, **params)
            if not r_is or not r_oos: continue
            holds = "✅" if r_is["net_pf"] > 1.2 and r_oos["net_pf"] > 1.2 else "❌"
            print(f"  {config_label:<45} {r_is['net_pf']:>8.3f} {r_is['daily_r_net']:>+7.3f}"
                  f" {r_oos['net_pf']:>9.3f} {r_oos['daily_r_net']:>+8.3f}  {holds}")

    # ═══ PROP FIRM PROJECTION ═══
    print(f"\n{'='*120}")
    print(f"  PROP FIRM PROJECTION (best MNQ config)")
    print(f"{'='*120}")

    # Run best config
    for tf in [3, 5]:
        best_is = run_mnq(df_is, tf_minutes=tf, lock_rr=0, lock_pct=0, stop_buffer=0.5)
        best_oos = run_mnq(df_oos, tf_minutes=tf, lock_rr=0, lock_pct=0, stop_buffer=0.5)
        if not best_is or not best_oos: continue

        avg_daily_r = (best_is["daily_r_net"] + best_oos["daily_r_net"]) / 2
        avg_risk = (best_is["avg_risk_nq"] + best_oos["avg_risk_nq"]) / 2

        print(f"\n  {tf}-min, No lock, stop_buffer=0.5:")
        print(f"  IS:  Net PF={best_is['net_pf']}, R/day={best_is['daily_r_net']:+.3f}, MaxDD={best_is['max_dd']:.1f}R")
        print(f"  OOS: Net PF={best_oos['net_pf']}, R/day={best_oos['daily_r_net']:+.3f}, MaxDD={best_oos['max_dd']:.1f}R")
        print(f"  Avg risk/contract: ${avg_risk:.2f}")
        print(f"  Cost/trade: ${MNQ_COST_RT:.2f} = {MNQ_COST_RT/avg_risk*100:.1f}% of 1R")

        if avg_daily_r > 0:
            for firm, dd in [("Apex 50K", 2500), ("Topstep 50K", 2000)]:
                contracts = dd / (25 * avg_risk)  # 25 divisions
                daily_dollars = avg_daily_r * avg_risk * contracts
                days_to_target = dd * 1.2 / daily_dollars if daily_dollars > 0 else 999
                print(f"\n  {firm} (DD=${dd}, 25div):")
                print(f"    Contracts: {contracts:.1f} MNQ")
                print(f"    Daily $: ${daily_dollars:.0f}")
                print(f"    Days to target: {days_to_target:.1f}")


if __name__ == "__main__":
    main()
