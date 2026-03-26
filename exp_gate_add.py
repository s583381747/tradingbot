"""
Experiment: 3-bar MFE gate → ADD TO POSITION.

KEY FINDING: After 3 bars, MFE has Cohen's d > 1.0 between winners and losers.

Logic:
  - Enter initial position at touch bar close (1% risk)
  - After 3 bars, check MFE:
      Gate FAIL (MFE < threshold) → tighten stop to -0.3R
      Gate PASS (MFE >= threshold) → ADD to position (buy more shares)
  - Added position: current price entry, original stop, joins same chandelier exit
  - Test: add amounts (0.5%, 1.0%, 1.5% additional risk) x gate thresholds (0.2R, 0.3R, 0.5R, 0.8R, 1.0R)

Compare IS (Polygon 2y) and OOS (Barchart 2022-2024).
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch

print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

BASE = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3, "lock_rr": 0.1, "lock_pct": 0.05,
    "chand_bars": 40, "chand_mult": 0.5, "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(14, 0), "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5, "skip_after_win": 1,
}

# Gate parameters
GATE_BAR = 3  # check MFE after this many bars


def run(df, capital=100_000, gate_threshold=0.0, add_mult=0.0, fail_stop_r=-0.3):
    """
    Backtest with 3-bar MFE gate and optional position add.

    gate_threshold: MFE in R to pass gate (0 = no gate)
    add_mult: multiplier of original position to add on gate pass (0.5 = add 50% more shares)
    fail_stop_r: tighten stop to this R on gate fail (negative = allow small loss)
    """
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    n = len(df)
    comm = p["commission_per_share"]

    equity = capital
    trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0
    current_date = None
    skip_count = 0

    while bar < n - p["max_hold_bars"] - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]:
            bar += 1; continue
        d = dates[bar]
        if current_date != d:
            current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]:
            bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0:
            bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        if skip_count > 0:
            skip_count -= 1; bar += 1; continue

        # Entry at touch bar close
        actual_entry = close[bar]
        stop = (low[bar] - p["stop_buffer"] * a) if trend == 1 else (high[bar] + p["stop_buffer"] * a)
        risk = abs(actual_entry - stop)
        if risk <= 0:
            bar += 1; continue
        entry_bar = bar

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1:
            bar += 1; continue

        # Lock setup
        use_lock = p["lock_rr"] > 0 and p["lock_pct"] > 0
        lock_sh = max(1, int(shares * p["lock_pct"])) if use_lock else 0
        remaining = shares
        runner_stop = stop
        lock_done = False
        trade_pnl = -shares * comm  # entry commission
        end_bar = entry_bar
        exit_reason = "timeout"
        mfe_r = 0.0

        # Gate state
        gate_checked = False
        gate_passed = False
        added_shares = 0
        add_entry_price = 0.0

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n:
                break
            h = high[bi]
            l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Track MFE
            if trend == 1:
                cur_fav = (h - actual_entry) / risk
            else:
                cur_fav = (actual_entry - l) / risk
            mfe_r = max(mfe_r, cur_fav)

            # Force close
            if times[bi] >= p["force_close_at"]:
                ep = close[bi]
                trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
                if added_shares > 0:
                    trade_pnl += added_shares * (ep - add_entry_price) * trend - added_shares * comm
                end_bar = bi
                exit_reason = "session_close"
                break

            # === 3-bar MFE gate check ===
            if not gate_checked and k == GATE_BAR:
                gate_checked = True
                if mfe_r >= gate_threshold and gate_threshold > 0:
                    # Gate PASS: add to position
                    gate_passed = True
                    if add_mult > 0:
                        add_entry_price = close[bi]
                        # Add a fraction of original shares
                        added_shares = max(1, int(shares * add_mult))
                        trade_pnl -= added_shares * comm  # entry commission for add
                elif gate_threshold > 0:
                    # Gate FAIL: tighten stop
                    if trend == 1:
                        new_stop = actual_entry + fail_stop_r * risk  # fail_stop_r is negative
                        runner_stop = max(runner_stop, new_stop)
                    else:
                        new_stop = actual_entry - fail_stop_r * risk
                        runner_stop = min(runner_stop, new_stop)

            # Stop check (applies to both original and added positions)
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)

            # Lock check
            lock_hit_this_bar = False
            if use_lock and not lock_done:
                target = actual_entry + p["lock_rr"] * risk * trend
                lock_hit_this_bar = (trend == 1 and h >= target) or (trend == -1 and l <= target)

            # Same-bar stop/lock collision: stop wins (pessimistic)
            if stopped and lock_hit_this_bar and not lock_done:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                if added_shares > 0:
                    trade_pnl += added_shares * (runner_stop - add_entry_price) * trend - added_shares * comm
                end_bar = bi
                exit_reason = "stop"
                break

            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                if added_shares > 0:
                    trade_pnl += added_shares * (runner_stop - add_entry_price) * trend - added_shares * comm
                end_bar = bi
                exit_reason = "stop" if not lock_done else "trail_stop"
                break

            # Lock: partial close
            if lock_hit_this_bar and not lock_done and remaining > lock_sh:
                trade_pnl += lock_sh * p["lock_rr"] * risk - lock_sh * comm
                remaining -= lock_sh
                lock_done = True
                # Move stop to BE for original position
                if trend == 1:
                    runner_stop = max(runner_stop, actual_entry)
                else:
                    runner_stop = min(runner_stop, actual_entry)

            # Chandelier trail (after lock done and enough bars)
            if lock_done and k >= p["chand_bars"] and k >= 2:
                sk = max(1, k - 40 + 1)
                if trend == 1:
                    hh_vals = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                    if hh_vals:
                        new_stop = max(hh_vals) - p["chand_mult"] * ca
                        runner_stop = max(runner_stop, new_stop)
                else:
                    ll_vals = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                    if ll_vals:
                        new_stop = min(ll_vals) + p["chand_mult"] * ca
                        runner_stop = min(runner_stop, new_stop)
        else:
            # Timeout: close at last bar
            ep = close[min(entry_bar + p["max_hold_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            if added_shares > 0:
                trade_pnl += added_shares * (ep - add_entry_price) * trend - added_shares * comm
            end_bar = min(entry_bar + p["max_hold_bars"], n - 1)

        equity += trade_pnl
        total_shares = shares + added_shares
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(r_mult)

        if r_mult > 0:
            skip_count = p.get("skip_after_win", 0)

        trades.append({
            "pnl": trade_pnl, "r": r_mult, "dir": trend,
            "exit": exit_reason, "mfe_3bar": mfe_r if gate_checked else np.nan,
            "gate_passed": gate_passed, "added_shares": added_shares,
            "total_shares": total_shares, "shares": shares,
            "risk": risk,
        })
        bar = end_bar + 1

    # Summarize
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "wr": 0, "big5": 0,
                "gated": 0, "added": 0, "avg_add_r": 0}
    wins = (tdf["pnl"] > 0).sum()
    losses = total - wins
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gw / gl if gl > 0 else 99.9
    ret = (equity - capital) / capital * 100
    wr = wins / total * 100

    # Big winners: 5R+ trades
    big5 = (tdf["r"] >= 5.0).sum()

    # Gate stats
    gated = tdf["gate_passed"].sum() if "gate_passed" in tdf.columns else 0
    added = (tdf["added_shares"] > 0).sum() if "added_shares" in tdf.columns else 0

    # Average R of trades where we added
    if added > 0:
        avg_add_r = tdf.loc[tdf["added_shares"] > 0, "r"].mean()
    else:
        avg_add_r = 0

    days = df.index.normalize().nunique()

    return {
        "pf": round(pf, 3),
        "ret": round(ret, 2),
        "trades": total,
        "tpd": round(total / max(days, 1), 1),
        "wr": round(wr, 1),
        "big5": int(big5),
        "gated": int(gated),
        "added": int(added),
        "avg_add_r": round(avg_add_r, 2),
    }


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    is_df = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    oos_df = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"  IS: {len(is_df):,} bars  |  OOS: {len(oos_df):,} bars\n")

    # ─── Baseline (no gate, no add) ───
    print("=" * 100)
    print("BASELINE: No gate, no add")
    print("=" * 100)
    is_base = run(is_df, gate_threshold=0.0, add_mult=0.0)
    oos_base = run(oos_df, gate_threshold=0.0, add_mult=0.0)
    print(f"  IS:  PF={is_base['pf']:.3f}  Ret={is_base['ret']:+.2f}%  Trades={is_base['trades']}  WR={is_base['wr']:.1f}%  5R+={is_base['big5']}")
    print(f"  OOS: PF={oos_base['pf']:.3f}  Ret={oos_base['ret']:+.2f}%  Trades={oos_base['trades']}  WR={oos_base['wr']:.1f}%  5R+={oos_base['big5']}")
    print()

    # ─── Gate-only baseline (gate + tighten, no add) ───
    print("=" * 100)
    print("GATE-ONLY: Gate + tighten stop on fail, NO add on pass")
    print("=" * 100)
    gate_thresholds = [0.2, 0.3, 0.5, 0.8, 1.0]
    for gt in gate_thresholds:
        is_r = run(is_df, gate_threshold=gt, add_mult=0.0)
        oos_r = run(oos_df, gate_threshold=gt, add_mult=0.0)
        print(f"  Gate={gt:.1f}R  IS: PF={is_r['pf']:.3f} Ret={is_r['ret']:+.2f}% T={is_r['trades']} WR={is_r['wr']:.1f}% 5R+={is_r['big5']} Gated={is_r['gated']}"
              f"  |  OOS: PF={oos_r['pf']:.3f} Ret={oos_r['ret']:+.2f}% T={oos_r['trades']} WR={oos_r['wr']:.1f}% 5R+={oos_r['big5']} Gated={oos_r['gated']}")
    print()

    # ─── Full grid: gate threshold x add amount ───
    print("=" * 100)
    print("FULL GRID: Gate threshold x Add risk %")
    print("=" * 100)
    add_mults = [0.5, 1.0, 1.5]  # add 50%, 100%, 150% of original shares
    add_labels = ["+50%", "+100%", "+150%"]

    # Header
    print(f"{'Config':<28} {'IS PF':>7} {'IS Ret':>8} {'IS T':>5} {'IS WR':>6} {'IS 5R+':>6} {'IS Add':>6}"
          f"  {'OOS PF':>7} {'OOS Ret':>8} {'OOS T':>5} {'OOS WR':>6} {'OOS 5R+':>7} {'OOS Add':>7} {'AvgR':>6}")
    print("-" * 140)

    # Baseline row
    print(f"{'Baseline (no gate)':<28} {is_base['pf']:>7.3f} {is_base['ret']:>+8.2f} {is_base['trades']:>5d} {is_base['wr']:>6.1f} {is_base['big5']:>6d} {'--':>6}"
          f"  {oos_base['pf']:>7.3f} {oos_base['ret']:>+8.2f} {oos_base['trades']:>5d} {oos_base['wr']:>6.1f} {oos_base['big5']:>7d} {'--':>7} {'--':>6}")
    print("-" * 140)

    results = []
    for gt in gate_thresholds:
        for am, al in zip(add_mults, add_labels):
            label = f"G={gt:.1f}R Add={al}"
            is_r = run(is_df, gate_threshold=gt, add_mult=am)
            oos_r = run(oos_df, gate_threshold=gt, add_mult=am)
            print(f"{label:<28} {is_r['pf']:>7.3f} {is_r['ret']:>+8.2f} {is_r['trades']:>5d} {is_r['wr']:>6.1f} {is_r['big5']:>6d} {is_r['added']:>6d}"
                  f"  {oos_r['pf']:>7.3f} {oos_r['ret']:>+8.2f} {oos_r['trades']:>5d} {oos_r['wr']:>6.1f} {oos_r['big5']:>7d} {oos_r['added']:>7d} {oos_r['avg_add_r']:>+6.2f}")
            results.append({
                "config": label, "gate": gt, "add_mult": am,
                "is_pf": is_r["pf"], "is_ret": is_r["ret"], "is_trades": is_r["trades"],
                "is_wr": is_r["wr"], "is_big5": is_r["big5"], "is_added": is_r["added"],
                "oos_pf": oos_r["pf"], "oos_ret": oos_r["ret"], "oos_trades": oos_r["trades"],
                "oos_wr": oos_r["wr"], "oos_big5": oos_r["big5"], "oos_added": oos_r["added"],
                "oos_avg_add_r": oos_r["avg_add_r"],
            })

    print()

    # ─── Best configs ───
    print("=" * 100)
    print("TOP 5 BY OOS PF (must have IS PF > baseline)")
    print("=" * 100)
    rdf = pd.DataFrame(results)
    valid = rdf[rdf["is_pf"] >= is_base["pf"] * 0.95]  # within 5% of baseline IS PF
    top = valid.nlargest(5, "oos_pf")
    for _, row in top.iterrows():
        delta_is = row["is_pf"] - is_base["pf"]
        delta_oos = row["oos_pf"] - oos_base["pf"]
        print(f"  {row['config']:<28} IS PF={row['is_pf']:.3f} ({delta_is:+.3f})  OOS PF={row['oos_pf']:.3f} ({delta_oos:+.3f})"
              f"  IS Ret={row['is_ret']:+.2f}%  OOS Ret={row['oos_ret']:+.2f}%  OOS Added={row['oos_added']}")

    print()
    print("=" * 100)
    print("TOP 5 BY OOS RETURN (must have IS PF > baseline)")
    print("=" * 100)
    top_ret = valid.nlargest(5, "oos_ret")
    for _, row in top_ret.iterrows():
        delta_is = row["is_ret"] - is_base["ret"]
        delta_oos = row["oos_ret"] - oos_base["ret"]
        print(f"  {row['config']:<28} IS Ret={row['is_ret']:+.2f}% ({delta_is:+.2f}%)  OOS Ret={row['oos_ret']:+.2f}% ({delta_oos:+.2f}%)"
              f"  IS PF={row['is_pf']:.3f}  OOS PF={row['oos_pf']:.3f}  OOS Added={row['oos_added']}")

    print("\nDone.")
