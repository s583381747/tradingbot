"""
Volume Node S/R — MNQ 3min framework.

Hypothesis: High-volume price levels act as S/R. Entries near volume nodes
(support for longs, resistance for shorts) have higher success rate.

Two approaches:
  A. FILTER: Only enter if touch is near a volume node (adds confluence)
  B. EXIT: Use volume nodes as profit target (replace or augment chandelier)
  C. STOP: Place stop behind nearest volume node (extra protection)

Volume profile: rolling N-bar volume-at-price histogram.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

S = {
    "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
}

QQQ_TO_NQ = 40; MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00


def resample(df, m):
    if m <= 1: return df
    return df.resample(f"{m}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def add_indicators(df, s):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=s["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(s["atr_period"]).mean()
    return df


def compute_volume_profile(high, low, close, volume, end_bar, lookback=100, n_bins=50):
    """Compute volume-at-price profile for the last `lookback` bars before `end_bar`.
    Returns sorted list of (price_level, volume) for high-volume nodes."""
    start = max(0, end_bar - lookback)
    if start >= end_bar:
        return []

    h_slice = high[start:end_bar]
    l_slice = low[start:end_bar]
    v_slice = volume[start:end_bar]

    price_min = l_slice.min()
    price_max = h_slice.max()
    if price_max <= price_min:
        return []

    bin_size = (price_max - price_min) / n_bins
    profile = np.zeros(n_bins)

    for i in range(len(h_slice)):
        # Distribute volume evenly across bar's range
        bar_low = l_slice[i]
        bar_high = h_slice[i]
        bar_vol = v_slice[i]
        low_bin = max(0, int((bar_low - price_min) / bin_size))
        high_bin = min(n_bins - 1, int((bar_high - price_min) / bin_size))
        n_bar_bins = high_bin - low_bin + 1
        if n_bar_bins > 0 and bar_vol > 0:
            vol_per_bin = bar_vol / n_bar_bins
            for b in range(low_bin, high_bin + 1):
                profile[b] += vol_per_bin

    # Find high-volume nodes (above 75th percentile)
    threshold = np.percentile(profile, 75)
    nodes = []
    for b in range(n_bins):
        if profile[b] >= threshold:
            price = price_min + (b + 0.5) * bin_size
            nodes.append((price, profile[b]))

    return sorted(nodes, key=lambda x: x[1], reverse=True)


def find_nearest_node(nodes, price, direction, atr_val):
    """Find nearest volume node in the specified direction.
    direction=1: look for support below price (for longs)
    direction=-1: look for resistance above price (for shorts)
    Returns (node_price, distance_in_atr) or None."""
    best = None
    best_dist = float('inf')

    for node_price, _ in nodes:
        if direction == 1:  # long: support below
            dist = price - node_price
            if dist > 0 and dist < best_dist:
                best = node_price
                best_dist = dist
        else:  # short: resistance above
            dist = node_price - price
            if dist > 0 and dist < best_dist:
                best = node_price
                best_dist = dist

    if best is None:
        return None, 0
    return best, best_dist / atr_val


def run(df_1min, s=S, vol_mode="none", vol_lookback=100, vol_filter_max_dist=1.0):
    """
    vol_mode:
      "none" — baseline, no volume S/R
      "filter" — only enter if touch is near a volume node
      "target" — exit at nearest resistance node (longs) / support node (shorts)
      "both" — filter + target
    """
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    volume = df["Volume"].values
    ema_f = df["ema_f"].values; ema_s = df["ema_s"].values; atr_arr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    tf = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    trades = []; bar = max(s["ema_slow"], s["atr_period"], vol_lookback) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0
    filtered_out = 0; target_exits = 0

    while bar < n - max_hold - 5:
        a = atr_arr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema_f[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= s["no_entry_after"]:
            bar += 1; continue
        d = dates[bar]
        if current_date != d:
            current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= s["daily_loss_r"]:
            bar += 1; continue

        c = close[bar]
        if c > ema_f[bar] and ema_f[bar] > ema_s[bar]: trend = 1
        elif c < ema_f[bar] and ema_f[bar] < ema_s[bar]: trend = -1
        else: bar += 1; continue

        tol = a * s["touch_tol"]
        if trend == 1:
            touch = low[bar] <= ema_f[bar] + tol and low[bar] >= ema_f[bar] - a * s["touch_below_max"]
        else:
            touch = high[bar] >= ema_f[bar] - tol and high[bar] <= ema_f[bar] + a * s["touch_below_max"]
        if not touch: bar += 1; continue
        if skip_count > 0: skip_count -= 1; bar += 1; continue

        # Volume S/R filter
        if vol_mode in ("filter", "both"):
            nodes = compute_volume_profile(high, low, close, volume, bar, vol_lookback)
            if nodes:
                # For longs: check if there's support near/below entry
                # For shorts: check if there's resistance near/above entry
                node_price, node_dist = find_nearest_node(nodes, close[bar], trend, a)
                if node_price is None or node_dist > vol_filter_max_dist:
                    filtered_out += 1
                    bar += 1; continue

        # Compute volume target for exits
        vol_target = None
        if vol_mode in ("target", "both"):
            nodes = compute_volume_profile(high, low, close, volume, bar, vol_lookback)
            if nodes:
                # For longs: target nearest resistance ABOVE entry
                # For shorts: target nearest support BELOW entry
                for np_price, _ in nodes:
                    if trend == 1 and np_price > close[bar] + a * 0.5:
                        vol_target = np_price; break
                    elif trend == -1 and np_price < close[bar] - a * 0.5:
                        vol_target = np_price; break

        # Execute trade
        entry = close[bar]
        stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        risk_qqq = abs(entry - stop)
        if risk_qqq <= 0: bar += 1; continue
        risk_mnq = risk_qqq * QQQ_TO_NQ * MNQ_PER_POINT * nc
        entry_cost = COMM_RT * nc / 2 + SPREAD
        entry_bar = bar; runner_stop = stop; be_triggered = False
        mfe = 0.0; trade_r = 0.0; end_bar = bar; exit_reason = "timeout"

        for k in range(1, max_hold + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_arr[bi] if not np.isnan(atr_arr[bi]) else a

            if trend == 1: mfe = max(mfe, (h - entry) / risk_qqq)
            else: mfe = max(mfe, (entry - l) / risk_qqq)

            if times[bi] >= s["force_close_at"]:
                trade_r = (close[bi] - entry) / risk_qqq * trend
                end_bar = bi; exit_reason = "close"; break

            # Volume target exit (only after BE triggered)
            if vol_target is not None and be_triggered:
                if (trend == 1 and h >= vol_target) or (trend == -1 and l <= vol_target):
                    trade_r = (vol_target - entry) / risk_qqq * trend
                    end_bar = bi; exit_reason = "vol_target"; target_exits += 1; break

            if gate_b > 0 and k == gate_b and not be_triggered:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * risk_qqq * trend
                    if trend == 1: runner_stop = max(runner_stop, ns)
                    else: runner_stop = min(runner_stop, ns)

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_r = (runner_stop - entry) / risk_qqq * trend
                end_bar = bi
                if be_triggered:
                    be_ref = entry + s["be_stop_r"] * risk_qqq * trend
                    exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_qqq else "trail"
                else: exit_reason = "stop"
                break

            if not be_triggered and s["be_trigger_r"] > 0:
                tp = entry + s["be_trigger_r"] * risk_qqq * trend
                if (trend == 1 and h >= tp) or (trend == -1 and l <= tp):
                    be_triggered = True
                    bl = entry + s["be_stop_r"] * risk_qqq * trend
                    if trend == 1: runner_stop = max(runner_stop, bl)
                    else: runner_stop = min(runner_stop, bl)

            if be_triggered and k >= chand_b:
                sk = max(1, k - chand_b + 1)
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
        else:
            trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_qqq * trend
            end_bar = min(entry_bar + max_hold, n - 1)

        raw_pnl = trade_r * risk_mnq
        exit_comm = COMM_RT * nc / 2
        exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
        be_slip_cost = BE_SLIP if exit_reason == "be" else 0
        net_pnl = raw_pnl - (entry_cost + exit_comm + exit_slip + be_slip_cost)

        cum_pnl += net_pnl; peak_pnl = max(peak_pnl, cum_pnl)
        max_dd = max(max_dd, peak_pnl - cum_pnl)
        trades.append({"net_pnl": net_pnl, "raw_r": trade_r, "exit": exit_reason, "risk_$": risk_mnq})
        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf); days = len(set(dates))
    if total == 0:
        return {"net_pf": 0, "daily_pnl": 0, "max_dd": 0, "trades": 0}
    gw = tdf.loc[tdf["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(tdf.loc[tdf["net_pnl"] <= 0, "net_pnl"].sum())
    exits = tdf["exit"].value_counts().to_dict()

    return {
        "net_pf": round(gw / gl if gl > 0 else 0, 3),
        "daily_pnl": round(cum_pnl / max(days, 1), 1),
        "max_dd": round(max_dd, 0),
        "total_pnl": round(cum_pnl, 0),
        "trades": total,
        "tpd": round(total / max(days, 1), 1),
        "big5": int((tdf["raw_r"] >= 5).sum()),
        "exits": exits,
        "filtered": filtered_out,
        "vol_exits": target_exits,
    }


if __name__ == "__main__":
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    print("=" * 80)
    print("VOLUME NODE S/R — MNQ 3min")
    print("=" * 80)

    # Baseline
    ri = run(df_is, vol_mode="none")
    ro = run(df_oos, vol_mode="none")
    print(f"\n  BASE  IS: PF={ri['net_pf']:.3f} $/d={ri['daily_pnl']:+.1f} DD=${ri['max_dd']:.0f} "
          f"N={ri['trades']} 5R+={ri['big5']}")
    print(f"  BASE OOS: PF={ro['net_pf']:.3f} $/d={ro['daily_pnl']:+.1f} DD=${ro['max_dd']:.0f} "
          f"N={ro['trades']} 5R+={ro['big5']}")

    # ═══ A: FILTER MODE ═══
    print(f"\n{'='*80}")
    print("A: VOLUME S/R FILTER (only enter near volume support/resistance)")
    print(f"{'='*80}")
    print(f"\n  {'LB':>4} {'MaxD':>5} {'IS_PF':>7} {'IS_$/d':>8} {'IS_DD':>7} {'IS_N':>6} {'IS_5R':>5} "
          f"{'OOS_PF':>7} {'OOS_$/d':>8} {'OOS_DD':>7} {'OOS_N':>6} {'OOS_5R':>5} {'Filt':>5}")

    for lb in [50, 100, 200]:
        for max_dist in [0.5, 1.0, 1.5, 2.0]:
            ri = run(df_is, vol_mode="filter", vol_lookback=lb, vol_filter_max_dist=max_dist)
            ro = run(df_oos, vol_mode="filter", vol_lookback=lb, vol_filter_max_dist=max_dist)
            print(f"  {lb:>4} {max_dist:>5.1f} {ri['net_pf']:>7.3f} {ri['daily_pnl']:>+8.1f} "
                  f"{ri['max_dd']:>7.0f} {ri['trades']:>6} {ri['big5']:>5} "
                  f"{ro['net_pf']:>7.3f} {ro['daily_pnl']:>+8.1f} "
                  f"{ro['max_dd']:>7.0f} {ro['trades']:>6} {ro['big5']:>5} {ri['filtered']:>5}")

    # ═══ B: TARGET MODE ═══
    print(f"\n{'='*80}")
    print("B: VOLUME NODE TARGET (exit at volume resistance/support)")
    print(f"{'='*80}")
    print(f"\n  {'LB':>4} {'IS_PF':>7} {'IS_$/d':>8} {'IS_DD':>7} {'IS_N':>6} {'IS_5R':>5} "
          f"{'OOS_PF':>7} {'OOS_$/d':>8} {'OOS_DD':>7} {'OOS_N':>6} {'OOS_5R':>5} {'VolEx':>6}")

    for lb in [50, 100, 200]:
        ri = run(df_is, vol_mode="target", vol_lookback=lb)
        ro = run(df_oos, vol_mode="target", vol_lookback=lb)
        print(f"  {lb:>4} {ri['net_pf']:>7.3f} {ri['daily_pnl']:>+8.1f} "
              f"{ri['max_dd']:>7.0f} {ri['trades']:>6} {ri['big5']:>5} "
              f"{ro['net_pf']:>7.3f} {ro['daily_pnl']:>+8.1f} "
              f"{ro['max_dd']:>7.0f} {ro['trades']:>6} {ro['big5']:>5} {ri['vol_exits']:>6}")

    # ═══ C: COMBINED (BOTH) ═══
    print(f"\n{'='*80}")
    print("C: COMBINED (filter + target)")
    print(f"{'='*80}")

    for lb in [100]:
        for max_dist in [1.0, 1.5]:
            ri = run(df_is, vol_mode="both", vol_lookback=lb, vol_filter_max_dist=max_dist)
            ro = run(df_oos, vol_mode="both", vol_lookback=lb, vol_filter_max_dist=max_dist)
            print(f"\n  LB={lb} MaxDist={max_dist}")
            print(f"    IS:  PF={ri['net_pf']:.3f} $/d={ri['daily_pnl']:+.1f} DD=${ri['max_dd']:.0f} "
                  f"N={ri['trades']} 5R+={ri['big5']} Exits={ri['exits']}")
            print(f"    OOS: PF={ro['net_pf']:.3f} $/d={ro['daily_pnl']:+.1f} DD=${ro['max_dd']:.0f} "
                  f"N={ro['trades']} 5R+={ro['big5']} Exits={ro['exits']}")
