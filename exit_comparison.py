"""
Exit strategy comparison: walk-forward + overfitting assessment.
Tests all candidate exit methods on Y1/Y2 independently.
"""
import pandas as pd, numpy as np, functools, datetime as dt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
COMM = 0.005; CAPITAL = 100000; RISK_PCT = 0.01; MAX_FWD = 180
TOUCH_TOL = 0.15; TOUCH_BELOW = 0.5; SIGNAL_OFFSET = 0.05; STOP_BUFFER = 0.3

def run_period(args):
    config_name, config, start_date, end_date, period_label = args

    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df) < 2000:
        return None

    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    tr = np.maximum(df['High']-df['Low'], np.maximum(
        (df['High']-df['Close'].shift(1)).abs(),(df['Low']-df['Close'].shift(1)).abs()))
    atr = tr.rolling(14).mean()
    high = df['High'].values; low = df['Low'].values; close = df['Close'].values
    ema = ema20.values; ema_s = ema50.values; atr_v = atr.values
    times = df.index.time; n = len(df); days = df.index.normalize().nunique()

    pa_pct = config['pa_pct']; pa_mode = config['pa_mode']
    pb_chand = config['pb_chand']

    equity = CAPITAL; wins = 0; losses = 0; gw = 0; gl = 0
    all_r = []; peak_eq = CAPITAL; max_dd = 0; bar = 55

    while bar < n - MAX_FWD - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]): bar += 1; continue
        if times[bar] >= dt.time(15, 30): bar += 1; continue
        trend = 0
        if close[bar] > ema[bar] and ema[bar] > ema_s[bar]: trend = 1
        elif close[bar] < ema[bar] and ema[bar] < ema_s[bar]: trend = -1
        if trend == 0: bar += 1; continue
        tol = a * TOUCH_TOL
        if trend == 1: it = (low[bar] <= ema[bar]+tol) and (low[bar] >= ema[bar]-a*TOUCH_BELOW)
        else: it = (high[bar] >= ema[bar]-tol) and (high[bar] <= ema[bar]+a*TOUCH_BELOW)
        if not it: bar += 1; continue
        bb = bar + 1
        if bb >= n: bar += 1; continue
        if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
        if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
        if trend == 1: sig = high[bar]+SIGNAL_OFFSET; stop = low[bar]-STOP_BUFFER*a
        else: sig = low[bar]-SIGNAL_OFFSET; stop = high[bar]+STOP_BUFFER*a
        risk = abs(sig-stop)
        if risk <= 0: bar += 1; continue
        triggered = False; entry_bar = -1
        for j in range(1, 4):
            cb = bb+j
            if cb >= n: break
            if trend == 1 and high[cb] >= sig: triggered = True; entry_bar = cb; break
            elif trend == -1 and low[cb] <= sig: triggered = True; entry_bar = cb; break
        if not triggered: bar += 1; continue

        shares = max(1, int(equity*RISK_PCT/risk))
        if shares*abs(sig) > equity*0.25: shares = max(1, int(equity*0.25/abs(sig)))
        pa_shares = max(1, int(shares*pa_pct)) if pa_pct > 0 else 0
        pb_shares = shares - pa_shares

        trade_pnl = -shares*COMM; pa_stop = stop; pb_stop = stop
        pa_done = False; pa_trailing = False; remaining_a = pa_shares; remaining_b = pb_shares

        for k in range(1, MAX_FWD+1):
            bi = entry_bar+k
            if bi >= n: break
            h = high[bi]; l = low[bi]; c = close[bi]
            cur_atr = atr_v[bi] if bi < n and not np.isnan(atr_v[bi]) else a

            if times[bi] >= dt.time(15, 58):
                if remaining_a > 0: trade_pnl += remaining_a*(c-sig)*trend - remaining_a*COMM
                trade_pnl += remaining_b*(c-sig)*trend - remaining_b*COMM
                break

            # Portion A stop
            if remaining_a > 0 and not pa_done:
                if (trend == 1 and l <= pa_stop) or (trend == -1 and h >= pa_stop):
                    trade_pnl += remaining_a*(pa_stop-sig)*trend - remaining_a*COMM
                    pa_done = True; remaining_a = 0

            # Portion B stop
            if (trend == 1 and l <= pb_stop) or (trend == -1 and h >= pb_stop):
                trade_pnl += remaining_b*(pb_stop-sig)*trend - remaining_b*COMM
                if remaining_a > 0:
                    trade_pnl += remaining_a*(pb_stop-sig)*trend - remaining_a*COMM
                    pa_done = True; remaining_a = 0
                break

            # Portion A exit
            if remaining_a > 0 and not pa_done:
                if pa_mode == 'fixed':
                    target = sig + config['pa_rr']*risk*trend
                    if (trend == 1 and h >= target) or (trend == -1 and l <= target):
                        trade_pnl += remaining_a*config['pa_rr']*risk - remaining_a*COMM
                        pa_done = True; remaining_a = 0
                        if trend == 1: pb_stop = max(pb_stop, sig)
                        else: pb_stop = min(pb_stop, sig)
                elif pa_mode == 'trail':
                    profit_rr = (c-sig)*trend/risk if risk > 0 else 0
                    if profit_rr >= 0.3 and not pa_trailing:
                        pa_trailing = True
                        if trend == 1: pa_stop = max(pa_stop, sig); pb_stop = max(pb_stop, sig)
                        else: pa_stop = min(pa_stop, sig); pb_stop = min(pb_stop, sig)
                    if pa_trailing:
                        pa_cb, pa_cm = config['pa_chand']
                        if k >= pa_cb:
                            sk = max(1, k-pa_cb+1)
                            if trend == 1:
                                hh = max(high[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < n)
                                pa_stop = max(pa_stop, hh - pa_cm*cur_atr)
                            else:
                                ll = min(low[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < n)
                                pa_stop = min(pa_stop, ll + pa_cm*cur_atr)

            # Portion B trail
            b_cb, b_cm = pb_chand
            if k >= b_cb and (pa_done or pa_trailing):
                sk = max(1, k-b_cb+1)
                if trend == 1:
                    hh = max(high[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < n)
                    pb_stop = max(pb_stop, hh - b_cm*cur_atr)
                else:
                    ll = min(low[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < n)
                    pb_stop = min(pb_stop, ll + b_cm*cur_atr)
        else:
            ep = close[min(entry_bar+MAX_FWD, n-1)]
            if remaining_a > 0: trade_pnl += remaining_a*(ep-sig)*trend - remaining_a*COMM
            trade_pnl += remaining_b*(ep-sig)*trend - remaining_b*COMM

        equity += trade_pnl
        r_val = trade_pnl/(shares*risk) if risk > 0 and shares > 0 else 0
        all_r.append(r_val)
        peak_eq = max(peak_eq, equity)
        dd = (peak_eq-equity)/peak_eq*100 if peak_eq > 0 else 0
        max_dd = max(max_dd, dd)
        if trade_pnl > 0: wins += 1; gw += trade_pnl
        else: losses += 1; gl += abs(trade_pnl)
        bar = entry_bar + 1

    total = wins+losses
    if total < 10: return None
    pf = gw/gl if gl > 0 else 0
    ret = (equity-CAPITAL)/CAPITAL*100
    wr = wins/total*100
    r_arr = np.array(all_r)

    return {
        'config': config_name, 'period': period_label,
        'pf': round(pf,3), 'ret': round(ret,2), 'wr': round(wr,1),
        'trades': total, 'max_dd': round(max_dd,2),
        'avg_r': round(r_arr.mean(),4), 'tpd': round(total/max(days,1),1),
    }


def main():
    configs = {
        "A: Fixed 0.3R + Chand30/1.5": {
            'pa_pct': 0.20, 'pa_mode': 'fixed', 'pa_rr': 0.3,
            'pb_chand': (30, 1.5)},
        "B: Trail(10/0.8) + Trail(40/1.0)": {
            'pa_pct': 0.20, 'pa_mode': 'trail', 'pa_chand': (10, 0.8),
            'pb_chand': (40, 1.0)},
        "C: Trail(15/1.0) + Chand30/1.5": {
            'pa_pct': 0.20, 'pa_mode': 'trail', 'pa_chand': (15, 1.0),
            'pb_chand': (30, 1.5)},
        "D: 100% Chand30/1.5 (no split)": {
            'pa_pct': 0.0, 'pa_mode': 'fixed', 'pa_rr': 99,
            'pb_chand': (30, 1.5)},
        "E: Trail(10/0.8) + Chand30/1.5": {
            'pa_pct': 0.20, 'pa_mode': 'trail', 'pa_chand': (10, 0.8),
            'pb_chand': (30, 1.5)},
        "F: Fixed 0.3R + Trail(40/1.0)": {
            'pa_pct': 0.20, 'pa_mode': 'fixed', 'pa_rr': 0.3,
            'pb_chand': (40, 1.0)},
    }

    periods = [
        ("2024-01-01", "2026-12-31", "FULL"),
        ("2024-01-01", "2025-03-22", "Y1"),
        ("2025-03-22", "2026-12-31", "Y2"),
        ("2024-04-01", "2024-10-01", "H1-24"),
        ("2024-10-01", "2025-04-01", "H2-24"),
        ("2025-04-01", "2025-10-01", "H1-25"),
        ("2025-10-01", "2026-04-01", "H2-25"),
    ]

    tasks = []
    for cname, cfg in configs.items():
        for start, end, plabel in periods:
            tasks.append((cname, cfg, start, end, plabel))

    print(f"Running {len(tasks)} configs × periods on {cpu_count()} CPUs...\n")

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        results = [r for r in pool.map(run_period, tasks) if r is not None]

    # ═══ COMPARISON TABLE ═══
    print("=" * 120)
    print("EXIT STRATEGY COMPARISON: FULL + WALK-FORWARD")
    print("=" * 120)

    for period in ["FULL", "Y1", "Y2"]:
        print(f"\n--- {period} ---")
        print(f"{'Config':<42} {'PF':>7} {'Ret%':>8} {'WR%':>6} {'DD%':>6} {'AvgR':>8} {'T/d':>5} {'Trades':>7}")
        print("-" * 95)
        subset = sorted([r for r in results if r['period'] == period],
                       key=lambda x: x['pf'], reverse=True)
        for r in subset:
            print(f"{r['config']:<42} {r['pf']:>7.3f} {r['ret']:>7.2f}% {r['wr']:>5.1f}% "
                  f"{r['max_dd']:>5.2f}% {r['avg_r']:>+7.4f} {r['tpd']:>4.1f} {r['trades']:>7}")

    # ═══ WALK-FORWARD CONSISTENCY ═══
    print(f"\n{'='*120}")
    print("WALK-FORWARD CONSISTENCY (Y1 vs Y2)")
    print("="*120)
    print(f"{'Config':<42} {'Y1 PF':>7} {'Y2 PF':>7} {'Δ PF':>7} {'Y1 Ret':>8} {'Y2 Ret':>8} {'Stable?':>8}")
    print("-" * 90)

    for cname in configs:
        y1 = next((r for r in results if r['config'] == cname and r['period'] == 'Y1'), None)
        y2 = next((r for r in results if r['config'] == cname and r['period'] == 'Y2'), None)
        if not y1 or not y2: continue
        delta_pf = y2['pf'] - y1['pf']
        # Stability: Y2 PF within 30% of Y1 AND both > 1.0
        stable = abs(delta_pf) / max(y1['pf'], 0.001) < 0.30 and y1['pf'] > 1.0 and y2['pf'] > 1.0
        print(f"{cname:<42} {y1['pf']:>7.3f} {y2['pf']:>7.3f} {delta_pf:>+6.3f} "
              f"{y1['ret']:>7.2f}% {y2['ret']:>7.2f}% {'✓' if stable else '✗':>7}")

    # ═══ HALF-YEAR CONSISTENCY ═══
    print(f"\n{'='*120}")
    print("HALF-YEAR CONSISTENCY")
    print("="*120)

    for cname in configs:
        halves = [r for r in results if r['config'] == cname and r['period'].startswith('H')]
        halves.sort(key=lambda x: x['period'])
        if not halves: continue
        pfs = [h['pf'] for h in halves]
        rets = [h['ret'] for h in halves]
        all_profitable = all(pf > 1.0 for pf in pfs)
        pf_std = np.std(pfs)
        h_str = " | ".join(f"{h['period']}={h['pf']:.2f}" for h in halves)
        print(f"  {cname:<40} {h_str}  std={pf_std:.3f} {'✓' if all_profitable else '✗'}")

    # ═══ OVERFITTING RISK ASSESSMENT ═══
    print(f"\n{'='*120}")
    print("OVERFITTING RISK ASSESSMENT")
    print("="*120)

    for cname in configs:
        full = next((r for r in results if r['config'] == cname and r['period'] == 'FULL'), None)
        y1 = next((r for r in results if r['config'] == cname and r['period'] == 'Y1'), None)
        y2 = next((r for r in results if r['config'] == cname and r['period'] == 'Y2'), None)
        halves = [r for r in results if r['config'] == cname and r['period'].startswith('H')]

        if not full or not y1 or not y2: continue

        risks = []

        # 1. Y1→Y2 degradation
        deg = (y1['pf'] - y2['pf']) / y1['pf'] * 100
        if deg > 20: risks.append(f"Y1→Y2 degradation {deg:.0f}%")

        # 2. Half-year variance
        if halves:
            pf_std = np.std([h['pf'] for h in halves])
            if pf_std > 0.3: risks.append(f"Half-year PF variance high (std={pf_std:.2f})")

        # 3. Any losing half
        losing_halves = [h for h in halves if h['pf'] < 1.0]
        if losing_halves: risks.append(f"{len(losing_halves)} losing half-year periods")

        # 4. Low trade count
        if full['trades'] < 500: risks.append(f"Low trade count ({full['trades']})")

        # 5. Parameter count (complexity penalty)
        n_params = sum(1 for k in ['pa_pct','pa_mode','pa_rr','pa_chand','pb_chand']
                       if k in configs[cname] and configs[cname][k] not in [0, 0.0, 99, 'fixed'])
        if n_params > 4: risks.append(f"High param count ({n_params})")

        risk_level = "LOW" if len(risks) == 0 else "MEDIUM" if len(risks) <= 1 else "HIGH"
        print(f"\n  {cname}")
        print(f"    Overfit risk: {risk_level}")
        print(f"    Full PF={full['pf']:.3f} | Y1={y1['pf']:.3f} → Y2={y2['pf']:.3f} (Δ={deg:+.0f}%)")
        if risks:
            for r in risks:
                print(f"    ⚠ {r}")
        else:
            print(f"    ✓ No red flags")


if __name__ == "__main__":
    main()
