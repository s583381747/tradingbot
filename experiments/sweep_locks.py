"""
Exhaustive sweep: 2-lock vs 3-lock vs 1-lock, all portion/RR combos.
Uses final strategy engine. All CPUs.
"""
import pandas as pd, numpy as np, functools, datetime as dt, itertools
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
COMM = 0.005; CAPITAL = 100000; RISK_PCT = 0.01; MAX_FWD = 180
TOUCH_TOL = 0.15; TOUCH_BELOW = 0.5; SIGNAL_OFFSET = 0.05; STOP_BUFFER = 0.3
RUNNER_TRAIL = 10; TRAIL_BUF = 0.3

_df = None; _high = None; _low = None; _close = None
_ema = None; _ema_s = None; _atr_v = None; _times = None; _n = 0

def _init():
    global _df, _high, _low, _close, _ema, _ema_s, _atr_v, _times, _n
    if _df is not None: return
    _df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    ema20 = _df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = _df['Close'].ewm(span=50, adjust=False).mean()
    tr = np.maximum(_df['High']-_df['Low'], np.maximum(
        (_df['High']-_df['Close'].shift(1)).abs(),(_df['Low']-_df['Close'].shift(1)).abs()))
    atr = tr.rolling(14).mean()
    _high = _df['High'].values; _low = _df['Low'].values; _close = _df['Close'].values
    _ema = ema20.values; _ema_s = ema50.values; _atr_v = atr.values
    _times = _df.index.time; _n = len(_df)


def run_config(args):
    """
    args = (n_locks, lock_rrs, lock_pcts, runner_trail, trail_buf)
    lock_rrs = list of RR targets [l1_rr, l2_rr, ...] or [l1_rr, l2_rr, l3_rr]
    lock_pcts = list of portions [l1_pct, l2_pct, ...], runner = 1 - sum
    """
    _init()
    n_locks, lock_rrs, lock_pcts, rt, tb = args
    runner_pct = 1.0 - sum(lock_pcts)
    if runner_pct < 0.01: return None

    equity = CAPITAL; wins = 0; losses = 0; gross_won = 0; gross_lost = 0; total_trades = 0
    peak_equity = CAPITAL; max_dd = 0
    bar = 55

    while bar < _n - MAX_FWD - 5:
        a = _atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(_ema[bar]) or np.isnan(_ema_s[bar]):
            bar += 1; continue
        if _times[bar] >= dt.time(15, 30): bar += 1; continue

        trend = 0
        if _close[bar] > _ema[bar] and _ema[bar] > _ema_s[bar]: trend = 1
        elif _close[bar] < _ema[bar] and _ema[bar] < _ema_s[bar]: trend = -1
        if trend == 0: bar += 1; continue

        tol = a * TOUCH_TOL
        if trend == 1:
            is_touch = (_low[bar] <= _ema[bar] + tol) and (_low[bar] >= _ema[bar] - a * TOUCH_BELOW)
        else:
            is_touch = (_high[bar] >= _ema[bar] - tol) and (_high[bar] <= _ema[bar] + a * TOUCH_BELOW)
        if not is_touch: bar += 1; continue

        bb = bar + 1
        if bb >= _n: bar += 1; continue
        if trend == 1 and _close[bb] <= _high[bar]: bar += 1; continue
        if trend == -1 and _close[bb] >= _low[bar]: bar += 1; continue

        if trend == 1: sig = _high[bar] + SIGNAL_OFFSET; stop = _low[bar] - STOP_BUFFER * a
        else: sig = _low[bar] - SIGNAL_OFFSET; stop = _high[bar] + STOP_BUFFER * a
        risk = abs(sig - stop)
        if risk <= 0: bar += 1; continue

        triggered = False; entry_bar = -1
        for j in range(1, 4):
            cb = bb + j
            if cb >= _n: break
            if trend == 1 and _high[cb] >= sig: triggered = True; entry_bar = cb; break
            elif trend == -1 and _low[cb] <= sig: triggered = True; entry_bar = cb; break
        if not triggered: bar += 1; continue

        shares = max(1, int(equity * RISK_PCT / risk))
        if shares * sig > equity * 0.25: shares = max(1, int(equity * 0.25 / sig))
        if equity < shares * risk or shares < 1: bar += 1; continue

        lock_shares = [max(1, int(shares * p)) for p in lock_pcts]
        lock_done = [False] * n_locks
        runner_stop = stop
        trade_pnl = -shares * COMM; remaining = shares; end_bar = entry_bar

        for k in range(1, MAX_FWD + 1):
            bi = entry_bar + k
            if bi >= _n: break
            h = _high[bi]; l = _low[bi]

            if _times[bi] >= dt.time(15, 58):
                ep = _close[bi]; trade_pnl += remaining * (ep - sig) * trend - remaining * COMM
                end_bar = bi; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * COMM
                end_bar = bi; break

            # Check each lock level
            for li in range(n_locks):
                if lock_done[li]: continue
                # All previous locks must be done
                if li > 0 and not lock_done[li-1]: continue

                target = sig + lock_rrs[li] * risk * trend
                hit = (trend == 1 and h >= target) or (trend == -1 and l <= target)
                if hit:
                    actual_exit = min(lock_shares[li], remaining - 1) if remaining > 1 else 0
                    if actual_exit > 0:
                        trade_pnl += actual_exit * lock_rrs[li] * risk - actual_exit * COMM
                        remaining -= actual_exit
                    lock_done[li] = True
                    # After first lock, move stop to breakeven
                    if li == 0:
                        if trend == 1: runner_stop = max(runner_stop, sig)
                        else: runner_stop = min(runner_stop, sig)

            # Runner trail (after first lock, or always if no locks)
            first_lock_done = lock_done[0] if n_locks > 0 else True
            if first_lock_done and k >= 5:
                sk = max(1, k - rt + 1)
                if trend == 1:
                    rl = min(_low[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                    runner_stop = max(runner_stop, rl - tb * a)
                else:
                    rh = max(_high[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                    runner_stop = min(runner_stop, rh + tb * a)
        else:
            ep = _close[min(entry_bar + MAX_FWD, _n-1)]
            trade_pnl += remaining * (ep - sig) * trend - remaining * COMM
            end_bar = min(entry_bar + MAX_FWD, _n-1)

        equity += trade_pnl; total_trades += 1
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_dd = max(max_dd, dd)
        if trade_pnl > 0: wins += 1; gross_won += trade_pnl
        else: losses += 1; gross_lost += abs(trade_pnl)
        bar = end_bar + 1

    if total_trades < 100: return None
    pf = gross_won / gross_lost if gross_lost > 0 else 0
    ret = (equity - CAPITAL) / CAPITAL * 100
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    return {
        'n_locks': n_locks,
        'rrs': tuple(lock_rrs), 'pcts': tuple(lock_pcts),
        'runner_pct': round(runner_pct, 2),
        'pf': round(pf, 3), 'ret': round(ret, 2),
        'trades': total_trades, 'wr': round(wr, 1),
        'equity': round(equity, 0),
        'max_dd': round(max_dd, 2),
    }


def main():
    configs = []

    # ═══ 1-LOCK (lock + runner) ═══
    for l1_rr in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for l1_pct in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
            configs.append((1, [l1_rr], [l1_pct], 10, 0.3))

    # ═══ 2-LOCK (lock1 + lock2 + runner) ═══
    for l1_rr in [0.3, 0.5, 0.75, 1.0]:
        for l2_rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if l2_rr <= l1_rr: continue
            for l1_pct in [0.20, 0.30, 0.40]:
                for l2_pct in [0.15, 0.20, 0.30]:
                    if l1_pct + l2_pct > 0.80: continue
                    configs.append((2, [l1_rr, l2_rr], [l1_pct, l2_pct], 10, 0.3))

    # ═══ 3-LOCK (lock1 + lock2 + lock3 + runner) ═══
    for l1_rr in [0.3, 0.5]:
        for l2_rr in [1.0, 1.5]:
            for l3_rr in [2.0, 2.5, 3.0]:
                if l3_rr <= l2_rr or l2_rr <= l1_rr: continue
                for l1p in [0.20, 0.30]:
                    for l2p in [0.15, 0.20]:
                        for l3p in [0.10, 0.15, 0.20]:
                            if l1p + l2p + l3p > 0.75: continue
                            configs.append((3, [l1_rr, l2_rr, l3_rr], [l1p, l2p, l3p], 10, 0.3))

    # ═══ 0-LOCK (pure runner, no partial exit) ═══
    configs.append((0, [], [], 10, 0.3))

    print(f"Sweeping {len(configs)} configs on {cpu_count()} CPUs...")

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        results = [r for r in pool.map(run_config, configs) if r is not None]

    results.sort(key=lambda x: x['pf'], reverse=True)

    # ═══ RESULTS ═══
    for n_locks_label in [0, 1, 2, 3]:
        group = [r for r in results if r['n_locks'] == n_locks_label]
        if not group: continue
        print(f"\n{'='*90}")
        print(f"{'NO LOCK (pure runner)' if n_locks_label == 0 else f'{n_locks_label}-LOCK'} — TOP 10")
        print(f"{'='*90}")
        print(f"{'RRs':<20}{'Pcts':<20}{'Run%':<7}{'PF':<8}{'Ret%':<9}{'Trades':<8}{'WR%':<7}{'MaxDD%':<8}")
        print("-" * 87)
        for r in group[:10]:
            print(f"{str(r['rrs']):<20}{str(r['pcts']):<20}{r['runner_pct']:<7}"
                  f"{r['pf']:<8}{r['ret']:<8}% {r['trades']:<8}{r['wr']:<6}% {r['max_dd']:<7}%")

    # Overall top 15
    print(f"\n{'='*90}")
    print(f"OVERALL TOP 15")
    print(f"{'='*90}")
    print(f"{'Locks':<7}{'RRs':<20}{'Pcts':<20}{'Run%':<7}{'PF':<8}{'Ret%':<9}{'Trades':<8}{'WR%':<7}{'MaxDD%':<8}")
    print("-" * 94)
    for r in results[:15]:
        print(f"{r['n_locks']:<7}{str(r['rrs']):<20}{str(r['pcts']):<20}{r['runner_pct']:<7}"
              f"{r['pf']:<8}{r['ret']:<8}% {r['trades']:<8}{r['wr']:<6}% {r['max_dd']:<7}%")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY BY LOCK COUNT")
    print("="*90)
    for nl in [0, 1, 2, 3]:
        g = [r for r in results if r['n_locks'] == nl]
        if not g: continue
        best = max(g, key=lambda x: x['pf'])
        avg_pf = np.mean([r['pf'] for r in g])
        print(f"  {nl}-lock: {len(g)} configs, avg PF={avg_pf:.3f}, best PF={best['pf']:.3f} "
              f"(RR={best['rrs']}, pcts={best['pcts']}, run={best['runner_pct']})")


if __name__ == "__main__":
    main()
