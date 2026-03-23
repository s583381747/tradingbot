"""Sweep parameters in pandas to maximize PF. Uses all CPUs."""
import pandas as pd, numpy as np, functools, os, sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import datetime as dt

print = functools.partial(print, flush=True)

DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
SIGNAL_OFFSET = 0.05; STOP_BUFFER = 0.3
RISK_PCT = 0.01; CAPITAL = 100000; COMM = 0.005; MAX_FWD = 180

# Preload data globally
_df = None; _high = None; _low = None; _close = None
_ema = None; _ema_s = None; _atr_v = None; _times = None; _n = 0

def _init():
    global _df, _high, _low, _close, _ema, _ema_s, _atr_v, _times, _n
    if _df is not None: return
    _df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    ema20 = _df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = _df['Close'].ewm(span=50, adjust=False).mean()
    tr = np.maximum(_df['High']-_df['Low'], np.maximum((_df['High']-_df['Close'].shift(1)).abs(),(_df['Low']-_df['Close'].shift(1)).abs()))
    atr = tr.rolling(14).mean()
    _high = _df['High'].values; _low = _df['Low'].values; _close = _df['Close'].values
    _ema = ema20.values; _ema_s = ema50.values; _atr_v = atr.values
    _times = _df.index.time; _n = len(_df)

def run_config(args):
    _init()
    touch_tol, touch_below, lock1_rr, lock2_rr, l1p, l2p, runner_trail, trail_buf = args

    equity = CAPITAL; wins = 0; losses = 0; gross_won = 0; gross_lost = 0; total_trades = 0
    bar = 50
    while bar < _n - MAX_FWD - 5:
        a = _atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(_ema[bar]) or np.isnan(_ema_s[bar]):
            bar += 1; continue
        if _times[bar] >= dt.time(15, 30): bar += 1; continue

        trend = 0
        if _close[bar] > _ema[bar] and _ema[bar] > _ema_s[bar]: trend = 1
        elif _close[bar] < _ema[bar] and _ema[bar] < _ema_s[bar]: trend = -1
        if trend == 0: bar += 1; continue

        tol = a * touch_tol
        if trend == 1:
            is_touch = (_low[bar] <= _ema[bar] + tol) and (_low[bar] >= _ema[bar] - a * touch_below)
        else:
            is_touch = (_high[bar] >= _ema[bar] - tol) and (_high[bar] <= _ema[bar] + a * touch_below)
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

        entry_comm = shares * COMM
        l1_shares = max(1, int(shares * l1p)); l2_shares = max(1, int(shares * l2p))
        runner_stop = stop; l1_done = False; l2_done = False
        trade_pnl = -entry_comm; remaining = shares; end_bar = entry_bar

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
            if not l1_done:
                if (trend == 1 and h >= sig + lock1_rr * risk) or (trend == -1 and l <= sig - lock1_rr * risk):
                    trade_pnl += l1_shares * lock1_rr * risk - l1_shares * COMM
                    remaining -= l1_shares; l1_done = True
                    if trend == 1: runner_stop = max(runner_stop, sig)
                    else: runner_stop = min(runner_stop, sig)
            if l1_done and not l2_done:
                if (trend == 1 and h >= sig + lock2_rr * risk) or (trend == -1 and l <= sig - lock2_rr * risk):
                    trade_pnl += l2_shares * lock2_rr * risk - l2_shares * COMM
                    remaining -= l2_shares; l2_done = True
            if l1_done and k >= 5:
                sk = max(1, k - runner_trail + 1)
                if trend == 1:
                    rl = min(_low[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                    runner_stop = max(runner_stop, rl - trail_buf * a)
                else:
                    rh = max(_high[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                    runner_stop = min(runner_stop, rh + trail_buf * a)
        else:
            ep = _close[min(entry_bar + MAX_FWD, _n-1)]
            trade_pnl += remaining * (ep - sig) * trend - remaining * COMM
            end_bar = min(entry_bar + MAX_FWD, _n-1)

        equity += trade_pnl; total_trades += 1
        if trade_pnl > 0: wins += 1; gross_won += trade_pnl
        else: losses += 1; gross_lost += abs(trade_pnl)
        bar = end_bar + 1

    pf = gross_won / gross_lost if gross_lost > 0 else 0
    ret = (equity - CAPITAL) / CAPITAL * 100
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    return (*args, round(pf, 3), round(ret, 2), total_trades, round(wr, 1))


def main():
    configs = []
    # Touch tolerance
    for tt in [0.05, 0.10, 0.15, 0.20, 0.30]:
        configs.append((tt, 0.5, 0.5, 1.5, 0.30, 0.20, 10, 0.3))
    # Lock1 R:R
    for l1 in [0.3, 0.5, 0.75, 1.0, 1.5]:
        configs.append((0.15, 0.5, l1, 2.0, 0.30, 0.20, 10, 0.3))
    # Lock2 R:R
    for l2 in [1.0, 1.5, 2.0, 2.5, 3.0]:
        configs.append((0.15, 0.5, 0.5, l2, 0.30, 0.20, 10, 0.3))
    # Portions
    for l1p, l2p in [(0.20,0.20),(0.30,0.20),(0.30,0.30),(0.40,0.20),(0.40,0.30),(0.50,0.20)]:
        configs.append((0.15, 0.5, 0.5, 1.5, l1p, l2p, 10, 0.3))
    # Trail length
    for rt in [5, 10, 15, 20, 30]:
        configs.append((0.15, 0.5, 0.5, 1.5, 0.30, 0.20, rt, 0.3))
    # Trail buffer
    for tb in [0.1, 0.2, 0.3, 0.5, 0.8]:
        configs.append((0.15, 0.5, 0.5, 1.5, 0.30, 0.20, 10, tb))
    # Touch below max
    for tbelow in [0.3, 0.5, 0.8, 1.0]:
        configs.append((0.15, tbelow, 0.5, 1.5, 0.30, 0.20, 10, 0.3))
    # Best combos
    for tt in [0.05, 0.10, 0.15]:
        for l1 in [0.5, 0.75]:
            for l2 in [1.5, 2.0, 2.5]:
                for tb in [0.2, 0.3]:
                    configs.append((tt, 0.5, l1, l2, 0.30, 0.20, 10, tb))

    print(f"Sweeping {len(configs)} configs on {cpu_count()} CPUs...")

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        results = list(pool.map(run_config, configs))

    results.sort(key=lambda x: x[8], reverse=True)  # sort by PF

    print(f"\n{'='*100}")
    print(f"TOP 20 BY PF")
    print(f"{'='*100}")
    print(f"{'TchTol':<8}{'Below':<7}{'L1RR':<7}{'L2RR':<7}{'L1%':<6}{'L2%':<6}{'Trail':<7}{'TBuf':<7}{'PF':<8}{'Ret%':<9}{'Trades':<8}{'WR%':<7}")
    print("-" * 82)
    for r in results[:20]:
        print(f"{r[0]:<8}{r[1]:<7}{r[2]:<7}{r[3]:<7}{r[4]:<6.0%}{r[5]:<6.0%}{r[6]:<7}{r[7]:<7}{r[8]:<8}{r[9]:<8}% {r[10]:<8}{r[11]:<6}%")

    print(f"\n  SENSITIVITY:")
    # Group by single param changes from baseline (0.15, 0.5, 0.5, 1.5, 0.30, 0.20, 10, 0.3)
    base = (0.15, 0.5, 0.5, 1.5, 0.30, 0.20, 10, 0.3)
    for name, idx in [("Touch Tol",0),("Lock1 RR",2),("Lock2 RR",3),("Trail Bars",6),("Trail Buf",7)]:
        group = [(r[idx], r[8], r[9], r[10]) for r in results
                 if all(r[i]==base[i] for i in range(8) if i != idx)]
        if group:
            group.sort(key=lambda x: x[0])
            print(f"\n  {name}:")
            for val, pf, ret, trades in group:
                print(f"    {val:<8} → PF={pf:<8} Ret={ret}% Trades={trades}")


if __name__ == "__main__":
    main()
