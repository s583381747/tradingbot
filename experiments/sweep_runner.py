"""
Sweep runner trail methods. Tests 7 different trail architectures.
Uses all CPUs.
"""
import pandas as pd, numpy as np, functools, datetime as dt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
COMM = 0.005; CAPITAL = 100000; RISK_PCT = 0.01; MAX_FWD = 180
TOUCH_TOL = 0.15; TOUCH_BELOW = 0.5; SIGNAL_OFFSET = 0.05; STOP_BUFFER = 0.3
LOCK_RR = 0.3; LOCK_PCT = 0.20

_high = None; _low = None; _close = None
_ema = None; _ema_s = None; _atr_v = None; _times = None; _n = 0
_ema_arr = None  # raw ema20 array for EMA-based trails

def _init():
    global _high, _low, _close, _ema, _ema_s, _atr_v, _times, _n, _ema_arr
    if _high is not None: return
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    tr = np.maximum(df['High']-df['Low'], np.maximum(
        (df['High']-df['Close'].shift(1)).abs(),(df['Low']-df['Close'].shift(1)).abs()))
    atr = tr.rolling(14).mean()
    _high = df['High'].values; _low = df['Low'].values; _close = df['Close'].values
    _ema = ema20.values; _ema_s = ema50.values; _atr_v = atr.values
    _ema_arr = ema20.values
    _times = df.index.time; _n = len(df)


def run_config(args):
    """
    args = (trail_method, trail_params, trail_activate_rr)
    trail_method: str
    trail_params: dict of method-specific params
    trail_activate_rr: float — don't start trailing until runner profit >= this R:R (0=immediate from BE)
    """
    _init()
    trail_method, trail_params, activate_rr = args

    equity = CAPITAL; wins = 0; losses = 0; gross_won = 0; gross_lost = 0
    total_trades = 0; be_exits = 0; runner_pnl_sum = 0; mfe_capture_sum = 0; mfe_count = 0
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

        lock_shares = max(1, int(shares * LOCK_PCT))
        runner_stop = stop; lock_done = False
        trade_pnl = -shares * COMM; remaining = shares; end_bar = entry_bar
        max_favorable = 0; highest_since = sig; lowest_since = sig
        runner_activated = False

        for k in range(1, MAX_FWD + 1):
            bi = entry_bar + k
            if bi >= _n: break
            h = _high[bi]; l = _low[bi]; c = _close[bi]
            ea = _ema_arr[bi] if bi < _n else sig
            cur_atr = _atr_v[bi] if bi < _n and not np.isnan(_atr_v[bi]) else a

            # Track MFE and extremes
            if trend == 1:
                highest_since = max(highest_since, h)
                mfe = h - sig
            else:
                lowest_since = min(lowest_since, l)
                mfe = sig - l
            max_favorable = max(max_favorable, mfe)

            # Session close
            if _times[bi] >= dt.time(15, 58):
                trade_pnl += remaining * (c - sig) * trend - remaining * COMM
                end_bar = bi; break

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * COMM
                end_bar = bi; break

            # Lock
            if not lock_done:
                if (trend == 1 and h >= sig + LOCK_RR * risk) or (trend == -1 and l <= sig - LOCK_RR * risk):
                    trade_pnl += lock_shares * LOCK_RR * risk - lock_shares * COMM
                    remaining -= lock_shares; lock_done = True
                    if trend == 1: runner_stop = max(runner_stop, sig)
                    else: runner_stop = min(runner_stop, sig)

            if not lock_done:
                continue

            # Check trail activation
            runner_profit_rr = ((c - sig) * trend) / risk if risk > 0 else 0
            if not runner_activated:
                if runner_profit_rr >= activate_rr:
                    runner_activated = True
                    # Reset tracking from activation point
                    highest_since = h if trend == 1 else highest_since
                    lowest_since = l if trend == -1 else lowest_since
                else:
                    continue

            # ═══ TRAIL METHODS ═══
            new_stop = runner_stop

            if trail_method == "trailing_low":
                # Current method: N-bar trailing low - buffer*ATR
                lb = trail_params.get("bars", 10)
                buf = trail_params.get("buffer", 0.3)
                if k >= lb:
                    sk = max(1, k - lb + 1)
                    if trend == 1:
                        rl = min(_low[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                        new_stop = rl - buf * a
                    else:
                        rh = max(_high[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                        new_stop = rh + buf * a

            elif trail_method == "ema_trail":
                # Trail along EMA20 - N*ATR
                mult = trail_params.get("mult", 1.0)
                if trend == 1:
                    new_stop = ea - mult * cur_atr
                else:
                    new_stop = ea + mult * cur_atr

            elif trail_method == "chandelier":
                # Highest high(N) - M*ATR
                lb = trail_params.get("bars", 15)
                mult = trail_params.get("mult", 2.0)
                if k >= lb:
                    sk = max(1, k - lb + 1)
                    if trend == 1:
                        hh = max(_high[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                        new_stop = hh - mult * cur_atr
                    else:
                        ll = min(_low[entry_bar+kk] for kk in range(sk, k+1) if entry_bar+kk < _n)
                        new_stop = ll + mult * cur_atr

            elif trail_method == "staircase_rr":
                # Step stop up by R:R levels
                step = trail_params.get("step", 1.0)
                current_rr = ((c - sig) * trend) / risk if risk > 0 else 0
                levels_passed = int(current_rr / step)
                if levels_passed >= 1:
                    lock_level = (levels_passed - 1) * step * risk
                    if trend == 1:
                        new_stop = sig + lock_level
                    else:
                        new_stop = sig - lock_level

            elif trail_method == "ema_cross":
                # Close when close crosses below EMA20
                if trend == 1 and c < ea:
                    trade_pnl += remaining * (c - sig) * trend - remaining * COMM
                    end_bar = bi; break
                elif trend == -1 and c > ea:
                    trade_pnl += remaining * (c - sig) * trend - remaining * COMM
                    end_bar = bi; break
                continue  # no stop update needed

            elif trail_method == "hybrid_ema":
                # EMA trail but floored at BE
                mult = trail_params.get("mult", 1.0)
                if trend == 1:
                    ema_level = ea - mult * cur_atr
                    new_stop = max(sig, ema_level)  # never below BE
                else:
                    ema_level = ea + mult * cur_atr
                    new_stop = min(sig, ema_level)

            elif trail_method == "time_ratchet":
                # Every N bars, tighten by 0.5R
                interval = trail_params.get("interval", 10)
                steps = k // interval
                if steps >= 1:
                    if trend == 1:
                        new_stop = sig + (steps - 1) * 0.5 * risk
                    else:
                        new_stop = sig - (steps - 1) * 0.5 * risk

            # Ratchet: only move stop in favorable direction
            if trend == 1:
                runner_stop = max(runner_stop, new_stop)
            else:
                runner_stop = min(runner_stop, new_stop)

        else:
            # Timeout
            ep = _close[min(entry_bar + MAX_FWD, _n-1)]
            trade_pnl += remaining * (ep - sig) * trend - remaining * COMM
            end_bar = min(entry_bar + MAX_FWD, _n-1)

        # Stats
        equity += trade_pnl; total_trades += 1
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_dd = max(max_dd, dd)
        if trade_pnl > 0: wins += 1; gross_won += trade_pnl
        else: losses += 1; gross_lost += abs(trade_pnl)

        # Runner analysis
        if lock_done:
            runner_exit_pnl = (runner_stop - sig) * trend * remaining / shares  # normalized
            runner_pnl_sum += runner_exit_pnl
            if max_favorable > 0:
                capture = runner_exit_pnl / max_favorable * 100
                mfe_capture_sum += max(0, capture)
            mfe_count += 1
            if runner_exit_pnl <= 0.001:
                be_exits += 1

        bar = end_bar + 1

    if total_trades < 100: return None
    pf = gross_won / gross_lost if gross_lost > 0 else 0
    ret = (equity - CAPITAL) / CAPITAL * 100
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    be_rate = be_exits / mfe_count * 100 if mfe_count > 0 else 0
    avg_capture = mfe_capture_sum / mfe_count if mfe_count > 0 else 0
    avg_runner = runner_pnl_sum / mfe_count if mfe_count > 0 else 0

    return {
        'method': trail_method, 'params': trail_params, 'activate': activate_rr,
        'pf': round(pf, 3), 'ret': round(ret, 2), 'trades': total_trades,
        'wr': round(wr, 1), 'max_dd': round(max_dd, 2),
        'be_rate': round(be_rate, 1), 'capture': round(avg_capture, 1),
        'runner_pnl': round(avg_runner, 4),
    }


def main():
    configs = []

    # ═══ 1. Current baseline ═══
    for bars in [5, 10, 15, 20]:
        for buf in [0.2, 0.3, 0.5]:
            configs.append(("trailing_low", {"bars": bars, "buffer": buf}, 0))

    # ═══ 2. EMA20 trail ═══
    for mult in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        configs.append(("ema_trail", {"mult": mult}, 0))

    # ═══ 3. Chandelier ═══
    for bars in [10, 15, 20, 30]:
        for mult in [1.5, 2.0, 2.5, 3.0]:
            configs.append(("chandelier", {"bars": bars, "mult": mult}, 0))

    # ═══ 4. Staircase R:R ═══
    for step in [0.5, 1.0, 1.5, 2.0]:
        configs.append(("staircase_rr", {"step": step}, 0))

    # ═══ 5. EMA cross ═══
    configs.append(("ema_cross", {}, 0))

    # ═══ 6. Hybrid EMA (floored at BE) ═══
    for mult in [0.3, 0.5, 0.8, 1.0, 1.5]:
        configs.append(("hybrid_ema", {"mult": mult}, 0))

    # ═══ 7. Time ratchet ═══
    for interval in [5, 10, 15, 20]:
        configs.append(("time_ratchet", {"interval": interval}, 0))

    # ═══ 8. Activation delay variants (best methods from above) ═══
    for method, params in [
        ("ema_trail", {"mult": 0.5}),
        ("ema_trail", {"mult": 1.0}),
        ("chandelier", {"bars": 15, "mult": 2.0}),
        ("hybrid_ema", {"mult": 0.5}),
        ("staircase_rr", {"step": 1.0}),
    ]:
        for act in [0, 0.5, 1.0, 1.5, 2.0]:
            configs.append((method, params, act))

    print(f"Sweeping {len(configs)} trail configs on {cpu_count()} CPUs...\n")

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        results = [r for r in pool.map(run_config, configs) if r is not None]

    results.sort(key=lambda x: x['pf'], reverse=True)

    # Group by method
    methods = sorted(set(r['method'] for r in results))
    for m in methods:
        group = [r for r in results if r['method'] == m]
        best = group[0]
        print(f"\n{'='*95}")
        print(f"{m.upper()} — TOP 5")
        print(f"{'='*95}")
        print(f"{'Params':<30}{'Act':<6}{'PF':<8}{'Ret%':<8}{'WR%':<7}{'DD%':<7}{'BE%':<7}{'Cap%':<7}{'RunPnL':<9}")
        print("-" * 89)
        for r in group[:5]:
            print(f"{str(r['params']):<30}{r['activate']:<6}{r['pf']:<8}{r['ret']:<7}% "
                  f"{r['wr']:<6}% {r['max_dd']:<6}% {r['be_rate']:<6}% {r['capture']:<6}% ${r['runner_pnl']:<8}")

    # Overall top 15
    print(f"\n{'='*95}")
    print(f"OVERALL TOP 15")
    print(f"{'='*95}")
    print(f"{'Method':<16}{'Params':<28}{'Act':<6}{'PF':<8}{'Ret%':<8}{'WR%':<7}{'DD%':<7}{'BE%':<7}{'Cap%':<7}")
    print("-" * 94)
    for r in results[:15]:
        print(f"{r['method']:<16}{str(r['params']):<28}{r['activate']:<6}{r['pf']:<8}"
              f"{r['ret']:<7}% {r['wr']:<6}% {r['max_dd']:<6}% {r['be_rate']:<6}% {r['capture']:<6}%")

    # Method summary
    print(f"\n{'='*95}")
    print("METHOD SUMMARY (best of each)")
    print("="*95)
    for m in methods:
        group = [r for r in results if r['method'] == m]
        best = max(group, key=lambda x: x['pf'])
        print(f"  {m:<16} PF={best['pf']:<8} Ret={best['ret']:<7}% BE_rate={best['be_rate']:<6}% "
              f"Capture={best['capture']:<6}% DD={best['max_dd']}%")


if __name__ == "__main__":
    main()
