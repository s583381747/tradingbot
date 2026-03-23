"""Walk-forward validation: train/test split, quarterly, bootstrap. All pandas."""
import pandas as pd, numpy as np, functools, datetime as dt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from entry_signal import add_indicators

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
COMM = 0.005; CAPITAL = 100000; RISK_PCT = 0.01; MAX_FWD = 180
# Plan F parameters (must match strategy_final.py)
TOUCH_TOL = 0.15; TOUCH_BELOW = 0.5; SIGNAL_OFFSET = 0.05; STOP_BUFFER = 0.3
LOCK1_RR = 0.3; LOCK1_PCT = 0.20
CHAND_BARS = 40; CHAND_MULT = 1.0
DAILY_R_LIMIT = 2.5


def run_period(args):
    """Run strategy on a date-filtered slice of data."""
    start_date, end_date, label = args

    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    mask = (df.index >= start_date) & (df.index < end_date)
    df = df[mask]
    if len(df) < 500:
        return (label, 0, 0, 0, 0, 0, 0, 0)

    df = add_indicators(df)

    high = df['High'].values; low = df['Low'].values; close = df['Close'].values
    ema = df['ema20'].values; ema_s = df['ema50'].values; atr_v = df['atr'].values
    times = df.index.time; dates = df.index.date; n = len(df)
    days = df.index.normalize().nunique()

    equity = CAPITAL; wins = 0; losses = 0; gross_won = 0; gross_lost = 0
    total_trades = 0; bar = 50
    daily_r_loss = 0.0; current_date = None

    while bar < n - MAX_FWD - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= dt.time(15, 30): bar += 1; continue

        # Daily R limit
        d_date = dates[bar]
        if current_date != d_date:
            current_date = d_date; daily_r_loss = 0.0
        if daily_r_loss >= DAILY_R_LIMIT: bar += 1; continue

        trend = 0
        if close[bar] > ema[bar] and ema[bar] > ema_s[bar]: trend = 1
        elif close[bar] < ema[bar] and ema[bar] < ema_s[bar]: trend = -1
        if trend == 0: bar += 1; continue

        tol = a * TOUCH_TOL
        if trend == 1:
            is_touch = (low[bar] <= ema[bar] + tol) and (low[bar] >= ema[bar] - a * TOUCH_BELOW)
        else:
            is_touch = (high[bar] >= ema[bar] - tol) and (high[bar] <= ema[bar] + a * TOUCH_BELOW)
        if not is_touch: bar += 1; continue

        bb = bar + 1
        if bb >= n: bar += 1; continue
        if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
        if trend == -1 and close[bb] >= low[bar]: bar += 1; continue

        if trend == 1: sig = high[bar] + SIGNAL_OFFSET; stop = low[bar] - STOP_BUFFER * a
        else: sig = low[bar] - SIGNAL_OFFSET; stop = high[bar] + STOP_BUFFER * a
        risk = abs(sig - stop)
        if risk <= 0: bar += 1; continue

        triggered = False; entry_bar = -1
        for j in range(1, 4):
            cb = bb + j
            if cb >= n: break
            if trend == 1 and high[cb] >= sig: triggered = True; entry_bar = cb; break
            elif trend == -1 and low[cb] <= sig: triggered = True; entry_bar = cb; break
        if not triggered: bar += 1; continue

        shares = max(1, int(equity * RISK_PCT / risk))
        if shares * sig > equity * 0.25: shares = max(1, int(equity * 0.25 / sig))
        if equity < shares * risk or shares < 1: bar += 1; continue

        # Bug #1: entry bar stop check
        if trend == 1 and low[entry_bar] <= stop:
            loss = shares * (stop - sig) - shares * COMM * 2
            equity += loss; total_trades += 1; losses += 1; gross_lost += abs(loss)
            if shares * risk > 0: daily_r_loss += abs(loss) / (shares * risk)
            bar = entry_bar + 1; continue
        if trend == -1 and high[entry_bar] >= stop:
            loss = shares * (sig - stop) * -1 - shares * COMM * 2
            equity += loss; total_trades += 1; losses += 1; gross_lost += abs(loss)
            if shares * risk > 0: daily_r_loss += abs(loss) / (shares * risk)
            bar = entry_bar + 1; continue

        lock_shares = max(1, int(shares * LOCK1_PCT))
        runner_stop = stop; lock_done = False
        trade_pnl = -shares * COMM; remaining = shares; end_bar = entry_bar

        for k in range(1, MAX_FWD + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            cur_atr = atr_v[bi] if bi < n and not np.isnan(atr_v[bi]) else a

            if times[bi] >= dt.time(15, 58):
                ep = close[bi]; trade_pnl += remaining * (ep - sig) * trend - remaining * COMM
                end_bar = bi; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)

            # Bug #2: same-bar stop/lock collision — stop wins
            lock_hit_bar = False
            if not lock_done:
                lock_hit_bar = (trend == 1 and h >= sig + LOCK1_RR * risk) or \
                               (trend == -1 and l <= sig - LOCK1_RR * risk)
            if stopped and lock_hit_bar and not lock_done:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * COMM
                end_bar = bi; break

            if stopped:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * COMM
                end_bar = bi; break

            # Lock: 20% at 0.3R → move stop to BE
            if lock_hit_bar and not lock_done:
                trade_pnl += lock_shares * LOCK1_RR * risk - lock_shares * COMM
                remaining -= lock_shares; lock_done = True
                if trend == 1: runner_stop = max(runner_stop, sig)
                else: runner_stop = min(runner_stop, sig)

            # Chandelier trail (Bug #3: exclude current bar)
            if lock_done and k >= CHAND_BARS:
                sk = max(1, k - CHAND_BARS + 1)
                if trend == 1:
                    hh = max(high[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n)
                    runner_stop = max(runner_stop, hh - CHAND_MULT * cur_atr)
                else:
                    ll = min(low[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n)
                    runner_stop = min(runner_stop, ll + CHAND_MULT * cur_atr)
        else:
            ep = close[min(entry_bar + MAX_FWD, n-1)]
            trade_pnl += remaining * (ep - sig) * trend - remaining * COMM
            end_bar = min(entry_bar + MAX_FWD, n-1)

        equity += trade_pnl; total_trades += 1
        if trade_pnl > 0: wins += 1; gross_won += trade_pnl
        else: losses += 1; gross_lost += abs(trade_pnl)
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(trade_pnl) / (shares * risk)
        bar = end_bar + 1

    pf = gross_won / gross_lost if gross_lost > 0 else 0
    ret = (equity - CAPITAL) / CAPITAL * 100
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    tpd = total_trades / max(days, 1)
    return (label, total_trades, round(tpd, 1), wins, losses, round(wr, 1), round(pf, 3), round(ret, 2))


def main():
    print("=" * 80)
    print("WALK-FORWARD VALIDATION (pandas, IBKR commission)")
    print("=" * 80)

    tasks = []

    # 1. Full period
    tasks.append(("2024-03-22", "2026-03-21", "FULL 2Y"))

    # 2. Walk-forward: Y1 vs Y2
    tasks.append(("2024-03-22", "2025-03-22", "Y1 (train)"))
    tasks.append(("2025-03-22", "2026-03-21", "Y2 (test)"))

    # 3. Rolling 6-month windows
    for start_y, start_m in [(2024,3),(2024,6),(2024,9),(2024,12),(2025,3),(2025,6),(2025,9)]:
        end_m = start_m + 6
        end_y = start_y
        if end_m > 12: end_m -= 12; end_y += 1
        s = f"{start_y}-{start_m:02d}-01"
        e = f"{end_y}-{end_m:02d}-01"
        tasks.append((s, e, f"6M {start_y}-{start_m:02d}"))

    # 4. Quarterly
    for year in [2024, 2025, 2026]:
        for q in range(1, 5):
            ms = (q-1)*3+1; me = q*3+1; ye = year
            if me > 12: me = 1; ye = year+1
            s = f"{year}-{ms:02d}-01"; e = f"{ye}-{me:02d}-01"
            tasks.append((s, e, f"{year}Q{q}"))

    print(f"\nRunning {len(tasks)} periods on {cpu_count()} CPUs...\n")

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        results = list(pool.map(run_period, tasks))

    # Print results grouped
    print(f"{'Period':<20} {'Trades':<8} {'T/day':<7} {'W/L':<10} {'WR%':<7} {'PF':<8} {'Return%':<10}")
    print("-" * 70)

    sections = [
        ("FULL", [r for r in results if "FULL" in r[0]]),
        ("WALK-FORWARD", [r for r in results if r[0] in ("Y1 (train)", "Y2 (test)")]),
        ("6-MONTH ROLLING", [r for r in results if "6M" in r[0]]),
        ("QUARTERLY", [r for r in results if "Q" in r[0] and "6M" not in r[0] and "FULL" not in r[0]]),
    ]

    for section_name, section_results in sections:
        print(f"\n  {section_name}:")
        profitable = 0; total = 0
        for label, trades, tpd, w, l, wr, pf, ret in sorted(section_results, key=lambda x: x[0]):
            if trades < 1: continue
            total += 1
            mark = "+" if pf >= 1.0 else "-"
            if pf >= 1.0: profitable += 1
            print(f"  {mark} {label:<18} {trades:<8} {tpd:<7} {w}/{l:<7} {wr:<6}% {pf:<8} {ret:>+8}%")
        if total > 0:
            print(f"  → Profitable: {profitable}/{total} ({profitable/total*100:.0f}%)")

    # 5. Bootstrap significance on full period
    print(f"\n{'='*80}")
    print("BOOTSTRAP SIGNIFICANCE TEST")
    print("="*80)

    full = [r for r in results if r[0] == "FULL 2Y"][0]
    _, trades, _, w, l, wr_pct, obs_pf, _ = full

    if w > 0 and l > 0:
        # Reconstruct trade PnLs approximately
        avg_win = full[7] / 100 * CAPITAL / w if w > 0 else 0  # rough approximation
        # Actually just use PF and WR to estimate
        # PF = (WR * AvgW) / ((1-WR) * AvgL)
        # We know PF and WR, solve for AvgW/AvgL ratio
        wr = wr_pct / 100
        rr_ratio = obs_pf * (1 - wr) / wr  # AvgW / AvgL

        # Simulate: given WR and R:R, is PF > 1.0 significant?
        np.random.seed(42)
        mc_pfs = []
        for _ in range(10000):
            # Random shuffle of win/loss labels
            outcomes = np.random.random(trades) < wr
            wins_mc = outcomes.sum()
            losses_mc = trades - wins_mc
            if losses_mc > 0 and wins_mc > 0:
                mc_pf = (wins_mc * rr_ratio) / losses_mc
            else:
                mc_pf = 0
            mc_pfs.append(mc_pf)

        mc_pfs = np.array(mc_pfs)
        p_value = (mc_pfs >= obs_pf).mean()
        print(f"  Observed PF: {obs_pf:.3f}")
        print(f"  WR: {wr_pct:.1f}%  R:R ratio: {rr_ratio:.2f}")
        print(f"  MC mean PF: {mc_pfs.mean():.3f}  MC 95th: {np.percentile(mc_pfs, 95):.3f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05: print("  → SIGNIFICANT (p<0.05)")
        elif p_value < 0.10: print("  → MARGINAL (p<0.10)")
        else: print("  → NOT SIGNIFICANT")


if __name__ == "__main__":
    main()
