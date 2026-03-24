"""
Live Trading Simulator — replay 1-min bars as if real-time.

Adds realistic execution constraints:
  1. 1-bar latency: signal on bar N → order fills on bar N+1
  2. Random slippage: $0.01-0.03 per share on each fill
  3. Partial fills possible (simulated)
  4. No look-ahead: decisions use only past data
  5. Real-time P&L tracking with equity curve

Usage: python3 live_sim.py [--days 20] [--slippage 0.02]
"""
from __future__ import annotations
import argparse, functools, datetime as dt, sys
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"


class LiveSimulator:
    """Simulates live bar-by-bar execution with realistic constraints."""

    def __init__(self, slippage_mean=0.015, slippage_std=0.01, latency_bars=1,
                 commission=0.005, capital=100000, risk_pct=0.01):
        self.slip_mean = slippage_mean
        self.slip_std = slippage_std
        self.latency = latency_bars
        self.comm = commission
        self.capital = capital
        self.equity = capital
        self.risk_pct = risk_pct

        # Strategy params (from strategy_final.py)
        self.ema_fast = 20
        self.ema_slow = 50
        self.atr_period = 14
        self.touch_tol = 0.15
        self.touch_below = 0.5
        self.signal_offset = 0.05
        self.stop_buffer = 0.3
        self.lock_rr = 0.3
        self.lock_pct = 0.20
        self.chand_bars = 40         # Plan G
        self.chand_mult = 0.5         # Plan G (tighter trail)

        # State
        self.position = 0          # shares held (+ long, - short)
        self.entry_price = 0
        self.stop_price = 0
        self.runner_stop = 0
        self.risk = 0
        self.lock_done = False
        self.lock_shares = 0
        self.entry_bar_idx = 0
        self.direction = 0

        # Pending orders
        self.pending_entry = None   # (dir, price, stop, risk, size, bar_placed)
        self.pending_lock = None    # (size, target_price, bar_placed)

        # Touch/bounce state
        self.touch_bar = -1
        self.touch_high = 0
        self.touch_low = 0
        self.touch_atr = 0
        self.touch_dir = 0

        # Chandelier lookback buffer
        self._high_buf = []
        self._low_buf = []

        # Tracking
        self.trades = []
        self.equity_curve = []
        self.bar_count = 0

    def _slippage(self):
        """Random slippage per share."""
        return max(0, np.random.normal(self.slip_mean, self.slip_std))

    def process_bar(self, bar_idx, o, h, l, c, vol, t, ema20, ema50, atr_val):
        """Process one bar. All decisions based on COMPLETED bars only."""
        self.bar_count = bar_idx

        # ═══ 1. Force close at session end ═══
        if t >= dt.time(15, 58):
            if self.position != 0:
                self._close_position(c, "session_close")
            self.pending_entry = None
            return

        # ═══ 2. Check pending entry fill ═══
        if self.pending_entry is not None:
            d, price, stop, risk, size, placed_bar = self.pending_entry
            # Can only fill after latency
            if bar_idx - placed_bar >= self.latency:
                # Check if this bar's range includes the signal price
                if d == 1 and h >= price:
                    slip = self._slippage()
                    fill_price = price + slip  # worse for long
                    self._open_position(d, fill_price, stop, risk, size, bar_idx)
                    self.pending_entry = None
                elif d == -1 and l <= price:
                    slip = self._slippage()
                    fill_price = price - slip  # worse for short
                    self._open_position(d, fill_price, stop, risk, size, bar_idx)
                    self.pending_entry = None
                elif bar_idx - placed_bar > 3 + self.latency:
                    self.pending_entry = None  # expired

        # ═══ 3. Manage position ═══
        if self.position != 0:
            self._manage(bar_idx, h, l, c, atr_val)

        # ═══ 4. Scan for new signals ═══
        # Always scan (track touches even when position open or pending)
        if t < dt.time(15, 30) and atr_val > 0:
            self._scan(bar_idx, o, h, l, c, ema20, ema50, atr_val)

        # Track equity
        mark_to_market = self.equity
        if self.position != 0:
            unrealized = self.position * (c - self.entry_price) * self.direction
            mark_to_market = self.equity + unrealized
        self.equity_curve.append({"bar": bar_idx, "equity": mark_to_market})

    def _scan(self, bar_idx, o, h, l, c, ema20, ema50, atr):
        """Scan for touch → bounce → signal. Uses only completed bar data."""

        # Check bounce FIRST — uses touch bar's direction, not current trend
        if self.touch_bar >= 0 and bar_idx - self.touch_bar == 1:
            d = self.touch_dir
            bounced = (d == 1 and c > self.touch_high) or (d == -1 and c < self.touch_low)
            if bounced and self.pending_entry is None and self.position == 0:
                # Place signal order (will fill next bar due to latency)
                if d == 1:
                    sig = self.touch_high + self.signal_offset
                    stop = self.touch_low - self.stop_buffer * self.touch_atr
                else:
                    sig = self.touch_low - self.signal_offset
                    stop = self.touch_high + self.stop_buffer * self.touch_atr
                risk = abs(sig - stop)
                if risk > 0:
                    size = max(1, int(self.equity * self.risk_pct / risk))
                    max_size = max(1, int(self.equity * 0.25 / abs(sig)))
                    size = min(size, max_size)
                    if size >= 1:
                        self.pending_entry = (d, sig, stop, risk, size, bar_idx)

        # Trend for new touch detection
        trend = 0
        if c > ema20 and ema20 > ema50: trend = 1
        elif c < ema20 and ema20 < ema50: trend = -1
        if trend == 0:
            self.touch_bar = -1
            return

        tol = atr * self.touch_tol

        # Check touch on THIS bar
        if trend == 1:
            is_touch = (l <= ema20 + tol) and (l >= ema20 - atr * self.touch_below)
        else:
            is_touch = (h >= ema20 - tol) and (h <= ema20 + atr * self.touch_below)

        if is_touch:
            self.touch_bar = bar_idx
            self.touch_high = h
            self.touch_low = l
            self.touch_atr = atr
            self.touch_dir = trend

    def _open_position(self, d, fill_price, stop, risk, size, bar_idx):
        self.position = size
        self.direction = d
        self.entry_price = fill_price
        self.stop_price = stop
        self.runner_stop = stop
        self.risk = risk
        self.lock_done = False
        self.lock_shares = max(1, int(size * self.lock_pct))
        self.entry_bar_idx = bar_idx
        self._high_buf = []
        self._low_buf = []
        # Deduct commission
        self.equity -= size * self.comm

    def _manage(self, bar_idx, h, l, c, cur_atr):
        d = self.direction

        # Stop check
        active_stop = self.runner_stop if self.lock_done else self.stop_price
        if (d == 1 and l <= active_stop) or (d == -1 and h >= active_stop):
            slip = self._slippage()
            exit_price = active_stop - slip * d  # worse
            self._close_position(exit_price, "stop" if not self.lock_done else "chandelier")
            return

        # Lock
        if not self.lock_done:
            target = self.entry_price + self.lock_rr * self.risk * d
            if (d == 1 and h >= target) or (d == -1 and l <= target):
                # Exit lock portion with slippage
                slip = self._slippage()
                lock_exit = target - slip * d  # conservative
                lock_pnl = self.lock_shares * (lock_exit - self.entry_price) * d
                self.equity += lock_pnl - self.lock_shares * self.comm
                self.position -= self.lock_shares
                self.lock_done = True
                # Move stop to BE
                if d == 1: self.runner_stop = max(self.runner_stop, self.entry_price)
                else: self.runner_stop = min(self.runner_stop, self.entry_price)

        # Chandelier trail — use lookback buffer for proper highest_high
        self._high_buf.append(h)
        self._low_buf.append(l)
        bars_held = bar_idx - self.entry_bar_idx
        if self.lock_done and bars_held >= self.chand_bars:
            lookback = self._high_buf[-self.chand_bars:-1] if len(self._high_buf) > 1 else self._high_buf
            if d == 1 and lookback:
                hh = max(lookback)
                new_stop = hh - self.chand_mult * cur_atr
                self.runner_stop = max(self.runner_stop, new_stop)
            elif d == -1:
                lb = self._low_buf[-self.chand_bars:-1] if len(self._low_buf) > 1 else self._low_buf
                if lb:
                    ll = min(lb)
                    new_stop = ll + self.chand_mult * cur_atr
                    self.runner_stop = min(self.runner_stop, new_stop)

    def _close_position(self, exit_price, reason):
        d = self.direction
        remaining = self.position
        pnl = remaining * (exit_price - self.entry_price) * d
        self.equity += pnl - remaining * self.comm

        self.trades.append({
            "entry": self.entry_price,
            "exit": exit_price,
            "pnl": pnl - remaining * self.comm * 2,  # round trip
            "direction": "LONG" if d == 1 else "SHORT",
            "reason": reason,
            "bars_held": self.bar_count - self.entry_bar_idx,
            "lock_hit": self.lock_done,
            "equity": self.equity,
        })

        self.position = 0
        self.direction = 0


def run_simulation(df, sim_days=None, **kwargs):
    """Run live simulation on dataframe."""
    sim = LiveSimulator(**kwargs)

    # Compute indicators using EXPANDING window (no look-ahead)
    ema20 = df["Close"].ewm(span=sim.ema_fast, adjust=False).mean().values
    ema50 = df["Close"].ewm(span=sim.ema_slow, adjust=False).mean().values
    tr = np.maximum(df["High"]-df["Low"], np.maximum(
        (df["High"]-df["Close"].shift(1)).abs(),(df["Low"]-df["Close"].shift(1)).abs()))
    atr = tr.rolling(sim.atr_period).mean().values

    times = df.index.time
    dates = df.index.date
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values; opn = df["Open"].values
    vol = df["Volume"].values

    # Use last N days if specified
    if sim_days:
        unique_dates = sorted(set(dates))
        start_date = unique_dates[-sim_days]
        start_idx = np.searchsorted(dates, start_date)
    else:
        start_idx = max(sim.ema_slow, sim.atr_period) + 10

    n = len(df)
    for i in range(start_idx, n):
        if np.isnan(atr[i]): continue
        sim.process_bar(i, opn[i], high[i], low[i], close[i], vol[i],
                        times[i], ema20[i], ema50[i], atr[i])

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=40, help="Simulate last N trading days")
    parser.add_argument("--slippage", type=float, default=0.015, help="Mean slippage $/share")
    parser.add_argument("--runs", type=int, default=20, help="Monte Carlo runs (random slippage)")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df)} bars")

    # Single detailed run
    print(f"\n{'='*70}")
    print(f"LIVE SIMULATION: last {args.days} days, slip=${args.slippage:.3f}/sh")
    print(f"{'='*70}")

    sim = run_simulation(df, sim_days=args.days, slippage_mean=args.slippage)

    trades = pd.DataFrame(sim.trades)
    if len(trades) == 0:
        print("No trades generated."); return

    total = len(trades); wins = (trades['pnl']>0).sum(); losses = total - wins
    gw = trades.loc[trades['pnl']>0, 'pnl'].sum()
    gl = abs(trades.loc[trades['pnl']<=0, 'pnl'].sum())
    pf = gw / gl if gl > 0 else 0
    ret = (sim.equity - sim.capital) / sim.capital * 100

    print(f"\n  Capital: ${sim.capital:,} → ${sim.equity:,.0f} ({ret:+.2f}%)")
    print(f"  Trades: {total} ({total/args.days:.1f}/day)")
    print(f"  WR: {wins/total*100:.1f}% ({wins}W / {losses}L)")
    print(f"  PF: {pf:.3f}")
    if wins > 0 and losses > 0:
        print(f"  AvgWin: ${gw/wins:.2f}  AvgLoss: ${gl/losses:.2f}")

    # Exit reasons
    print(f"\n  Exit reasons:")
    for reason in trades['reason'].unique():
        sub = trades[trades['reason']==reason]
        print(f"    {reason:<15}: {len(sub):>4} ({len(sub)/total*100:>5.1f}%)  "
              f"avgPnL=${sub['pnl'].mean():.2f}")

    # Daily P&L
    eq = pd.DataFrame(sim.equity_curve)
    print(f"\n  Lock hit rate: {trades['lock_hit'].mean()*100:.1f}%")

    # Monte Carlo: run N times with random slippage seeds
    print(f"\n{'='*70}")
    print(f"MONTE CARLO: {args.runs} runs with random slippage (mean=${args.slippage})")
    print(f"{'='*70}")

    mc_pfs = []; mc_rets = []; mc_wrs = []
    for seed in range(args.runs):
        np.random.seed(seed)
        s = run_simulation(df, sim_days=args.days, slippage_mean=args.slippage)
        t = pd.DataFrame(s.trades)
        if len(t) == 0: continue
        w = (t['pnl']>0).sum(); l = len(t) - w
        gw = t.loc[t['pnl']>0, 'pnl'].sum()
        gl = abs(t.loc[t['pnl']<=0, 'pnl'].sum())
        mc_pfs.append(gw/gl if gl > 0 else 0)
        mc_rets.append((s.equity - s.capital) / s.capital * 100)
        mc_wrs.append(w/len(t)*100)

    mc_pfs = np.array(mc_pfs); mc_rets = np.array(mc_rets)
    print(f"  PF:  mean={mc_pfs.mean():.3f}  min={mc_pfs.min():.3f}  max={mc_pfs.max():.3f}  std={mc_pfs.std():.3f}")
    print(f"  Ret: mean={mc_rets.mean():.2f}%  min={mc_rets.min():.2f}%  max={mc_rets.max():.2f}%")
    print(f"  WR:  mean={np.mean(mc_wrs):.1f}%")
    print(f"  Profitable runs: {(mc_rets > 0).sum()}/{len(mc_rets)} ({(mc_rets > 0).mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
