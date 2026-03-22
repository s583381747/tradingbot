"""
Trade detail analyzer for the current strategy.
Runs a backtest, tracks every trade, and prints comprehensive statistics.
"""

from __future__ import annotations

import datetime
import statistics
from collections import defaultdict

import backtrader as bt

# Import the strategy without modification
from strategy import Strategy


# ─────────────────────────────────────────────────────────────────────
# Custom Analyzer: captures per-trade entry/exit bar, duration, P&L
# ─────────────────────────────────────────────────────────────────────

class TradeDetailAnalyzer(bt.Analyzer):
    """Records entry bar, exit bar, duration, and P&L for every closed trade."""

    def __init__(self):
        super().__init__()
        self._trades = []          # completed trades
        self._open_trades = {}     # tradeid -> {entry_bar, entry_dt}

    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            # Record entry
            bar_num = len(self.strategy)
            dt = self.strategy.data.datetime.datetime(0)
            self._open_trades[trade.ref] = {
                "entry_bar": bar_num,
                "entry_dt": dt,
            }

        if trade.isclosed:
            bar_num = len(self.strategy)
            dt = self.strategy.data.datetime.datetime(0)
            entry_info = self._open_trades.pop(trade.ref, None)
            if entry_info is None:
                # Fallback: trade opened before analyzer was added
                entry_bar = bar_num
                entry_dt = dt
            else:
                entry_bar = entry_info["entry_bar"]
                entry_dt = entry_info["entry_dt"]

            self._trades.append({
                "entry_bar": entry_bar,
                "exit_bar": bar_num,
                "entry_dt": entry_dt,
                "exit_dt": dt,
                "duration_bars": bar_num - entry_bar,
                "pnl": trade.pnlcomm,       # net P&L after commission
                "pnl_gross": trade.pnl,      # gross P&L
                "size": abs(trade.size) if trade.size != 0 else abs(trade.history[0][1] if trade.history else 0),
            })

    def get_analysis(self):
        return self._trades


# ─────────────────────────────────────────────────────────────────────
# Run backtest
# ─────────────────────────────────────────────────────────────────────

def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)

    data = bt.feeds.GenericCSVData(
        dataname="data/QQQ_1Min_2025-09-21_2026-03-21.csv",
        dtformat="%Y-%m-%d %H:%M:%S",
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
    )
    cerebro.adddata(data)

    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(TradeDetailAnalyzer, _name="trade_details")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="builtin_ta")

    print("Starting backtest ...")
    results = cerebro.run()
    strat = results[0]

    trades = strat.analyzers.trade_details.get_analysis()
    builtin = strat.analyzers.builtin_ta.get_analysis()
    final_value = cerebro.broker.getvalue()

    return trades, builtin, final_value, data


# ─────────────────────────────────────────────────────────────────────
# Analyze and print
# ─────────────────────────────────────────────────────────────────────

def analyze(trades, builtin, final_value, data):
    print("=" * 70)
    print("  TRADE DETAIL ANALYSIS")
    print("=" * 70)

    # ── Portfolio summary ──
    print(f"\nFinal portfolio value : ${final_value:,.2f}")
    print(f"Net P&L              : ${final_value - 100_000:,.2f}")
    print(f"Total closed trades  : {len(trades)}")

    if not trades:
        print("\nNo trades to analyze.")
        return

    # ── Compute trading days from the data ──
    all_dates = set()
    for t in trades:
        all_dates.add(t["entry_dt"].date())
        all_dates.add(t["exit_dt"].date())

    # Also count total trading days in the dataset
    # We'll derive unique dates from entry/exit datetimes we saw, but
    # for "total trading days in the data" we need to scan the data feed.
    # Since the data feed is consumed, we count unique dates from trades
    # plus estimate from the CSV date range.
    # Let's read unique dates from the CSV directly for accuracy.
    trading_days_in_data = set()
    with open("data/QQQ_1Min_2025-09-21_2026-03-21.csv", "r") as f:
        header = f.readline()
        for line in f:
            date_str = line.split(",")[0].split(" ")[0]
            trading_days_in_data.add(date_str)
    total_trading_days = len(trading_days_in_data)

    # ── Basic stats ──
    total_trades = len(trades)
    avg_trades_per_day = total_trades / total_trading_days if total_trading_days else 0

    durations = [t["duration_bars"] for t in trades]
    pnls = [t["pnl"] for t in trades]

    avg_duration = statistics.mean(durations)
    median_duration = statistics.median(durations)

    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]

    win_durations = [t["duration_bars"] for t in winners]
    lose_durations = [t["duration_bars"] for t in losers]

    win_pnls = [t["pnl"] for t in winners]
    lose_pnls = [t["pnl"] for t in losers]

    avg_win_dur = statistics.mean(win_durations) if win_durations else 0
    avg_lose_dur = statistics.mean(lose_durations) if lose_durations else 0

    avg_win_pnl = statistics.mean(win_pnls) if win_pnls else 0
    avg_lose_pnl = statistics.mean(lose_pnls) if lose_pnls else 0

    largest_win = max(pnls) if pnls else 0
    largest_loss = min(pnls) if pnls else 0

    # ── Consecutive wins / losses ──
    max_consec_wins = 0
    max_consec_losses = 0
    cur_wins = 0
    cur_losses = 0
    for t in trades:
        if t["pnl"] > 0:
            cur_wins += 1
            cur_losses = 0
            max_consec_wins = max(max_consec_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_consec_losses = max(max_consec_losses, cur_losses)

    # ── Trades by hour of day (entry hour) ──
    hour_dist = defaultdict(int)
    for t in trades:
        hour_dist[t["entry_dt"].hour] += 1

    # ── Print everything ──
    print(f"\n{'─' * 70}")
    print("  GENERAL STATISTICS")
    print(f"{'─' * 70}")
    print(f"  Total trading days in data   : {total_trading_days}")
    print(f"  Total closed trades          : {total_trades}")
    print(f"  Average trades per day       : {avg_trades_per_day:.2f}")
    print(f"  Winning trades               : {len(winners)}  ({len(winners)/total_trades*100:.1f}%)")
    print(f"  Losing trades                : {len(losers)}  ({len(losers)/total_trades*100:.1f}%)")

    print(f"\n{'─' * 70}")
    print("  DURATION STATISTICS  (1 bar = 1 minute)")
    print(f"{'─' * 70}")
    print(f"  Average trade duration       : {avg_duration:.1f} bars (minutes)")
    print(f"  Median trade duration        : {median_duration:.1f} bars (minutes)")
    print(f"  Min trade duration           : {min(durations)} bars")
    print(f"  Max trade duration           : {max(durations)} bars")
    print(f"  Avg winning trade duration   : {avg_win_dur:.1f} bars (minutes)")
    print(f"  Avg losing trade duration    : {avg_lose_dur:.1f} bars (minutes)")

    print(f"\n{'─' * 70}")
    print("  PROFIT / LOSS STATISTICS")
    print(f"{'─' * 70}")
    print(f"  Avg profit per winning trade : ${avg_win_pnl:,.2f}")
    print(f"  Avg loss per losing trade    : ${avg_lose_pnl:,.2f}")
    print(f"  Largest single win           : ${largest_win:,.2f}")
    print(f"  Largest single loss          : ${largest_loss:,.2f}")
    print(f"  Total gross profit (winners) : ${sum(win_pnls):,.2f}")
    print(f"  Total gross loss (losers)    : ${sum(lose_pnls):,.2f}")
    if lose_pnls:
        profit_factor = abs(sum(win_pnls) / sum(lose_pnls)) if sum(lose_pnls) != 0 else float("inf")
        print(f"  Profit factor                : {profit_factor:.2f}")

    print(f"\n{'─' * 70}")
    print("  STREAK STATISTICS")
    print(f"{'─' * 70}")
    print(f"  Max consecutive wins         : {max_consec_wins}")
    print(f"  Max consecutive losses       : {max_consec_losses}")

    print(f"\n{'─' * 70}")
    print("  DISTRIBUTION OF TRADES BY HOUR OF DAY (entry hour)")
    print(f"{'─' * 70}")
    for hour in sorted(hour_dist.keys()):
        bar = "#" * hour_dist[hour]
        print(f"  {hour:02d}:00  {hour_dist[hour]:>4d}  {bar}")

    # ── Individual trade log (first 20 + last 10) ──
    print(f"\n{'─' * 70}")
    print("  INDIVIDUAL TRADE LOG  (first 20 trades)")
    print(f"{'─' * 70}")
    print(f"  {'#':>4s}  {'Entry Time':<20s}  {'Exit Time':<20s}  {'Dur':>5s}  {'P&L':>10s}")
    print(f"  {'─'*4}  {'─'*20}  {'─'*20}  {'─'*5}  {'─'*10}")
    for i, t in enumerate(trades[:20], 1):
        marker = "W" if t["pnl"] > 0 else "L"
        print(f"  {i:4d}  {str(t['entry_dt']):<20s}  {str(t['exit_dt']):<20s}  {t['duration_bars']:5d}  ${t['pnl']:>9,.2f} {marker}")

    if len(trades) > 20:
        print(f"\n  ... ({len(trades) - 30} trades omitted) ...\n")
        print(f"  INDIVIDUAL TRADE LOG  (last 10 trades)")
        print(f"  {'─'*4}  {'─'*20}  {'─'*20}  {'─'*5}  {'─'*10}")
        for i, t in enumerate(trades[-10:], len(trades) - 9):
            marker = "W" if t["pnl"] > 0 else "L"
            print(f"  {i:4d}  {str(t['entry_dt']):<20s}  {str(t['exit_dt']):<20s}  {t['duration_bars']:5d}  ${t['pnl']:>9,.2f} {marker}")

    print(f"\n{'=' * 70}")
    print("  END OF ANALYSIS")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trades, builtin, final_value, data = run_backtest()
    analyze(trades, builtin, final_value, data)
