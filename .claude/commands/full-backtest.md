---
argument-hint: "策略文件路径，如: src/strategy/implementations/squeeze_breakout.py"
---
# Full Backtest

Run comprehensive backtest on: $ARGUMENTS

1. Load NQ data from `data/barchart_nq/NQ_1min_continuous_RTH.csv`
2. Split: IS=2022-2023, OOS=2024-2026
3. Run with CORRECT cost model (per-contract, gap-through fill)
4. Output:
   - IS / OOS / 4Y results: PF, WR, DD, $/day, Sharpe, Sortino, Calmar
   - Yearly breakdown (2022-2026)
   - Exit type distribution
   - Cost/risk percentage
   - Walk-forward (6-month rolling windows)
   - Cost stress test (1x, 1.5x, 2x)
   - 1-bar delay test
5. Compare vs baseline (EMA20 touch v11: PF=1.465 MNQ×2, PF=1.836 NQ×1)
6. Write report to `src/backtest/reports/`
