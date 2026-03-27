# Red/Blue Team Status — Last Updated: 2026-03-27

## Mission
Find new NQ strategy opportunities that beat EMA20 touch (PF=1.47, DD=$1,753 on 10min MNQ×2).
Target: PF > 1.5, DD < $2,000 for Topstep 50K, OR PF > 1.8, DD < 15% for personal $50K.

## Current Baseline (v11 — what blue team must beat)
- 10min bar, EMA20 touch close, gate=0.0, 2 MNQ
- 4Y NQ: PF=1.465, DD=$1,753, $26/day, Sharpe=2.20
- NQ×1: PF=1.836, DD=$5,952, $202/day, Sharpe=3.42
- Core weakness: MNQ cost/risk = 4.4%, eating 73% of raw edge at 3min

## Team Status

### Blue Team
- [ ] blue-quant: Design 2-3 new strategy ideas
- [ ] blue-backtest: Validate with walk-forward + stress test

### Red Team
- [ ] red-critic: Code audit + statistical attack on blue strategies
- [ ] red-breaker: Destruction tests + worst-case scenarios

## Key Constraints
- Data: data/barchart_nq/NQ_1min_continuous_RTH.csv (417K 1min bars, 2022-2026)
- Cost model: PER CONTRACT (spread × nc, slip × nc)
- Stop fill: gap-through using Open price
- Shared engine: src/backtest_engine.py
