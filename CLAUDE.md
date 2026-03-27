# CLAUDE.md — NQ Futures Quantitative Trading System

## Project Overview
NQ/MNQ futures intraday trading system. Validated on 4.2Y real NQ continuous contract data.
Core instruments: NQ ($20/pt) and MNQ ($2/pt) on CME Globex.
Goal: Sharpe > 2.0, MaxDD < 15% of account, positive every calendar year.

## Current Best Strategy
**Volatility Squeeze Breakout (30min NQ)** — PF=1.53, Sharpe=2.33, cost/risk=0.6%
- Two independent agents converged on this approach
- Trend following + vol compression filter + pullback entry
- Edge is in EXIT SYSTEM (gate + BE + chandelier), NOT entry signal

## Critical Finding (Red Team Validated)
**Entry signals provide ZERO alpha.** Random entry + current exit system achieves PF=1.484.
54.5% of random strategies beat EMA20 touch. p=0.50. All edge comes from:
1. MFE gate (cut losers fast at breakeven)
2. BE trigger + stop move (protect small winners)
3. Chandelier trail (let big winners run)

## Technology Stack
- Python 3.13+, pandas, numpy
- Data: NQ 1-min continuous RTH (Barchart, Panama Canal adjusted)
- Backtest: custom vectorized engine (src/backtest/engine.py)
- Costs: per-contract model with gap-through stop fill
- No external backtest frameworks (no zipline/backtrader)

## Cost Model (MANDATORY — all agents must use)
All costs are PER CONTRACT. Never flat per trade.
```python
MNQ: comm=$2.46 RT, spread=$0.50×nc, stop_slip=$1.00×nc, be_slip=$1.00×nc
NQ:  comm=$2.46 RT, spread=$5.00×nc, stop_slip=$1.25×nc, be_slip=$1.25×nc
```
Stop fill: gap-through model — `fill_price = min(stop, bar_open)` for longs.
Shared engine: `src/backtest/engine.py`

## Data
- `data/barchart_nq/NQ_1min_continuous_RTH.csv` — 417K bars, 2021-12 to 2026-03
- Timezone: file is CT, add 1 hour for ET
- RTH: 09:30-16:00 ET
- IS = 2022-2023, OOS = 2024-2026 (forward walk)

## Risk Hard Rules (ALL agents must obey)
- MaxDD < $2,000 for Topstep 50K evaluation
- MaxDD < 15% of account for personal trading
- Daily loss limit: 2R (R = risk per trade)
- No entry after 14:00 ET, force close at 15:58 ET
- No look-ahead bias — signals use only closed bar data
- New strategies must pass 4Y IS+OOS + walk-forward before acceptance

## Coding Standards
- All strategy functions return a DataFrame with columns: pnl, r, exit, date, cost, risk
- Stop fill must use gap-through model (never fill at exact stop price)
- Cost model must be per-contract (check with: cost/risk% should be 0.4-5%)
- float64 precision, UTC timestamps internally, ET for display

## Agent Collaboration Protocol
- Strategy code → `src/strategy/implementations/`
- Backtest reports → `src/backtest/reports/`
- Risk reviews → `docs/RISK_REVIEW_*.md`
- All agents update `docs/STATUS.md` on completion
- Red team agents may MESSAGE blue team directly when finding fatal flaws

## Directory Ownership (prevent Git conflicts)
- alpha-researcher → `src/alpha/`, `docs/STRATEGY_SPEC.md`
- backtest-engineer → `src/backtest/`, `tests/test_backtest_*`
- risk-manager → `src/risk/`, `docs/RISK_*.md`
- execution-engineer → `src/execution/`, `tests/test_execution_*`
- data-engineer → `src/data/`, `configs/`
- strategies → `src/strategy/implementations/`

## Experiment History (15 experiments, see .lab/log.md)
Key conclusions:
1. ML entry filters don't work (AUC=0.53, 70 features tested)
2. Volume S/R targets are harmful (cap big winners)
3. DCP inversely correlated with big winners
4. gate_tighten=0.0 (BE on gate fail) validated on 7/7 walk-forward periods
5. 30min NQ bars optimal for cost/risk (0.4-0.6% vs 4.4-8.4% on 3min MNQ)
6. Cost bug found and fixed: spread/slip must be per contract, not flat
