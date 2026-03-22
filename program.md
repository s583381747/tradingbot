# Autoresearch Strategy Optimization — Agent Instructions

## Overview
You are an autonomous optimization agent. Your goal is to iteratively improve
`strategy.py` to maximize the composite score printed by `prepare.py`.

## Setup (run once at start)
1. Create and checkout a branch: `git checkout -b optim/auto-<timestamp>`
2. Run baseline: `PYTHONUTF8=1 python prepare.py > run.log 2>&1`
3. Extract score: `grep "^score:" run.log`
4. Record baseline in `results.tsv`

## Experiment Loop
Repeat until converged (score stops improving for 5+ iterations):

1. **Modify** `strategy.py` — change ONE thing at a time
2. **Commit** the change: `git add strategy.py && git commit -m "exp: <description>"`
3. **Run** backtest: `PYTHONUTF8=1 python prepare.py > run.log 2>&1`
4. **Extract** all metrics from `run.log`:
   ```
   score=$(grep "^score:" run.log | awk '{print $2}')
   total_return=$(grep "^total_return:" run.log | awk '{print $2}')
   sharpe=$(grep "^sharpe:" run.log | awk '{print $2}')
   max_drawdown=$(grep "^max_drawdown:" run.log | awk '{print $2}')
   win_rate=$(grep "^win_rate:" run.log | awk '{print $2}')
   profit_factor=$(grep "^profit_factor:" run.log | awk '{print $2}')
   total_trades=$(grep "^total_trades:" run.log | awk '{print $2}')
   ```
5. **Decide**:
   - If score > previous best → **keep** (status=kept)
   - If score <= previous best → **revert**: `git revert HEAD --no-edit` (status=reverted)
6. **Log** to `results.tsv` (append one row)

## What to Try (priority order)
1. **Indicator periods**: EMA_PERIOD (10-50), ATR_PERIOD (7-21), EMA_SLOPE_PERIOD (3-10)
2. **Pullback sensitivity**: PULLBACK_TOUCH_MULT (0.2-1.5), loosen to get more trades
3. **Trend filters**: EMA_SLOPE_THRESHOLD (0.0-0.05), REQUIRE_ABOVE_VWAP (True/False)
4. **Volume filter**: USE_VOLUME_FILTER, VOLUME_BREAKOUT_MULT (1.0-2.5)
5. **Stop loss**: SL_TYPE, SL_ATR_MULT, SL_OFFSET
6. **Take profit**: TP_ATR_MULT (1.0-4.0), PARTIAL_TP_PCT (0.3-0.8)
7. **Trailing stop**: TRAIL_TYPE, TRAIL_ATR_MULT (0.5-2.0)
8. **Session**: NO_ENTRY_AFTER_HOUR/MINUTE, OPENING_RANGE_BARS
9. **Entry logic changes**: relax breakout conditions, add new filters, modify state machine
10. **Risk management**: RISK_PCT, MAX_DAILY_TRADES, MAX_POSITION_PCT

## Rules
- **Only modify `strategy.py`** — never touch `prepare.py` or `program.md`
- The class **must** be named `Strategy` (this is the import contract)
- **One change per experiment** — so you can attribute score changes
- If score = -10, the strategy crashed or made <5 trades — investigate and fix
- Keep all hyperparameters as module-level constants at the top of `strategy.py`
- If stuck, try a fundamentally different approach (e.g., mean reversion, momentum)

## Score Components (for reference)
- Sharpe ratio (30%): higher is better, clamped [-2, 3]
- Profit factor (25%): gross_won / gross_lost, clamped [0, 3]
- Total return (20%): %, clamped [-20, 50]
- Win rate (10%): %, clamped [0, 100]
- Drawdown penalty (15%): lower drawdown = better
- Trade count gate: <5=-10, <20=0.2x, 20-50=0.6x, 50-500=1.0x, >500=0.8x

## Tips
- The baseline strategy likely has very few trades — focus first on getting trade count into 50-500 range
- Loosening PULLBACK_TOUCH_MULT and EMA_SLOPE_THRESHOLD usually increases trade count fastest
- Disabling REQUIRE_ABOVE_VWAP and USE_VOLUME_FILTER removes the strictest filters
- After getting enough trades, focus on improving win_rate and profit_factor
- Check `run.log` for errors if score = -10
