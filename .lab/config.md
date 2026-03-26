# Research Configuration — MNQ on Real NQ Data

## Objective
Fine-tune strategy parameters on REAL NQ continuous contract data (not QQQ×40 proxy).
Maximize OOS PF while minimizing MaxDD for Topstep 50K.

## Primary Metric
Composite score = OOS_PF × 10 - OOS_MaxDD / 100
Measured by: `python3 run_nq_backtest.py` (prints IS/OOS results)

## Secondary Metrics
- IS Net PF (higher)
- OOS Net PF (higher, must > 1.3)
- OOS Daily $ (higher, target > $60/day)
- IS MaxDD (lower)
- OOS MaxDD (lower, HARD < $2,000)
- IS/OOS PF ratio (> 0.80 = not overfit)
- 5R+ count (must not decrease >25% from baseline)

## Data
- NQ continuous RTH 1min: `data/barchart_nq/NQ_1min_continuous_RTH.csv`
- IS: 2024-01 to 2026-03 (~2 years)
- OOS: 2022-01 to 2023-12 (~2 years)

## Cost Model (MNQ, real slippage)
- Commission: $2.46/contract RT
- Spread: $0.50/trade (1 tick)
- Stop slippage: $1.00/trade (real: mean 1.94 ticks)
- BE slippage: $1.00/BE exit
- MNQ = $2/point (direct NQ prices, no QQQ×40 conversion)

## Scope
- `run_nq_backtest.py` — strategy params dict `S`
- Parameters to tune: tf_minutes, stop_buffer, gate_bars, gate_mfe, gate_tighten,
  be_trigger_r, be_stop_r, chand_bars, chand_mult, daily_loss_r, skip_after_win,
  touch_tol, touch_below_max, no_entry_after

## Constraints
- No partial exits (MNQ whole contracts only)
- No look-ahead bias
- OOS must be profitable (PF > 1.0)
- MaxDD < $2,000 (Topstep 50K)
- Do NOT add new indicators or features (proven futile in exp 5-7)
- Focus on existing parameter optimization only

## Budget
3 minutes per experiment

## Termination
Run until interrupted or OOS daily > $70 with MaxDD < $800 on NQ real data

## Baseline (v8 on NQ real data)
IS: PF=2.137 $/d=+76.3 DD=$911 | OOS: PF=1.967 $/d=+59.9 DD=$853
Score = 1.967 × 10 - 853/100 = 11.14

## Best
v8 on NQ: score=11.14
