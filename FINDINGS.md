# EMA20 Trend-Following Strategy — Complete Findings

> Date: 2026-03-22 | Status: Phase 2A complete, pending Phase 2B validation

## TL;DR

We built a QQQ 1-minute intraday trend-following strategy (EMA20 pullback), optimized it to score 0.4558 (PF 1.21, +0.62%), then rigorously tested whether the edge is real. **The entry concept has genuine predictive power (30-60 minute horizon), but the exact parameters are overfit. Widening stops from 2.5x to 5.0x ATR improved PF to 1.48.** Next step: validate on 5-min bars and SPY data.

---

## Strategy Description (Plain Text)

**Entry:** Wait for price to pull back to EMA20 on 1-min QQQ, confirm uptrend (EMA20 slope > 0.012%/bar, EMA20 > EMA50), confirm bounce (close > prior bar high), RSI < 63 (not chasing). Mirror for shorts.

**Exit:** Initial stop at entry - 2.5×ATR. Trailing stop: EMA20 - 6×ATR (ratchets up only). Time exit: close losing trades after 45 bars. Force close at 15:58.

**Sizing:** 1% account risk per trade, max 25% of account in one position.

---

## Phase 1: Diagnostic Experiments

### Exp 1.1 — Parameter Sensitivity: OVERFIT CONFIRMED
- **4 robust params** (smooth curves): stop ATR mult, EMA trail offset, losers_max_bars, min_pullback_bars
- **4 cliff params** (score halves at ±10%): ema_slope_threshold, pullback_touch_mult, rsi_overbought, rsi_oversold
- **Verdict:** Entry filter params are curve-fit to this specific 6-month window

### Exp 1.2 — Random Baseline: ENTRY EDGE IS REAL
- 0/1000 random iterations beat the strategy's score (0.4558)
- Random entry mean PF: 0.23, mean return: -2.1%
- Strategy PF: 1.21, return: +0.62%
- **Verdict:** Entry signal genuinely outperforms random, even trend-biased random

### Exp 1.3 — Alpha Decomposition: MEDIUM-TERM SIGNAL
- Entry signal NOT significant at 5-10 bars (p > 0.4)
- **Significant at 30 bars** (p=0.020, +0.12%, 57% WR) **and 60 bars** (p=0.003, +0.21%, 69% WR)
- Exit system alone (random entry): DESTRUCTIVE (PF 0.24)
- Entry + hold-to-EOD: $715 PnL, 42 trades, +0.27%/trade
- Buy & hold QQQ: -2.61% over same period
- **Verdict:** Entry is a 30-60 minute predictor. Current 2.5x ATR stop kills trades in 5-15 min before signal materializes.

---

## Phase 2A: Architecture Fixes

### Exp 2.1 — Wide Stops: CONFIRMED IMPROVEMENT
| Stop Width | PF | Return | Trades |
|-----------|-----|--------|--------|
| 2.5 ATR (baseline) | 1.210 | +0.62% | 54 |
| **5.0 ATR** | **1.483** | **+1.17%** | 54 |
| 7.0 ATR | 1.485 | +1.18% | 54 |
| 10.0 ATR | 1.485 | +1.18% | 54 |

- At 5.0+ ATR, initial stop never fires — EMA trail exits first
- **Conclusion:** 2.5x stop was premature, killing trades before the 30-60min signal window

### Exp 2.3 — Parameter Blunting: FRAGILE
| Blunting | stop=2.5 PF | stop=5.0 PF | Trades |
|----------|-------------|-------------|--------|
| L0 (overfit) | 1.210 | 1.483 | 54 |
| L1 (mild: slope=0.01, rsi=65/35, pb=1.3) | 0.711 | **1.239** | 70 |
| L2 (moderate: slope=0.01, rsi=68/28, pb=1.5) | 0.496 | 0.649 | 116 |
| L3 (heavy) | 0.361 | 0.446 | 198 |

- L1 + wide stop still profitable (PF 1.24)
- L2+ collapses — edge depends on entry precision
- Wide stops help universally across all blunting levels
- **Verdict:** Edge is FRAGILE — survives mild blunting only

---

## Key Insights

1. **Signal timescale mismatch was the #1 bug.** Entry predicts 30-60 min ahead, but 2.5x ATR stop fires in 5-15 min. Fixing this (→5.0x) doubled returns.

2. **Entry concept is real but parameter-sensitive.** The EMA pullback + trend + momentum concept outperforms random (0/1000). But exact thresholds (slope=0.012, RSI=63) are likely overfit.

3. **Exit system has no independent alpha.** Random entries + our exits = PF 0.24. The trail/stop system only works paired with good entries.

4. **The 3-phase trail system from the plan doesn't work on 1-min data.** Candle trail is too aggressive. Only the EMA baseline trail is useful.

5. **TP and add-on entries hurt performance.** Trend-following needs winners to run. Partial TP and add-ons consistently reduced scores.

---

## Current Best Configuration

```python
# strategy.py optimized params
EMA_SLOPE_THRESHOLD = 0.012      # OVERFIT — needs wider value in production
PULLBACK_TOUCH_MULT = 1.2        # OVERFIT
RSI_OVERBOUGHT = 63              # OVERFIT
RSI_OVERSOLD = 32                # OVERFIT
INITIAL_STOP_ATR_MULT = 2.5      # Should be 5.0+ (Phase 2A finding)
EMA_TRAIL_OFFSET = 6.0           # ROBUST
LOSERS_MAX_BARS = 45             # ROBUST
```

Score: 0.4558 | PF: 1.21 | Return: +0.62% | Trades: 54 (6 months)

With stop=5.0: PF: 1.48 | Return: +1.17% (not yet applied to defaults)

---

## Pending: Phase 2B

1. **5-min bars** — Does the signal work with less noise? Are params more stable?
2. **SPY cross-validation** — Apply QQQ-optimized params to SPY without re-tuning
3. **Walk-forward analysis** — Train on rolling 90-day windows, test on next 30 days

## Pending: Phase 3 (requires 2-year data)

4. **Walk-forward on 2 years** — Download via Alpaca API (credentials in .env)
5. **Monte Carlo / bootstrap** — Statistical significance with larger sample
6. **Realistic execution** — Slippage model, next-bar-open fills

---

## File Structure

```
strategy.py          — Main strategy (backtrader, self-contained)
prepare.py           — Evaluation harness (DO NOT MODIFY)
analyze_trades.py    — Trade statistics analyzer

exp_1_1_sensitivity.py   — Parameter sensitivity analysis
exp_1_2_random_baseline.py — Random entry baseline (500 iters)
exp_1_3_alpha_decomp.py  — Alpha source decomposition
exp_2_1_wide_stops.py    — Wide stop experiments
exp_2_3_blunting.py      — Parameter blunting test

OPTIMIZATION_REPORT.md   — Detailed optimization report
FINDINGS.md              — This file (complete summary)

scripts/                 — Data download & backtest utilities
data/                    — Market data (QQQ 1-min, 6 months)
```

## How to Continue on Another Machine

```bash
git clone https://github.com/s583381747/tradingbot.git
cd tradingbot
pip install backtrader pandas numpy
# Copy .env with Alpaca credentials if needed for data download
PYTHONUTF8=1 python prepare.py   # Verify: should print score=0.455767
```
