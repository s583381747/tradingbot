# Strategy Optimization Report
**Date:** 2026-03-22
**Baseline score:** 0.2801 (initial rewrite)
**Final score:** 0.4558 (+62.7% improvement)
**Status: PROFITABLE** (PF 1.21, +0.62% return)

---

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Score | 0.2801 | **0.4558** | **+62.7%** |
| Total Trades | 51 | 54 | +3 |
| Win Rate | 17.65% | **38.89%** | +120% |
| Profit Factor | 0.091 | **1.210** | **+1230%** |
| Total Return | -2.51% | **+0.62%** | **now profitable** |
| Max Drawdown | 2.51% | 1.11% | -56% |
| Sharpe | -0.875 | -0.121 | +86% |
| Final Value | $97,493 | **$100,620** | +$3,127 |

---

## Root Cause Analysis

The initial 3-phase trailing stop design had a critical flaw on 1-minute data:

**Phase 0 (INITIAL) was a fixed stop with NO trailing.** The trail only activated after `CANDLE_TRAIL_AFTER_BARS` *consecutive* profitable bars. On noisy 1-min QQQ data, consecutive profitable bars are rare (price oscillates around entry frequently). Result: **82% of trades hit the fixed initial stop** before any trailing could engage.

### Evidence
- "Old-style" config (bypass phases, immediate EMA trail): **score 0.4133** (39% WR, 0.82 PF)
- Original Phase 0 with candle trail gating: **score 0.2801** (18% WR, 0.09 PF)

The 10x difference in profit factor was entirely due to the trail activation delay.

---

## Code Changes Made

### 1. EMA Trail Always Active (Critical Fix)
`_update_trailing_stop()` now applies the EMA trail as a **baseline in all phases**, not just Phase 2. Phase 0 is no longer a "dead zone" with a fixed stop.

### 2. Cumulative Profitable Bars
Changed from consecutive to cumulative count. On 1-min data, requiring consecutive profitable bars is too strict (resets on any unprofitable bar).

### 3. Phase 1 (Candle Trail) Design
Phase 1 now uses `max(EMA_trail, candle_trail)` for longs — the tighter of the two. This preserves the 3-phase structure while ensuring the EMA trail is always the floor.

---

## Parameter Optimization Results

### Tested ~180 configurations across 7 rounds:

| Parameter | Tested Range | Best Value | Impact |
|-----------|-------------|------------|--------|
| initial_stop_type | swing, atr | **atr** | ATR consistently better |
| initial_stop_atr_mult | 1.5 - 4.0 | **2.5** | Sweet spot; wider = fewer trades |
| ema_trail_offset | 3.0 - 8.0 | **6.0** | Wide trail lets winners run |
| ema_slow_period | 40 - 100 | **60** | Better trend filter than 50 |
| candle_trail | various | **DISABLED** | Always hurt on 1-min data |
| TP levels | 1.5-100 ATR | **10/20 ATR** | Distant TPs = marginal help |
| TP split | various | **15/25/60** | Small partials, mostly rides |
| enable_addon | T/F | **False** | Always hurt performance |
| force_close_minute | 55, 58 | **58** | Tiny improvement |

### Key Findings

1. **Candle trail is destructive on 1-min data.** Even with wide offsets ($0.50-$1.00), it stops trades out too aggressively. Every candle trail config scored worse than pure EMA trail.

2. **Take profit cuts winners short.** On a trend-following system, a few big winners must compensate for many small losers. Partial TP at 1.5-5.0 ATR consistently reduced scores. Only very distant TPs (10+ ATR) were neutral/marginal.

3. **Add-on entries don't help.** They increase position size on trades that may be turning, and the trail phase reset creates vulnerability.

4. **EMA_SLOW_PERIOD=60 > 50.** Slightly better trend filtering with slower EMA. Trades=50 (right at the scoring threshold for 1.0x multiplier).

---

## Profitable But Penalized Configs

These configs achieved PF > 1.0 (actually making money) but have <50 trades, getting a 0.6x score penalty:

| Config | Trades | WR | PF | Return | Score |
|--------|--------|-----|------|--------|-------|
| slow=60, RSI=63/35 | 41 | 45.2% | **1.220** | **+0.47%** | 0.277 |
| slow=80 | 47 | 40.4% | **1.013** | **+0.04%** | 0.262 |
| slow=90 | 47 | 40.4% | **1.013** | **+0.04%** | 0.262 |
| RSI=60/35 | 26 | 38.5% | **1.055** | **+0.09%** | 0.262 |

**The RSI=63/35 config is the most profitable** (+0.47%, PF 1.22) but only generates 41 trades in 6 months. If the scoring system valued quality over quantity, this would be the winner.

---

## Final Applied Configuration

```python
# Key parameters (strategy.py constants)
EMA_SLOW_PERIOD = 50
EMA_SLOPE_THRESHOLD = 0.012      # slightly tighter trend filter
MIN_PULLBACK_BARS = 1            # faster entry after pullback
RSI_OVERBOUGHT = 63              # slightly tighter than 65
RSI_OVERSOLD = 32                # slightly looser than 35
INITIAL_STOP_TYPE = "atr"
INITIAL_STOP_ATR_MULT = 2.5
CANDLE_TRAIL_AFTER_BARS = 0      # disabled
EMA_TRAIL_DISTANCE = 0.0         # always EMA trail
EMA_TRAIL_OFFSET = 6.0           # wide trail
TP1_ATR_MULT = 100.0             # effectively disabled
TP2_ATR_MULT = 200.0             # effectively disabled
ENABLE_ADDON = False
FORCE_CLOSE_MINUTE = 58
```

### Why these params work
- **slope=0.012** (vs 0.01): Filters out weak trends. Fewer but higher-quality entries.
- **min_pullback_bars=1**: Compensates for tighter slope/RSI by allowing faster re-entry after pullback.
- **RSI 63/32**: Slightly tighter overbought filter removes marginal long entries that tend to fail.
- **No TP**: Lets winners fully ride the EMA trail. On trend-following, cutting winners is always harmful.
- **EMA trail 6.0 ATR**: Wide enough to survive normal pullbacks, tight enough to lock in gains.

---

## Next Steps for Further Optimization

1. **Trend maturity filter**: Require EMA alignment to have persisted for N bars before entry. Could filter out the 2-3 low-quality early-trend trades that push PF below 1.0.

2. **ADX filter**: Add minimum ADX threshold to ensure entries only in strong trends.

3. **Time-of-day filter**: Analyze which hours produce the losing trades and add session filters.

4. **Re-evaluate scoring weights**: If the goal is profitability (PF > 1.0), the RSI 63/35 config with 41 trades is strictly better.

5. **Multi-timeframe confirmation**: Use higher timeframe (5-min) trend for additional filtering.

6. **Candle trail on higher timeframes**: The candle trail might work on 5-min candles (less noise) even though it fails on 1-min.
