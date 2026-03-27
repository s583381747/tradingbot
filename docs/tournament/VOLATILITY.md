# Tournament: Volatility-Gated Trend Strategy (NQ)

## Strategy Overview

**Core Idea:** Trend-following works best during LOW volatility regimes.
When ATR is contracting (low percentile) and Bollinger Bandwidth is narrow,
trends are smooth and orderly. Pullback entries during these contraction
phases deliver better R-multiples with lower stop-outs.

**Mechanism:** EMA25/60 trend + pullback touch + dual volatility contraction gate (ATR pctl + BB width pctl)

**Key finding:** OOS PF (1.659) EXCEEDS IS PF (1.391) -- negative decay indicates genuine edge, not curve-fitting. Volatility filter cuts DD by ~53% vs unfiltered baseline ($9.7K vs $20.9K).

**Timeframe:** 30-minute bars
**Instrument:** NQ (full-size, $20/point)
**Contracts:** 1

## Signal Logic

1. **Trend:** EMA25 > EMA60, close above EMA25 (long)
2. **Pullback Touch:** Low within 0.1*ATR of EMA25
3. **Vol Gate (ATR):** ATR percentile < 45% of last 50 bars
4. **Vol Gate (BW):** BB Bandwidth percentile < 55% of last 40 bars

## Risk Management

- Initial stop: 0.4*ATR below touch bar extreme
- Breakeven: Entry+0.1R after 1.0R MFE
- Chandelier trail: 3.0*ATR off 10-bar high/low, after 1.5R
- Max hold: 15 bars (7.5hrs)
- EOD close: 15:50:00
- Daily loss limit: 2.0R

## Cost Model (NQ per contract)

| Cost Component | Amount |
|---|---|
| Commission RT | $2.46 |
| Spread (entry) | $5.0 |
| Stop slippage | $1.25 |

## Results Summary

| Metric | IS (2022-2023) | OOS (2024-2026) | Full (4Y+) |
|---|---|---|---|
| Profit Factor | 1.391 | 1.659 | 1.533 |
| Total PnL | $+18,151 | $+39,681 | $+57,148 |
| Max Drawdown | $7,362 | $9,725 | $9,725 |
| $/day | $+193.1 | $+345.1 | $+272.1 |
| Sharpe | 1.755 | 2.820 | 2.334 |
| Trades | 103 | 132 | 236 |
| Win Rate | 53.4% | 51.5% | 52.1% |
| Trades/Day | 1.10 | 1.15 | 1.12 |
| 5R+ | 1 | 1 | 2 |
| Cost/Risk | 0.7% | 0.6% | 0.6% |
| DD % of $50K | 14.7% | 19.4% | 19.4% |

## Exit Breakdown

| Exit | IS | OOS | Full |
|---|---|---|---|
| be | 13 | 12 | 25 |
| eod | 37 | 57 | 94 |
| stop | 43 | 52 | 96 |
| timeout | 1 | 0 | 1 |
| trail | 9 | 11 | 20 |

## Yearly Breakdown

| Year | PF | PnL | Max DD | Trades | $/day | Win% |
|---|---|---|---|---|---|---|
| 2022 | 1.628 | $+16,580 | $7,362 | 48 | $+368.4 | 56.2% |
| 2023 | 1.079 | $+1,571 | $6,200 | 55 | $+32.1 | 50.9% |
| 2024 | 1.872 | $+24,660 | $7,231 | 72 | $+397.7 | 52.8% |
| 2025 | 1.360 | $+9,273 | $9,483 | 50 | $+210.8 | 50.0% |
| 2026 | 1.733 | $+5,065 | $6,233 | 11 | $+506.5 | 45.5% |

## OOS Decay Analysis

- IS PF: 1.391
- OOS PF: 1.659
- PF Decay: -19.3%
- **Verdict: PASS**

## Parameters

```python
STRATEGY = {
    "tf_minutes": 30,
    "ema_fast": 25,
    "ema_slow": 60,
    "atr_period": 14,
    "touch_tol": 0.1,
    "touch_below_max": 0.4,
    "vol_lookback": 50,
    "vol_threshold": 45,
    "bw_period": 20,
    "bw_mult": 2.0,
    "bw_lookback": 40,
    "bw_threshold": 55,
    "no_entry_before": datetime.time(9, 15),
    "no_entry_after": datetime.time(15, 0),
    "stop_buffer": 0.4,
    "be_trigger_r": 1.0,
    "be_stop_r": 0.1,
    "chand_bars": 10,
    "chand_mult": 3.0,
    "trail_activate_r": 1.5,
    "max_hold_bars": 15,
    "force_close_at": datetime.time(15, 50),
    "daily_loss_r": 2.0,
    "skip_after_win": 1,
    "n_contracts": 1,
}
```