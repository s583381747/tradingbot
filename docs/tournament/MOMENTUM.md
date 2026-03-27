# MOMENTUM Strategy — Donchian Channel Breakout + ATR-Ratchet Trail

**Agent:** strategist-momentum
**Generated:** 2026-03-27 14:40

## Strategy Logic

- **Timeframe:** 15-minute bars (resampled from 1-min RTH)
- **Entry Long:** Close > 10-bar Donchian high + EMA50 slope positive (8 bars) + price > EMA
- **Entry Short:** Close < 10-bar Donchian low + EMA slope negative + price < EMA
- **Direction mode:** long
- **Initial Stop:** 1.5x ATR from entry
- **Trail:** ATR-ratchet: 2.5x ATR from highest-high (long) / lowest-low (short) since entry
- **Time filter:** No entries after 14:00 ET, force close 15:58 ET
- **Max hold:** 26 bars

## Cost Model

| | MNQ x2 | NQ x1 |
|---|---|---|
| Point value | $2/pt x2 | $20/pt x1 |
| Commission RT | $2.46/contract | $2.46/contract |
| Spread | $0.50/contract | $5.00/contract |
| Stop slippage | $1.00/contract | $1.25/contract |
| Gap-through | min/max(stop, open) | min/max(stop, open) |

## Selected Parameters

```python
PARAMS = {
    "bar_minutes": 15,
    "dc_entry_period": 10,
    "ema_period": 50,
    "ema_slope_bars": 8,
    "atr_period": 14,
    "stop_atr_mult": np.float64(1.5),
    "trail_atr_mult": np.float64(2.5),
    "no_entry_after": datetime.time(14, 0),
    "force_close_at": datetime.time(15, 58),
    "max_hold_bars": 26,
    "direction": 'long',
}
```

Selected via robustness ranking (median rank across PF, Sharpe, N, $/day, DD) from 768 IS param combos.

## Results: MNQ x2

### Summary

| Period | PF | WR% | Trades | Sharpe | $/day | MaxDD | Cost/Risk% |
|--------|-----|------|--------|--------|-------|-------|------------|
| IS (2022-2023) | 1.212 | 42.5 | 273 | 1.26 | 28.4 | $2205 | 2.7% |
| OOS (2024-2026) | 0.962 | 40.5 | 341 | -0.23 | -5.8 | $5367 | 2.3% |
| FULL (2022-2026) | 1.066 | 41.4 | 614 | 0.39 | 9.6 | $5367 | 2.5% |

### Yearly Breakdown

| Year | PF | WR% | Trades | Sharpe | $/day | MaxDD | Sortino | Calmar |
|------|-----|------|--------|--------|-------|-------|---------|--------|
| 2022 | 1.277 | 45.2 | 115 | 1.69 | 45.0 | $1660 | 4.22 | 6.82 |
| 2023 | 1.145 | 40.5 | 158 | 0.87 | 16.5 | $1611 | 2.44 | 2.58 |
| 2024 | 1.001 | 38.8 | 160 | 0.0 | 0.1 | $5048 | 0.01 | 0.0 |
| 2025 | 0.948 | 42.0 | 157 | -0.29 | -8.3 | $4695 | -0.61 | -0.45 |
| 2026 | 0.796 | 41.7 | 24 | -1.45 | -29.0 | $1462 | -2.39 | -4.16 |
| ALL | 1.066 | 41.4 | 614 | 0.39 | 9.6 | $5367 | 0.91 | 0.45 |

### Trade Analysis

- **5R+ trades:** 0
- **3R+ trades:** 17
- **Avg winner R:** 1.14
- **Avg loser R:** -0.83
- **Max consecutive losses:** 9
- **Best day:** $3693
- **Worst day:** $-1271

### Direction Breakdown

| Direction | Trades | PnL | Avg R |
|-----------|--------|-----|-------|
| long | 614 | $5235 | -0.016 |

### Exit Distribution

| Exit Type | Count | % |
|-----------|-------|---|
| stop | 366 | 59.6% |
| eod | 247 | 40.2% |
| time | 1 | 0.2% |

## Results: NQ x1

### Summary

| Period | PF | WR% | Trades | Sharpe | $/day | MaxDD | Cost/Risk% |
|--------|-----|------|--------|--------|-------|-------|------------|
| IS (2022-2023) | 1.265 | 43.6 | 273 | 1.53 | 172.5 | $10766 | 0.6% |
| OOS (2024-2026) | 1.002 | 40.8 | 341 | 0.01 | 1.8 | $23715 | 0.5% |
| FULL (2022-2026) | 1.111 | 42.0 | 614 | 0.65 | 78.7 | $23715 | 0.6% |

### Yearly Breakdown

| Year | PF | WR% | Trades | Sharpe | $/day | MaxDD | Sortino | Calmar |
|------|-----|------|--------|--------|-------|-------|---------|--------|
| 2022 | 1.321 | 46.1 | 115 | 1.92 | 255.2 | $8124 | 4.89 | 7.92 |
| 2023 | 1.206 | 41.8 | 158 | 1.19 | 113.0 | $7650 | 3.45 | 3.72 |
| 2024 | 1.043 | 38.8 | 160 | 0.28 | 32.0 | $23233 | 0.72 | 0.35 |
| 2025 | 0.986 | 42.0 | 157 | -0.08 | -11.1 | $20798 | -0.16 | -0.13 |
| 2026 | 0.834 | 45.8 | 24 | -1.15 | -114.5 | $7217 | -1.89 | -3.33 |
| ALL | 1.111 | 42.0 | 614 | 0.65 | 78.7 | $23715 | 1.5 | 0.84 |

### Trade Analysis

- **5R+ trades:** 0
- **3R+ trades:** 17
- **Avg winner R:** 1.14
- **Avg loser R:** -0.81
- **Max consecutive losses:** 9
- **Best day:** $18488
- **Worst day:** $-6322

### Direction Breakdown

| Direction | Trades | PnL | Avg R |
|-----------|--------|-----|-------|
| long | 614 | $42969 | 0.007 |

### Exit Distribution

| Exit Type | Count | % |
|-----------|-------|---|
| stop | 366 | 59.6% |
| eod | 247 | 40.2% |
| time | 1 | 0.2% |

## Walk-Forward Validation (MNQ x2)

| Period | PF | WR% | Trades | Sharpe | $/day | DD |
|--------|-----|------|--------|--------|-------|----|
| 2023-01-2023-07 | 1.48 | 43.2 | 81 | 2.48 | 52.6 | $1584 |
| 2023-07-2024-01 | 0.907 | 39.7 | 73 | -0.64 | -10.3 | $1611 |
| 2024-01-2024-07 | 0.834 | 32.5 | 83 | -1.24 | -26.1 | $4381 |
| 2024-07-2025-01 | 1.179 | 45.5 | 77 | 1.08 | 26.2 | $1993 |
| 2025-01-2025-07 | 1.229 | 49.3 | 69 | 1.03 | 37.7 | $1739 |

Profitable periods: 3/5

## OOS Degradation Analysis

**MNQ x2:**
- PF: IS=1.212 -> OOS=0.962 (-20.6%)
- Sharpe: IS=1.26 -> OOS=-0.23
- $/day: IS=28.4 -> OOS=-5.8

**NQ x1:**
- PF: IS=1.265 -> OOS=1.002 (-20.8%)
- Sharpe: IS=1.53 -> OOS=0.01
- $/day: IS=172.5 -> OOS=1.8

## Parameter Stability (IS sweep)

- Total combos tested: 768
- PF > 1.0: 700/768 (91%)
- PF > 1.2: 88/768
- PF > 1.5: 0/768
- Best IS PF: 1.431
- Median IS PF: 1.093
