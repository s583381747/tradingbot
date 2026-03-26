# MNQ Real Slippage Data (from Tradovate account)

Source: /Users/mac/Downloads/Orders (2).csv
Sample: 45 MNQ stop fills, Jan-Mar 2026

## Stop Order Slippage

| Metric | Ticks | USD |
|--------|-------|-----|
| Mean | 1.94 | $0.97 |
| Median | 1.0 | $0.50 |
| P90 | 5.2 | $2.60 |
| P95 | 6.8 | $3.40 |
| Max | 18.0 | $9.00 |

## Distribution
| Slippage | Count | % |
|----------|-------|---|
| 0 ticks | 18 | 40% |
| 1 tick | 12 | 27% |
| 2 ticks | 3 | 7% |
| 3+ ticks | 12 | 27% |

## Comparison
| Product | Mean slip (ticks) | Zero slip % |
|---------|-------------------|-------------|
| NQ | 0.25 | 75% |
| MNQ | 1.94 | 40% |
| MES | 0.26 | 74% |

## Key Insight
MNQ slippage is ~8x worse than NQ/MES. Fat-tailed distribution — 27% of stops slip 3+ ticks.
Our backtest assumed 1 tick ($0.50). Real mean is 2 ticks ($1.00).
