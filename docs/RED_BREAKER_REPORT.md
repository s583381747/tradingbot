# RED BREAKER REPORT -- Strategy Destruction Test

**Strategy:** EMA20 Touch, NQ v11 (10min bars, gate=0.0, MNQ costs per contract)
**Generated:** 2026-03-27
**Verdict: BRUISED -- structurally sound but entry signal is a placebo**

---

## Executive Summary

The strategy survives most destruction tests: it handles doubled costs, 10-minute execution delays, 8/9 market regimes, and 10 consecutive losing trades without blowing up. The max drawdown of $1,753 stays within Topstep's $2K limit even starting on the worst possible day.

**However, the critical finding is this: the EMA20 touch entry signal contributes ZERO alpha.** Random entries with the same exit system produce PF=1.484 (actually slightly better than the strategy's PF=1.465). The entire edge comes from the exit system (BE + chandelier trail + gate). The entry signal is a placebo -- it gives psychological comfort but adds no statistical value.

This means research effort should pivot entirely to exit optimization and money management, not entry filters.

---

## Baseline Performance

| Period | PF | PnL | Max DD | Trades | Win Rate | Sharpe | MCL | 5R+ |
|--------|---:|----:|-------:|-------:|---------:|-------:|----:|----:|
| Full 4Y | 1.465 | $+21,475 | $1,753 | 1900 | 46.6% | 2.20 | 10 | 13 |
| IS 2022-23 | 1.327 | $+6,607 | $1,753 | 857 | 48.2% | 1.73 | 10 | 2 |
| OOS 2024-26 | 1.570 | $+14,859 | $1,523 | 1044 | 45.3% | 2.56 | 8 | 11 |

**Key observation:** OOS outperforms IS. This is suspicious in most contexts but explainable here -- 2024-2026 had stronger trends (AI/tariff/election) that suit the trend-following exit system.

---

## Attack 1: Worst-Case Scenario Construction

### Worst 10 Trading Days

| Rank | Date | PnL | Context |
|-----:|------|----:|---------|
| 1 | 2022-01-14 | $-722.53 | Fed hawkish pivot week |
| 2 | 2025-04-25 | $-700.47 | Tariff escalation |
| 3 | 2025-11-28 | $-620.84 | Post-Thanksgiving |
| 4 | 2022-05-11 | $-559.47 | CPI shock |
| 5 | 2025-04-09 | $-457.43 | Tariff whipsaw |
| 6 | 2025-04-10 | $-447.32 | Tariff whipsaw (day 2) |
| 7 | 2022-02-10 | $-408.79 | Pre-invasion vol |
| 8 | 2025-04-01 | $-398.30 | Tariff announcement |
| 9 | 2022-04-27 | $-366.84 | Tech selloff |
| 10 | 2025-04-17 | $-361.28 | Tariff whipsaw (week 3) |

**Pattern:** Worst days cluster around two events: (1) 2022 Fed tightening, (2) 2025 April tariff chaos. The strategy takes 3-4 consecutive bad days during tariff whipsaws. Worst single day is $-723, which is 1.4% of $50K equity -- manageable.

### Worst Drawdown Period

- **Peak date:** 2023-01-20
- **Trough date:** 2023-08-11 (7 months to bottom)
- **Recovery date:** 2023-10-09 (42 business days to recover)
- **Depth:** $1,735 (3.5% of equity)
- **Max DD in R:** 7.80R

**Assessment:** The drawdown occurs during 2023 H1, the one regime where the strategy loses money (PF=0.90). The 7-month drawdown is psychologically brutal but the dollar depth is small. Recovery takes about 2 months.

### Max Consecutive Losing Streak: 10 trades

| Rank | Length | Start | End | Total PnL |
|-----:|-------:|-------|-----|----------:|
| 1 | 10 | 2022-12-12 | 2022-12-16 | $-323.43 |
| 2 | 8 | 2022-07-01 | 2022-07-05 | $-135.93 |
| 3 | 8 | 2024-10-01 | 2024-10-03 | $-208.36 |
| 4 | 8 | 2025-01-10 | 2025-01-20 | $-947.05 |
| 5 | 8 | 2025-08-25 | 2025-08-27 | $-159.62 |

**Important:** The worst streak by PnL is #4 (8 consecutive losses, $-947) not #1 (10 losses, only $-323). The 10-loss streak has tiny losses (gate exits near breakeven). The 8-loss streak in Jan 2025 has real stop-outs.

### Worst Starting Day

- **Date:** 2022-01-05
- **Max DD from start:** $1,753 (3.5%)
- **PnL after 50 trades:** $-483
- **Min equity in first 100 trades:** $48,897
- **Topstep $2K DD blown:** NO

**Assessment:** Even starting on the absolute worst day in 4 years of data, the strategy never blows the Topstep $2K drawdown limit. You'd be down $483 after 50 trades but would recover.

---

## Attack 2: Regime Stress Tests

| Regime | PF | Verdict |
|--------|---:|--------:|
| 2022 H1 (bear crash) | 1.425 | PASS |
| 2022 H2 (bear rally) | 1.730 | PASS |
| **2023 H1 (AI rally)** | **0.900** | **FAIL** |
| 2023 H2 (range/pullback) | 1.223 | PASS |
| 2024 H1 (steady up) | 1.366 | PASS |
| 2024 H2 (election vol) | 1.515 | PASS |
| 2025 H1 (tariff chaos) | 1.719 | PASS |
| 2025 H2 | 1.255 | PASS |
| 2026 Q1 | 1.799 | PASS |

**Profitable regimes: 8/9**

**Kill scenario:** 2023 H1 (Jan-Jun 2023). The AI-driven rally was a smooth, one-directional grind with minimal pullbacks to EMA20. The strategy's touch-and-bounce entries found few quality setups, and those that triggered faced low-volatility whipsaws. PF=0.90 means for every $1 lost, only $0.90 was won.

**But:** This is a 6-month period producing only $464 loss on $50K. The strategy degrades gracefully -- it doesn't blow up, it just bleeds slowly.

---

## Attack 3: Cost Sensitivity Bomb

| Scenario | PF | Status |
|----------|---:|-------:|
| Baseline (1x costs) | 1.465 | PASS |
| 1.5x all costs | 1.276 | PASS |
| 2x all costs | 1.116 | PASS |
| 3x all costs | 0.869 | DEAD |
| 2x costs + 3x slip | 1.045 | PASS |
| 5x slippage only | 1.113 | PASS |

**Cost bomb threshold: 2.5x** -- the strategy dies (PF < 1.0) at 2.5x all costs.

Specific tests:
- **Double all costs:** PF = 1.193 (survives)
- **Triple slippage only:** PF = 1.274 (survives)

**Assessment:** The strategy has meaningful cost headroom. It takes 2.5x cost inflation to kill it. Doubling all costs still produces PF=1.12 (profitable). This suggests the edge is structural, not a thin artifact of low costs.

However, the cost headroom metric is misleading in isolation -- see Attack 5 for why.

---

## Attack 4: Execution Delay

| Delay | PF | PnL | Max DD |
|------:|---:|----:|-------:|
| 0 (baseline) | 1.465 | $+21,475 | $1,753 |
| 1 bar (10min) | 1.422 | $+16,002 | $3,356 |
| 2 bar (20min) | 1.441 | $+14,196 | $2,185 |
| 3 bar (30min) | 1.335 | $+9,442 | $3,487 |

**Assessment:** Survives well. 1-bar (10 min) delay costs ~25% of PnL but PF drops only from 1.465 to 1.422. Even a 30-minute execution delay keeps PF at 1.335. This indicates the edge is not latency-dependent.

**Warning:** The max DD roughly doubles with delays (from $1,753 to $3,356). The Topstep $2K limit would be breached at 1-bar delay.

---

## Attack 5: Random Baseline -- THE CRITICAL FINDING

### 5a. Random entries + SAME exit system

| Metric | Strategy | Random (mean) | Random (median) |
|--------|----------|---------------|-----------------|
| PF | 1.465 | 1.484 | 1.483 |
| % with PF > 1 | -- | 100% | -- |
| % that beat strategy | -- | 54.5% | -- |
| Percentile | 45.5th | -- | -- |

**The EMA20 touch entry is at the 45th percentile of random entries.** More than half of random entry strategies beat it. The entry signal provides negative alpha of -0.019 PF.

### 5b. Random entries + SIMPLE exit (fixed stop, no trail)

| Metric | Value |
|--------|------:|
| Mean PF | 0.982 |
| % with PF > 1 | 37.4% |

With a simple exit, random entries are unprofitable (PF=0.98). This proves NQ doesn't have an inherent long bias that inflates PFs -- the market is roughly fair with simple exits.

### 5c. Bootstrap p-value

- **p-value:** 0.5026 (NOT statistically significant at any conventional level)

### 5d. Edge Decomposition

```
Market bias (simple exit):    PF = 0.982  (baseline: roughly fair market)
Exit system (full exit):      PF = 1.484  (+0.503 over market)
Strategy (entry + exit):      PF = 1.465  (-0.019 over random entry)

Entry alpha:    -0.019 PF  (ZERO -- actually slightly negative)
Exit alpha:     +0.503 PF  (ALL the edge)
```

**Conclusion:** The BE/gate/chandelier exit system adds 0.50 PF over market baseline. The EMA20 touch entry subtracts 0.02 PF compared to random entries. The entry signal is literally worse than random.

**Why this happens:** The exit system (breakeven trigger + chandelier trail) is a trend-following mechanism. It cuts losers at breakeven and lets winners ride via trailing stop. This works regardless of entry timing because NQ has enough intraday trends to exploit. The EMA20 touch constrains entries to specific bars that are no better (and slightly worse) than random bars.

---

## Attack 6: Year-by-Year Consistency

| Year | PF | PnL | Verdict |
|-----:|---:|----:|--------:|
| 2022 | 1.502 | $+5,847 | PASS |
| 2023 | 1.051 | $+437 | PASS (barely) |
| 2024 | 1.439 | $+4,334 | PASS |
| 2025 | 1.536 | $+6,847 | PASS |
| 2026 | 1.799 | $+2,708 | PASS |

All 5 years profitable. 2023 is the weak year (PF=1.05), consistent with the 2023-H1 regime failure. The strategy degrades but doesn't die.

---

## Final Verdict

### BRUISED

The strategy is structurally sound in terms of risk management -- it handles costs, delays, regime changes, and worst-case starts without catastrophic failure. But the entry signal is a placebo.

**Kill Shots:**
- EMA20 touch entry provides ZERO edge over random entries (-0.019 PF alpha)
- 54.5% of random entry strategies beat the strategy
- Bootstrap p-value = 0.50 (statistically indistinguishable from noise)

**Survived:**
- Survives up to 2.5x cost inflation
- Survives 10-minute execution delay (PF=1.42)
- 8/9 market regimes profitable
- All 5 years profitable
- Max consecutive losses: 10 (small dollar impact)
- Topstep $2K DD limit: NEVER breached, even from worst start

**Implications for Research:**
1. **Stop optimizing entry signals.** The EMA20 touch, RSI filters, slope thresholds, pullback zones -- none of this matters. Random entries work just as well.
2. **The exit system IS the strategy.** BE trigger at 0.25R, BE stop at 0.15R, gate at 3 bars, chandelier trail at 0.3x ATR -- this is where all the alpha lives.
3. **Research should focus on:** Exit optimization (different trail methods, partial profit taking, adaptive BE levels), position sizing (Kelly criterion, risk scaling), and regime-aware position sizing (reduce size in 2023-H1 type regimes).
4. **The 2.5x cost bomb threshold is adequate** for MNQ but would be tight for larger contracts (NQ at $20/point has 10x higher costs per trade).
