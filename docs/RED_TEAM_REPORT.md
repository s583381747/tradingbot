# RED TEAM REPORT -- Adversarial Audit of Baseline Strategy

**Agent:** red-critic
**Date:** 2026-03-26
**Target:** strategy_mnq.py v11 (EMA20 Touch Close, 10min, MNQ x2)
**Claimed Performance:** NQ 4Y PF=1.465, DD=$1,753, $26/day, Sharpe=2.20

---

## EXECUTIVE SUMMARY

The baseline strategy has **real structural edge** but the backtest infrastructure contains **critical cost model bugs** in multiple experiment files that inflated validation results. The primary strategy file (strategy_mnq.py, run_nq_backtest.py) has correct per-contract costs, but 6+ experiment files used for walk-forward validation and tuning do NOT scale spread and slippage with contract count. This means all experiment results that used dynamic sizing (nc > 1) systematically undercharged costs.

**Overall Verdict: CONDITIONAL PASS** -- the core strategy logic is sound but the supporting evidence is corrupted by cost model inconsistencies.

---

## 1. CODE AUDIT

### 1.1 Look-ahead Bias Check

| Check | Status | Details |
|-------|--------|---------|
| EMA uses only past data | PASS | `ewm(adjust=False)` is causal |
| ATR uses only past data | PASS | `rolling(14).mean()` is causal |
| Entry on Close of signal bar | PASS | `entry = close[bar]` -- entering at close, next bar execution is implicit |
| Stop based on current bar | PASS | `stop = low[bar] - buffer * atr` -- uses current (completed) bar |
| Indicators computed before loop | PASS | All indicators precomputed on full dataset but this is fine for causal indicators |

**FINDING: PASS** -- No look-ahead bias detected in signal generation.

### 1.2 Same-bar Entry+Exit

**FINDING: PASS** -- The trade loop starts at `k=1` (line 209 of strategy_mnq.py: `for k in range(1, max_hold + 1)`), so exit checking begins on the bar AFTER entry. No same-bar issue.

### 1.3 Cost Model Audit -- CRITICAL BUGS FOUND

#### 1.3.1 strategy_mnq.py (PRIMARY FILE) -- CORRECT
```python
# Line 198: entry_cost = COMM_PER_CONTRACT_RT * active_nc / 2 + SPREAD_PER_CONTRACT * active_nc
# Line 280: exit_comm = COMM_PER_CONTRACT_RT * active_nc / 2
# Line 281: exit_slip = STOP_SLIP_PER_CONTRACT * active_nc if exit_reason in ("stop", "trail") else 0
# Line 282: be_slip = BE_SLIP_PER_CONTRACT * active_nc if exit_reason == "be" else 0
```
All costs scale with `active_nc`. CORRECT.

#### 1.3.2 run_nq_backtest.py -- CORRECT
```python
# Line 131: entry_cost = COMM_PER_CONTRACT_RT * nc / 2 + SPREAD_PER_CONTRACT * nc
# Line 190: exit_slip = STOP_SLIP_PER_CONTRACT * nc
```
CORRECT.

#### 1.3.3 exp_nq_walkforward.py -- BUG: SPREAD NOT PER CONTRACT
```python
# Line 96: ec = COMM_RT * nc_dyn / 2 + SPREAD    # <-- SPREAD is flat $0.50, NOT * nc_dyn
# Line 138: xs = STOP_SLIP if ex in ("stop", "trail") else 0    # <-- flat $1.00, NOT * nc_dyn
# Line 139: bs = BE_SLIP if ex == "be" else 0    # <-- flat $1.00, NOT * nc_dyn
```
**IMPACT:** With dynamic sizing (avg nc ~5-8 contracts), this undercharges:
- Spread: $0.50 instead of $2.50-4.00 (5-8x undercount)
- Stop slip: $1.00 instead of $5.00-8.00 (5-8x undercount)
- BE slip: $1.00 instead of $5.00-8.00 (5-8x undercount)

This file was used for the "7/7 walk-forward periods profitable" claim. **That result is INVALID for dynamic sizing runs.**

#### 1.3.4 exp_nq_stats.py -- SAME BUG
```python
# Line 87: ec = COMM_RT * nc / 2 + SPREAD    # flat spread
# Line 129: xs = STOP_SLIP if ex in ("stop", "trail") else 0    # flat slippage
```
This file produced the "Sharpe=2.20" headline number. **With nc > 1, the Sharpe is inflated.**

#### 1.3.5 exp_nq_tune.py -- SAME BUG
```python
# Line 85: ec = COMM_RT * nc / 2 + SPREAD    # flat spread
# Line 128: xs = STOP_SLIP if ex in ("stop", "trail") else 0    # flat slippage
```
This file was used to tune parameters (gate_tighten, be_trigger_r, etc.). **Parameters were tuned with understated costs.**

#### 1.3.6 exp_nq_audit.py -- SAME BUG
```python
# Line 56: ec = COMM_RT * nc / 2 + SPREAD    # flat
# Line 102: xs = STOP_SLIP if ex in ("stop", "trail") else 0    # flat
```
The audit that validated gate_tighten=0.0 used incorrect costs.

#### 1.3.7 exp_nq_realistic.py -- SAME BUG
```python
# Line 117: ec = COMM_RT * nc / 2 + SPREAD    # flat
# Line 159: xs = STOP_SLIP if ex in ("stop", "trail") else 0    # flat
```
The "Topstep 50K simulation" that was supposed to be the realistic validation used incorrect costs for dynamic sizing runs.

**SEVERITY: P0 (CRITICAL)**

When nc=2 (fixed in strategy_mnq.py), the undercharge per trade is:
- Spread: $0.50 missing (should be $1.00 total, charged $0.50)
- Stop slip: $1.00 missing on stop exits
- Total undercharge: ~$1.00-1.50 per trade

When nc=5 (dynamic sizing in experiments), the undercharge per trade is:
- Spread: $2.00 missing (should be $2.50 total, charged $0.50)
- Stop slip: $4.00 missing on stop exits
- Total undercharge: ~$4.00-6.00 per trade

With ~600+ trades over 4 years, this is ~$2,400-3,600 of phantom profit at nc=5.

**Note:** For the fixed nc=2 case in strategy_mnq.py itself, the cost model IS correct. The baseline PF=1.465 from run_nq_backtest.py is valid since that file has correct per-contract costs. But ALL dynamic-sizing experiment results (walk-forward, realistic, stats) are inflated.

### 1.4 Stop Fill Model

#### strategy_mnq.py (Lines 236-243) and run_nq_backtest.py (Lines 155-160):
```python
# Gap-through: if bar opened past stop, fill at open (worse)
if trend == 1:
    fill_price = min(runner_stop, opn[bi]) if opn[bi] < runner_stop else runner_stop
else:
    fill_price = max(runner_stop, opn[bi]) if opn[bi] > runner_stop else runner_stop
```
**FINDING: PASS (with caveats)** -- This correctly models gap-through: if a bar opens below your stop (longs), you get filled at the open, not the stop. This is realistic for stop-market orders.

**CAVEAT 1:** The experiment files (exp_nq_walkforward.py, exp_nq_stats.py, exp_nq_tune.py, exp_nq_audit.py) do NOT have gap-through logic. They use:
```python
r = (rs - entry) / rp * tr    # fills exactly at stop price
```
This is optimistic -- in real markets, gap-through fills can be significantly worse than the stop price.

**CAVEAT 2:** There is no modeling of:
- Limit-up / limit-down (LULD) pauses where stop orders queue but don't fill
- Trading halts where you cannot exit at all
- Flash crash scenarios with multiple points of slippage beyond 1 tick

### 1.5 Resample Function

```python
def resample(df, minutes):
    if minutes <= 1:
        return df
    return df.resample(f"{minutes}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()
```

**FINDING: POTENTIAL ISSUE** -- `pd.resample` uses fixed time boundaries (e.g., 09:30, 09:40, 09:50 for 10min bars), NOT rolling from market open. This means:
1. The first bar of each day may have fewer minutes than expected (market opens at 09:30 ET, but the 10min resample boundary is 09:30 so this is actually OK for RTH data starting at 09:30).
2. The last bar of the day (15:50-16:00) is only 10 minutes if market closes at 16:00. But force_close is at 15:58, so some trades close within the last 10min bar before it completes. **This is fine** -- the strategy checks `times[bi] >= force_close_at` within the bar loop.

**Concern:** The `dropna()` call removes bars where OHLCV is NaN. At market boundaries (overnight gaps in 24h data, or missing data), this could create bars that span non-contiguous time periods. Since we use RTH-only data, this is less of a concern, but **any 10min bar that spans a data gap will have distorted OHLCV**.

### 1.6 Entry Logic

```python
# Line 145-150: Trend filter
if c > ema_f[bar] and ema_f[bar] > ema_s_arr[bar]:
    trend = 1
elif c < ema_f[bar] and ema_f[bar] < ema_s_arr[bar]:
    trend = -1
```

**FINDING: ACCEPTABLE** -- This is a standard EMA crossover trend definition. No look-ahead.

```python
# Line 153-158: Touch detection
if trend == 1:
    touch = low[bar] <= ema_f[bar] + tol and low[bar] >= ema_f[bar] - a * touch_below_max
```

**FINDING: SUBTLE ISSUE** -- The touch condition uses `low[bar]` which is the LOW of the current (closed) bar. The entry is at `close[bar]`. This means you're entering at the close AFTER confirming the low touched the EMA during the bar. In real-time, you would NOT know the bar's low until the bar closes, so entering at the close is realistic. **PASS**.

However, in a 10-minute bar regime, this means waiting up to 10 minutes after the touch to enter. In fast-moving markets, the close could be significantly above the EMA touch point, increasing the effective stop distance. This is actually conservative, so **PASS**.

---

## 2. STATISTICAL ATTACK

### 2.1 Sample Size

**4Y NQ data (2022-2026):** ~1,060 trading days
**Trade count at 10min, nc=2:** Likely ~500-700 trades (based on ~0.5 trades/day)

For a strategy with ~12 tunable parameters, 500-700 trades is borderline. Using the rule of thumb of 30 trades per degree of freedom:
- 12 params x 30 = 360 trades minimum
- We likely exceed this, so **MARGINAL PASS**.

However, several parameters (gate_tighten, be_trigger_r, chand_mult) were tuned on this same 4Y dataset. The effective degrees of freedom are higher.

### 2.2 Multiple Comparison Problem

From exp_nq_audit.py and the git log:
- gate_tighten was tested at: -0.1, -0.05, 0.0, +0.05, +0.1 (5 values)
- gate_mfe was tested at: 0.15, 0.2, 0.25, 0.3 (4 values)
- be_trigger_r was tested at: 0.15, 0.2, 0.25, 0.3 (4 values)
- skip_after_win: 0, 1, 2, 3 (4 values)
- tf_minutes: 1, 3, 5, 10, 15 (5 values)

Minimum combinations tested: 5 x 4 x 4 x 4 x 5 = 1,600 implicit comparisons

**Bonferroni correction at alpha=0.05:** significance level = 0.05 / 1600 = 0.000031

The claimed PF improvement from V8 (gate=-0.1) to V11 (gate=0.0, 10min) is meaningful (PF 1.19 -> 1.54), but:
- Most of this comes from tf_minutes change (3min -> 10min), which reduces cost/risk ratio from 8.4% to 4.4%
- This is a REAL effect (fewer trades = lower total cost) not a statistical artifact
- The gate_tighten=0.0 change alone is smaller and IS within noise range for small sample sizes

**FINDING: PARTIAL PASS** -- The timeframe change is mechanically sound (lower cost impact). The gate change needs more out-of-sample validation.

### 2.3 IS/OOS Integrity

The project has used multiple IS/OOS split definitions:
1. `strategy_mnq.py`: IS = QQQ 2024-2026, OOS = QQQ 2022-2024
2. `run_nq_backtest.py`: IS = NQ 2024-2026, OOS = NQ 2022-2024
3. `exp_nq_realistic.py`: IS = NQ 2022-2023, OOS = NQ 2024-2026
4. `exp_nq_audit.py`: IS = NQ 2024+, OOS = NQ 2022-2024

**FINDING: FAIL** -- The IS/OOS splits have been flipped between experiments. In exp_nq_audit.py, the "OOS" is 2022-2024 and "IS" is 2024+. But in exp_nq_realistic.py, it's reversed: IS is 2022-2023 and OOS is 2024-2026. This means parameters were effectively tuned looking at BOTH halves of the data. The "OOS" was never truly out of sample.

The walk-forward test (exp_nq_walkforward.py) helps mitigate this since it uses rolling windows, but the claim "7/7 periods profitable" was made with incorrect costs (see Section 1.3).

### 2.4 Walk-forward Validity

The walk-forward test uses 12-month train / 6-month test windows, but with **fixed parameters** (no re-optimization per window). This is called "anchored walk-forward" -- it tests robustness of fixed params, not adaptability.

With only 7 test periods of 6 months each, the probability of all 7 being profitable by chance (assuming 50% base rate) is 0.5^7 = 0.78%. This is statistically significant.

However, with cost bugs (Section 1.3), some marginal periods may flip from profitable to unprofitable with correct costs. **NEEDS REVALIDATION.**

---

## 3. OVERFITTING DETECTION

### 3.1 Parameter Count

The strategy has the following tunable parameters:
1. tf_minutes (10)
2. ema_fast (20)
3. ema_slow (50)
4. atr_period (14)
5. touch_tol (0.15)
6. touch_below_max (0.5)
7. stop_buffer (0.4)
8. gate_bars (3)
9. gate_mfe (0.2)
10. gate_tighten (0.0)
11. be_trigger_r (0.25)
12. be_stop_r (0.15)
13. chand_bars (25)
14. chand_mult (0.3)
15. daily_loss_r (2.0)
16. skip_after_win (1)
17. no_entry_after (14:00)
18. force_close_at (15:58)
19. max_hold_bars (180)

**19 parameters** for ~600 trades = 31.6 trades per parameter. This is right at the minimum acceptable threshold.

### 3.2 Parameter Sensitivity (from exp_nq_audit.py analysis)

The audit tested small perturbations around V9 params on OOS:
- gate_tighten: -0.02 to +0.02 -- reportedly stable
- gate_mfe: 0.22 to 0.28 -- reportedly stable
- be_trigger: 0.18 to 0.22 -- reportedly stable

**FINDING: NEED TO VERIFY** -- These sensitivity tests were run with incorrect costs. Need revalidation.

### 3.3 Cross-instrument Validation

QQQ data was tested in walk-forward (exp_nq_walkforward.py). QQQ is a strong cross-validation because:
- Different data source (Barchart vs Polygon)
- ETF vs futures (slightly different microstructure)
- QQQ uses x40 multiplier proxy for NQ

If results hold on both NQ and QQQ, this is genuine evidence of robustness. **CONDITIONAL PASS** (pending cost fix revalidation).

---

## 4. MARKET REALITY ATTACK

### 4.1 Execution Feasibility

| Aspect | Assessment |
|--------|-----------|
| Signal timing | PASS -- enter at bar close, which in 10min bars gives clear signal |
| Order type | Stop-market for exits -- realistic for MNQ |
| Entry order | Market order at bar close -- slight slippage not modeled |
| Latency | PASS for 10min bars -- plenty of time to compute and send order |

**FINDING: MOSTLY PASS** -- But entry slippage is zero in the model. At the close of a 10-min bar, there is some rush as other algos react. A conservative $0.25-0.50 per contract entry slippage should be added (currently only spread is charged).

### 4.2 Extreme Events

**Flash Crash (e.g., 2010, 2015 Aug, 2020 March):**
- The force_close_at=15:58 helps limit overnight risk
- daily_loss_r=2.0 limits intraday loss to 2R
- But NO handling of trading halts or LULD pauses
- During a halt, your stop order sits in the queue. When trading resumes, the fill price could be MUCH worse than the stop price. The gap-through model only considers the bar's open, but in a halt scenario, the fill could gap 10+ points past the stop.

**FINDING: FAIL** -- No halt/LULD modeling. Impact: in the 4-year backtest period, there were multiple NASDAQ halt events (March 2020, various single-stock halts that froze the index). These should produce worse fills than modeled.

### 4.3 Liquidity

**MNQ typical volume:** 100K-200K contracts/day
**Strategy trades:** 2 MNQ contracts, ~0.5 trades/day
**Market impact:** NEGLIGIBLE for 2 contracts. Even at 10 contracts with dynamic sizing, MNQ has sufficient liquidity.

**FINDING: PASS**

### 4.4 Slippage Model

Current model:
- Entry: $0.50/contract spread (1 tick at $0.50/tick for MNQ)
- Stop exit: $1.00/contract (2 ticks)
- BE exit: $1.00/contract
- Market close exit: $0.00 slippage

**Critique:**
1. **Stop slippage of 2 ticks is reasonable** for MNQ in normal conditions. Real-world data (comment in code: "Tradovate MNQ mean 1.94 ticks = $0.97") supports this.
2. **Market close exit at zero slippage is optimistic.** At 15:58, many traders are closing positions. A $0.25-0.50/contract slippage should apply.
3. **Entry spread of 1 tick is reasonable** for limit orders but optimistic for market orders. If entering with a market order at bar close, 1-2 ticks of slippage is more realistic.

**FINDING: MARGINAL** -- The model is acceptable for conservative estimates with 2 MNQ. For 5-10 contracts, the impact of undermodeled slippage compounds.

---

## 5. REGIME DEPENDENCY

### 5.1 Market Regime Coverage

The 4-year dataset (2022-2026) covers:
- **2022 H1:** Bear market (NQ -30%)
- **2022 H2:** Range/recovery
- **2023:** Bull recovery
- **2024:** Strong bull
- **2025:** Mixed/volatile (tariffs, AI rotation)
- **2026 Q1:** Recovery

This is reasonable regime diversity. The strategy trades both long and short, which should help in bear markets.

### 5.2 Regime-by-Regime Performance

From exp_nq_audit.py year-by-year analysis (V8 with gate=-0.1):
- The strategy was reportedly profitable in all years
- But these results use incorrect costs

**FINDING: NEEDS REVALIDATION** with correct per-contract costs.

### 5.3 Edge Decay

The strategy was originally developed on QQQ data (2024-2026 IS), then validated on NQ and QQQ OOS (2022-2024). If the edge is decaying, recent periods should show worse performance.

The walk-forward claim of 7/7 profitable periods suggests no decay, but this needs cost-corrected revalidation.

---

## 6. SPECIFIC BUGS AND ISSUES

### BUG-1: Cost model inconsistency (CRITICAL, P0)

**Files affected:** exp_nq_walkforward.py, exp_nq_stats.py, exp_nq_tune.py, exp_nq_audit.py, exp_nq_realistic.py, exp_vol_sr_mnq.py, exp_ml_filter_v2.py, exp_ml_filter_v3.py, exp_reentry_mnq.py

**Pattern:** `ec = COMM_RT * nc / 2 + SPREAD` where SPREAD=$0.50 is flat, not `SPREAD * nc`.
Similarly: `xs = STOP_SLIP` is flat $1.00, not `STOP_SLIP * nc`.

**Fix:** Replace with:
```python
ec = COMM_RT * nc / 2 + SPREAD * nc
xs = STOP_SLIP * nc if ex in ("stop", "trail") else 0
bs = BE_SLIP * nc if ex == "be" else 0
```

**Impact:** All walk-forward, sensitivity, audit, and realistic simulation results done with these files are invalid for nc > 1.

### BUG-2: Missing gap-through in experiment files (HIGH, P1)

**Files affected:** exp_nq_walkforward.py, exp_nq_stats.py, exp_nq_tune.py, exp_nq_audit.py

**Pattern:** Stop fills at exact stop price:
```python
r = (rs - entry) / rp * tr
```
Should use gap-through model:
```python
if tr == 1:
    fill = min(rs, opn[bi]) if opn[bi] < rs else rs
else:
    fill = max(rs, opn[bi]) if opn[bi] > rs else rs
r = (fill - entry) / rp * tr
```

**Impact:** Overstates stop-loss fill quality. In volatile periods (2022), gap-through events are more common. This likely inflates OOS performance during the bear market period.

### BUG-3: Market close exit has zero slippage (LOW, P2)

**All files:** When `exit_reason == "close"`, no slippage or spread is charged:
```python
exit_slip = STOP_SLIP_PER_CONTRACT * active_nc if exit_reason in ("stop", "trail") else 0
```
Market orders to close at 15:58 should include at least entry-equivalent spread costs.

### BUG-4: Chandelier trail lookback off-by-one (LOW, P3)

```python
# Line 266-268 of strategy_mnq.py:
sk = max(1, k - chand_b + 1)
hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
```
The range `(sk, k)` excludes the current bar `k`, which means the trailing stop is based on the high of bars BEFORE the current bar. This is correct for preventing look-ahead, but the `max(1, k - chand_b + 1)` can produce windows shorter than chand_b in early bars after BE trigger. This is a minor edge case.

### ISSUE-5: EMA computed on full dataset before slicing (INFORMATIONAL)

In experiment files, data is loaded and then sliced into IS/OOS periods:
```python
nq_is = nq[nq.index >= "2024-01-01"]
nq_oos = nq[(nq.index >= "2022-01-01") & (nq.index < "2024-01-01")]
```
Then indicators are computed within each slice. This means the first ~50 bars of each slice have EMA warming up from scratch. For IS starting at 2024-01-01, the EMA20 at bar 1 is just the first close, not a true 20-period average.

**Impact:** Minimal -- the strategy skips the first `max(ema_slow, atr_period) + 5 = 55` bars. But for walk-forward windows starting mid-data, the EMA warm-up could miss the first week of trades.

### ISSUE-6: Resample may create phantom bars at day boundaries (INFORMATIONAL)

The `dropna()` call in resample removes bars with NaN values. If there's a gap in the data (missing minutes), the resampled bar will have NaN for Close (since "last" of an empty set is NaN). This is handled correctly by dropna, but the resulting bar index may have gaps that are not obvious in the loop.

---

## 7. ATTACKS TO PREPARE FOR BLUE TEAM

### Attack A: Random Baseline Test

Generate 1000 random entry strategies with:
- Same exit logic (BE, trail, gate, force close)
- Random entry timing (same frequency as real strategy)
- Same cost model
- Measure what % beat the blue team's PF

If > 5% beat the strategy, the entry logic adds no value.

The existing random baseline (exp_1_2_random_baseline.py) used a DIFFERENT strategy version (old QQQ-based), different cost model, and different exit logic. **It needs to be redone** with the current v11 exit logic on NQ data.

### Attack B: Regime Split Test

Run each blue team strategy on isolated 6-month periods:
- 2022-H1 (crash), 2022-H2 (range), 2023-H1, 2023-H2, 2024-H1, 2024-H2, 2025-H1, 2025-H2

If any period has PF < 0.8, the strategy is regime-dependent.

### Attack C: Cost Sensitivity Analysis

Sweep total costs from 50% to 200% of current model:
- At what cost level does each strategy go to PF < 1.0?
- This reveals the "cost buffer" -- how much room there is before edge disappears.

### Attack D: Bootstrap Confidence Interval

Resample trades with replacement (1000 iterations), compute PF distribution:
- If the 5th percentile PF < 1.0, the strategy is not reliably profitable.

### Attack E: Monte Carlo Drawdown

Permute trade order 10,000 times:
- What is the 95th percentile max drawdown?
- Does it exceed the Topstep $2,000 trailing DD limit?

---

## 8. SCORECARD

| Category | Item | Status | Impact |
|----------|------|--------|--------|
| Code | Look-ahead bias | PASS | -- |
| Code | Same-bar entry+exit | PASS | -- |
| Code | Cost model (primary files) | PASS | -- |
| Code | Cost model (experiment files) | **FAIL** | P0: All dynamic-sizing results invalid |
| Code | Gap-through (primary) | PASS | -- |
| Code | Gap-through (experiments) | **FAIL** | P1: Inflated OOS results |
| Code | Close exit slippage | FAIL | P2: Minor undercharge |
| Stats | Sample size | MARGINAL PASS | Borderline at ~31 trades/param |
| Stats | Multiple comparison | MARGINAL PASS | TF change is mechanical, not statistical |
| Stats | IS/OOS integrity | **FAIL** | Both halves have been used for tuning |
| Stats | Walk-forward | NEEDS REVALIDATION | 7/7 claim uses buggy costs |
| Overfit | Param count | MARGINAL | 19 params for ~600 trades |
| Overfit | Param sensitivity | NEEDS REVALIDATION | Tested with buggy costs |
| Overfit | Cross-instrument | CONDITIONAL PASS | QQQ validation good, needs cost fix |
| Market | Execution feasibility | PASS | 10min bars, MNQ, simple orders |
| Market | Extreme events | **FAIL** | No halt/LULD modeling |
| Market | Liquidity | PASS | Negligible impact |
| Market | Slippage model | MARGINAL | Close exit slippage missing |
| Regime | Diversity | PASS | 4Y covers multiple regimes |
| Regime | Per-regime PF | NEEDS REVALIDATION | -- |
| Regime | Edge decay | NEEDS REVALIDATION | -- |

---

## 9. RECOMMENDATIONS

### Immediate (Before any blue team work):

1. **FIX BUG-1:** Correct all experiment files to use per-contract spread and slippage. This is a 1-line fix per file.
2. **FIX BUG-2:** Add gap-through fill model to all experiment files.
3. **REVALIDATE:** Re-run walk-forward, stats, and realistic tests with fixed costs. Report corrected PF and DD numbers.

### For Blue Team Validation:

4. **Mandate `src/backtest_engine.py` usage:** All blue team strategies MUST use the shared engine (which has correct costs) instead of copy-pasting trade loops with bugs.
5. **True holdout:** Reserve NQ 2025-H2 to 2026-Q1 as a final holdout that NO strategy touches until the end.
6. **Random baseline:** Run 1000 random entries with current exit logic on NQ 4Y to establish the null hypothesis.

### For Production Readiness:

7. **Add LULD/halt handling:** At minimum, add a worst-case slippage multiplier (3-5x) for bars where the open gaps > 2 ATR from previous close.
8. **Add market-close slippage:** Charge spread on "close" exits.
9. **Monte Carlo DD:** Before trading real money, verify 95th percentile DD < prop firm limit.

---

## 10. VERDICT ON BASELINE

**The strategy has genuine edge**, based on:
- Sound trend-following logic (EMA touch pullback is a well-documented pattern)
- Correct primary implementation (strategy_mnq.py, run_nq_backtest.py)
- Multiple timeframe validation (QQQ + NQ)
- Multiple regime coverage (2022-2026)

**But the supporting evidence is COMPROMISED** by:
- Cost model bugs in 9+ experiment files
- IS/OOS contamination (both halves used for tuning)
- Missing gap-through in validation experiments
- No extreme event modeling

**Bottom line:** PF=1.465 from `run_nq_backtest.py` (which has correct costs, correct gap-through, fixed nc=2) is the MOST TRUSTWORTHY number. All other numbers from experiment files need revalidation.

The bar for blue team strategies is: **beat PF=1.465 on NQ 4Y with correct per-contract costs, gap-through fills, and walk-forward validation showing no losing 6-month periods.**

---

---

## 11. AUDIT OF BLUE TEAM VALIDATION FRAMEWORK

The blue-backtest agent produced `src/walk_forward_validator.py` and `docs/BLUE_VALIDATION.md`. I audited both.

### 11.1 walk_forward_validator.py -- GOOD NEWS

The validator uses **NQ costs** (not MNQ):
```python
NQ_PT_VAL    = 20.0    # $20/point (NQ full contract)
COMM_RT      = 2.46
SPREAD       = 5.00    # $5.00/contract (vs $0.50 for MNQ)
SLIP_STOP    = 1.25    # $1.25/contract (vs $1.00 for MNQ)
SLIP_BE      = 1.25
```

And critically, the cost model is CORRECT -- all costs are per contract:
```python
# Line 174: ec = c_comm * nc / 2 + c_spread * nc    # spread * nc = CORRECT
# Line 249: xs = c_slip_s * nc if ex in ("stop", "trail") else 0    # slip * nc = CORRECT
# Line 250: bs = c_slip_b * nc if ex == "be" else 0    # slip * nc = CORRECT
```

Gap-through fills are implemented:
```python
# Lines 208-211: fill = min(rs, O[bi]) if O[bi] < rs else rs
```

**VERDICT: The validator framework is correctly implemented.** This is good engineering.

### 11.2 BLUE_VALIDATION.md -- RED FLAGS

The validation report shows:
- **OOS PF = 1.951, IS PF = 1.686** -- OOS is BETTER than IS. This is unusual.
- **OOS Sharpe = 3.68** -- Extremely high for any trading strategy.
- **Sensitivity: FRAGILE** (PF range 0.951 across +/-20% perturbations) -- This is a problem.

#### RED FLAG 1: OOS > IS (Negative Decay)

The OOS outperforms IS by 15.7%. This could mean:
1. The 2024-2026 period is inherently easier for this strategy (trending NQ)
2. Parameters were (perhaps unconsciously) tuned on the OOS period
3. Pure luck

Given that parameters were tuned on NQ data across multiple experiments that flipped IS/OOS definitions (see Section 2.3), option 2 is highly likely. This is not a validation pass -- it is a warning sign.

#### RED FLAG 2: Sensitivity is FRAGILE

PF range of 0.951 across +/-20% perturbations means some parameter combinations push PF below 1.0. For a strategy with 19 parameters, this is concerning. A truly robust strategy should have PF > 1.0 for ALL perturbations within +/-20%.

#### RED FLAG 3: NQ vs MNQ Cost Discrepancy

The validator uses NQ costs ($5.00 spread, $1.25 slip) with n_contracts=2, while the primary strategy file uses MNQ costs ($0.50 spread, $1.00 slip). Since the point value is also different ($20 vs $2), the cost-to-risk ratio differs:
- NQ at $20/pt: risk = $20 * stop_points * 2 contracts
- MNQ at $2/pt: risk = $2 * stop_points * 2 contracts

For the SAME underlying move, NQ risk is 10x MNQ risk, but NQ costs are 5x-10x MNQ costs. The cost/risk PERCENTAGE may be similar, but the absolute dollar impact is different.

**The validator's results are for NQ (1 full contract = 10 MNQ)**. This is NOT directly comparable to the MNQ strategy in strategy_mnq.py. The PF=1.465 from run_nq_backtest.py (MNQ costs, 2 contracts) is a different number from PF=1.951 in the validator (NQ costs, 2 contracts, which is equivalent to 20 MNQ).

This is a critical apples-to-oranges comparison that could mislead the team.

#### Walk-Forward Results

6/6 periods profitable with correct costs is solid evidence. However:
- All 6 periods are in 2023-2026, which is mostly a bull/recovery market
- We need the 2022-H1 crash period in the walk-forward test windows
- The walk-forward starts testing at 2023-01 (after 12-month training from 2022-01), so the 2022 crash was only used as training data, not tested

The yearly breakdown shows 2022 PF=1.818 and 2023 PF=1.495. 2023 is the weakest year, which aligns with it being a ranging market. This is useful but note that yearly != walk-forward (the yearly result includes all parameters being known in advance).

### 11.3 Conclusion on Blue Validation

The framework is well-built. The results are technically valid for NQ at NQ costs. But:
1. The OOS > IS anomaly is a red flag
2. Parameter sensitivity is FRAGILE
3. Results are not directly comparable to MNQ strategy claims
4. Walk-forward does not test the worst period (2022-H1 crash)

---

## 12. FINAL KILL LIST (for blue team strategies, when they arrive)

Any blue team strategy will be subjected to ALL of the following. Failure on ANY is a kill:

1. **Cost model correctness** -- must use per-contract costs, must include gap-through
2. **IS/OOS integrity** -- must use 2022-2023 as IS, 2024-2026 as OOS, no peeking
3. **Walk-forward** -- 6+ periods, ALL profitable
4. **Sensitivity** -- PF range < 0.3 across +/-20% perturbations
5. **Survives 2x costs** -- PF > 1.0 at double the cost model
6. **Random baseline** -- must beat 95th percentile of 1000 random entry strategies
7. **Regime split** -- PF > 1.0 in every isolated 6-month period including 2022-H1
8. **No IS/OOS contamination** -- must not have been tuned on OOS data

**The bar is high because it should be.**

---

*Red Critic signing off. If everything passes, I haven't tried hard enough.*
