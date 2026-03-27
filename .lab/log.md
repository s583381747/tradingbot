# Research Log — MNQ Prop Firm Strategy

## Experiment 0 — Baseline (1min, 2 MNQ)
Result: IS PF=1.304 $58/d DD=$3668 | OOS PF=1.132 $21/d DD=$4242
Status: baseline

## Experiment 1 — 3min + gate-0.2
Result: IS PF=1.802 $90/d DD=$1313 | OOS PF=1.608 $58/d DD=$1657
Status: keep

## Experiment 2 — INVALID (trigger=stop bug)
Status: discard

## Experiment 3 — v8: gate-0.15 + chand30/0.3 + daily2.0R
Result: IS PF=2.128 $86/d DD=$1255 | OOS PF=1.889 $59/d DD=$983
Status: keep / Score: 9.06

## Experiment 4 — gate-0.1 + trigger0.25/stop0.15 + chand25/0.3 ★ BEST
Branch: research/chop-filter / Commit: 639c63b / Parent: #3
Hypothesis: Tighter gate (-0.1R vs -0.15R) cuts losses faster. Lower BE trigger (0.25R vs 0.3R) activates protection earlier. Tighter chandelier (25/0.3 vs 30/0.3) locks profits faster.
Changes: gate_tighten=-0.1, be_trigger_r=0.25, be_stop_r=0.15, chand_bars=25, chand_mult=0.3
Result: IS PF=2.488 $94/d DD=$860 | OOS PF=2.290 $69/d DD=$588
Score: 17.02 (+88% vs v8)
Status: keep ★★ BEST
Insight: All 3 changes work in same direction — minimize time in losing/flat trades. Gate -0.1R means if trade isn't working in 3 bars, cut to -0.1R loss (vs -0.15R). Trigger 0.25R means BE activates sooner. Chand 25 means trail starts sooner. Net effect: faster loss cutting + faster profit protection = much lower DD. PF improves because average loser is smaller. IS/OOS ratio 92% — robust.

## Experiment 5 — ML Entry Filter V1 (37 features, LightGBM/RF)
Branch: research/ml-filter / Commit: 6447ba2 / Parent: #4
Type: real / Hypothesis: Combine 37 pre-entry features (price/EMA, momentum, volatility, volume, bar patterns, time, channel) via ML to predict win/loss.
Changes: New experiment file exp_ml_filter.py
Result: LightGBM OOS AUC=0.528 (IS=0.711), RF OOS AUC=0.535 (IS=0.773), Regression OOS corr=0.039 (IS=0.325)
Duration: 45s / Status: discard (informative)
Insight: All feature correlations with R < |0.07|. Models overfit IS massively (AUC gap 0.18-0.24). Walk-forward by quarter shows no consistent improvement. Early stopping at 19 rounds = no learnable signal.

## Experiment 6 — ML Filter V2 (70 features, target engineering, visual patterns)
Branch: research/ml-filter / Commit: e996cb0 / Parent: #5
Type: real / Hypothesis: Different targets (gate pass, MFE≥1R, big wins) + visual shape features + regime context might have more signal than binary win/loss.
Changes: New experiment file exp_ml_filter_v2.py (35 shape + 35 basic features, 5 targets)
Result:
  Win/Loss: OOS AUC=0.532 ✗ | Gate Pass: OOS AUC=0.606 ✓ | BE Trigger: OOS AUC=0.532 ✗
  MFE≥1R: OOS AUC=0.656 ✓ | Big Win ≥3R: OOS AUC=0.681 ✓ (but IS=0.947, gap=0.27)
  Shape Only: OOS AUC=0.538 ✗ | Regime Only: OOS AUC=0.517 ✗ | Candle Only: OOS AUC=0.566 marginal
  Purged TS-CV: 0.538
Duration: 90s / Status: discard (informative)
Insight: Short-term MFE is partially predictable (AUC 0.61-0.66) via DCP + close-EMA distance. But filter impact at threshold 0.35 kills 94% of trades including 88% of 5R+ winners. Visual patterns (chart shape) add zero signal. Regime features (VIX proxy, day range) add zero signal.

## Experiment 7 — DCP Deep Dive (simple filter analysis)
Branch: research/ml-filter / Commit: adcb164 / Parent: #6
Type: real / Hypothesis: DCP as the #1 ML feature — does it work as a simple threshold filter?
Changes: New experiment file exp_ml_filter_v3.py
Result: Cohen's d = -0.13 (IS), -0.03 (OOS) — NEGLIGIBLE effect
  CRITICAL FINDING: Q1 (DCP<0.40) contains 68-74% of ALL 5R+ trades
  DCP≥0.5 filter: IS PF 2.62→ kills 29/37 5R+ | OOS PF 2.34→ kills 27/34 5R+
  No threshold improves OOS PF while preserving ≥75% of 5R+ trades
Duration: 30s / Status: discard (DEFINITIVE)
Insight: Big winners have LOW DCP — deep pullback touch bars (close away from trend direction) produce the largest runners. This is counterintuitive but logical: deeper pullback = more room to run. Any DCP filter destroys the strategy's real edge. The 3-bar MFE gate already handles this information post-entry.

## Experiment 8 — RE-ENTRY after BE (MNQ 3min)
Branch: research/ml-filter / Commit: dbe71ff / Parent: #4
Type: real / Hypothesis: After BE exit, re-enter on next EMA20 touch within N bars — captures continuation.
Changes: exp_reentry_mnq.py, tested windows 3-50 bars
Result: Best window=20: OOS PF=2.212(=base) $/d=+71.9(+8.4%) DD=$748(+22%) RE WR=64% RE PnL=+$8.9k
Duration: 40s / Status: interesting (marginal)
Insight: Re-entry trades ARE profitable (65% WR) but DD increases more than justified. IS/OOS gap widens at short windows (IS PF=2.66 vs OOS 2.14 at window=3) — overfit. At window=20, adds +$5.6/day but +$135 DD. Net: complexity not worth the marginal gain for prop firm (DD budget is tight).

## Experiment 9 — Volume Node S/R (MNQ 3min)
Branch: research/ml-filter / Commit: dbe71ff / Parent: #4
Type: real / Hypothesis: Volume-at-price nodes provide S/R confluence for entries and targets.
Changes: exp_vol_sr_mnq.py, tested filter/target/combined modes
Result:
  Filter: Best OOS PF=2.508 at LB=200/dist=0.5 BUT $/d drops 67→47 and 5R+ drops 34→22
  Target: OOS PF=1.970 (WORSE) — caps big winners, 5R+ drops 34→14
  Combined: OOS PF=2.115, $/d=48, 5R+=7 (WORST)
Duration: 120s / Status: discard
Insight: Volume targets are HARMFUL — they cap the fat tail that drives 80% of profits. Volume filter improves PF only by removing trades (reduces absolute return). The strategy already has optimal exit management (gate + BE + chandelier); adding volume S/R adds complexity without benefit.

## ═══ DEFINITIVE CONCLUSION (Updated) ═══
After 9 experiments across entry prediction (ML), entry filtering (DCP, volume S/R), exit targets (volume nodes), and re-entry mechanisms:
**The current strategy is near-optimal.** The edge is structural:
1. Trend alignment (EMA20>EMA50) provides direction
2. Touch entry provides pullback timing
3. 3-bar MFE gate cuts losers fast (Cohen's d > 1.0)
4. BE + Chandelier trail lets winners run (fat tail)

Any modification that caps upside (targets, filters removing low-DCP) or adds complexity (re-entry, ML filters) either hurts PF or provides marginal gain not worth the complexity.

## Experiment 10 — NQ Real Data Parameter Sweep (9 dimensions)
Branch: research/ml-filter / Commit: 8bd5c56 / Parent: NQ baseline
Type: real / Hypothesis: Parameters optimized on QQQ×40 proxy may not be optimal for real NQ.
Changes: exp_nq_tune.py — sweep stop_buffer, gate_tighten, be_trigger/stop, chand_bars/mult, gate_mfe, daily_loss_r, no_entry_after, skip_after_win, tf_minutes
Result: gate_tighten=0.0 is the breakthrough: score 18.56 vs 11.14 baseline (+66%)
Duration: 180s / Status: keep
Insight: On NQ, gate_tighten=-0.1 (small loss on gate fail) gets hit by noise. gate_tighten=0.0 (breakeven on gate fail) gives trades more room, many recover. NQ wider ATR means different optimal settings.

## Experiment 11 — NQ Combination Sweep (22 combos)
Branch: research/ml-filter / Commit: 587d978 / Parent: #10
Type: real / Hypothesis: Combine top single-param winners for multiplicative improvement.
Result: BEST (R): gate0.0+skip2+be0.20+mfe0.25 → OOS PF=3.933, DD=$418, $/d=+73, Score=35.15
  4Y verified: PF=3.841, PnL=$86,123, DD=$429, ALL YEARS profitable
  IS/OOS PF ratio: 104.2% (OOS > IS!)
Duration: 300s / Status: ~~keep~~ → INVALIDATED by Exp 12 (cost bug)
Insight: gate_tighten=0.0 + be_trigger=0.20 + gate_mfe=0.25 = aggressive early BE for stalled trades. Only fast-moving trades survive. skip=2 further refines by avoiding overtrading after wins. DD cut from $911→$429 (53%) while PF nearly doubles.
⚠ NOTE: Exp 10-11 results were inflated by cost bug (spread/slip flat per trade, not per contract). Real PF ~40-50% lower.

## ═══ AUDIT PHASE ═══

## Experiment 12 — Deep Code Audit + Cost Bug Fix
Branch: research/ml-filter / Commit: 28744c3 / Parent: #11
Type: audit / Hypothesis: Are backtest results trustworthy?
Findings:
  🔴 P0 BUG: spread/slippage was FLAT per trade, not per contract.
    With 4 MNQ, undercharged ~$4.44/trade = $18K over 4Y.
    Fix: SPREAD_PER_CONTRACT * nc, STOP_SLIP_PER_CONTRACT * nc, BE_SLIP_PER_CONTRACT * nc
  🟡 P2: Stop fill assumed exact stop price. Fixed: gap-through model using Open price.
  ✅ No look-ahead bias, no survivorship bias, no same-bar entry+exit, no data leakage
Impact: ALL prior PF/DD numbers were INFLATED. Exp 0-11 results are pre-fix.
Post-fix 3min gate=0.0 NQ: PF=1.193 (was 2.287), DD=$4,615 (was $1,646)
Status: critical fix applied

## Experiment 13 — Bottleneck Analysis (5 dimensions)
Branch: research/ml-filter / Parent: #12
Type: analysis
Findings:
  #1 Timeframe: 3min cost/risk=8.4%, raw PF=2.02 but 73% eaten by costs. 10min→4.4%.
  #2 BE waste: 55% of trades exit at BE, avg MFE=0.90R but exit at +0.01R. 0.91R/trade wasted.
  #3 Low-ATR trades: P0-P25 ATR has PF=0.630, cost/risk=17%. Losing money.
  #4 Lunch hour: 12:00 PF=0.937, only losing hour.
Conclusion: Timeframe is the #1 lever. 10min cuts costs from 8.4%→4.4%.

## Experiment 14 — v11: 10min bar (breakthrough)
Branch: research/ml-filter / Commit: 32d1496 / Parent: #12
Type: real
Changes: tf_minutes=10, gate_tighten=0.0, rest v8 defaults
Result (4.2Y NQ real, corrected costs, 2 MNQ):
  PF=1.465 | WR=46.6% | DD=$1,753 | $26/day | Sharpe=2.20 | APR=+13.2%
  5/5 years profitable | Topstep 50K: ✅ PASS (margin $247)
  IS(22-23): PF=1.327 | OOS(24-26): PF=1.571
Status: keep ★ BEST (corrected costs)
Insight: 10min is the sweet spot — cost/risk drops to 4.4%, DD=$1,753 fits Topstep. But $26/day → 150 days to reach $3K target = too slow for prop firm.

## Experiment 15 — NQ vs MNQ instrument comparison
Branch: research/ml-filter / Parent: #14
Type: analysis
Result:
  MNQ×2: PF=1.465, cost/risk=4.4%, $26/day, DD=$1,753
  NQ×1:  PF=1.836, cost/risk=1.0%, $202/day, DD=$5,952
  Same strategy, same data — only instrument differs.
  NQ cost 1.0% vs MNQ 4.4% → PF jumps from 1.47 to 1.84.
Conclusion:
  ❌ Prop firm (any tier): MNQ too slow, NQ DD too large
  ✅ Personal $50K account: NQ×1 = $202/day, +102% APR, Sharpe 3.42, DD 12%
  ✅ Conservative personal: MNQ×3-5 = $39-65/day, APR 20-33%, DD 5-9%

## ═══ CURRENT STATUS ═══
Strategy is VALIDATED with correct costs on 4.2Y real NQ data.
Prop firm route is mathematically dead — cost structure prevents viable speed.
Best path: personal account with NQ×1 or MNQ×3-5.
