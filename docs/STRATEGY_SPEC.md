# Strategy Specification — Active Strategies

## 1. EMA20 Touch v11 (Baseline)
- **Timeframe**: 10min NQ
- **Entry**: EMA20>EMA50 trend + pullback touch EMA20 + bar close entry
- **Gate**: 3-bar MFE < 0.2R → move stop to breakeven
- **BE**: price reaches +0.25R → stop to +0.15R
- **Trail**: Chandelier (25-bar highest high - 0.3×ATR) after BE
- **Cost/Risk**: 4.4% (MNQ×2), 1.0% (NQ×1)
- **Status**: Validated. Entry=zero alpha (red team confirmed). Edge is exit system.
- **4Y NQ×1**: PF=1.836, DD=$5,952, $202/day, Sharpe=3.42

## 2. Volatility Squeeze Breakout (Tournament Winner)
- **Timeframe**: 30min NQ
- **Entry**: EMA25/60 trend + pullback touch + vol compression gate (ATR pctile<45% AND BB width<55%)
- **Exit**: BE at 1R, Chandelier trail (3.0 ATR, 10-bar) at 1.5R+
- **Cost/Risk**: 0.6% — the key advantage
- **Status**: Two independent agents converged. Needs red team Kill Test.
- **4Y NQ×1**: PF=1.533, DD=$9,725, $66/day, Sharpe=2.33

## 3. MTF Pullback Short-Only (Under Investigation)
- **Timeframe**: Daily trend + 30min structure + 10min entry
- **Short side PF**: 1.68 on 98 trades
- **Status**: Needs more data. Short-only reduces trade count.

## Rejected Strategies
- ORB (8 variants): OOS decays in 2024-2025 regime
- Mean Reversion: PF=1.089, NQ is structural trending instrument
- ML Entry Filters: AUC=0.53, no predictive power
- Volume S/R Targets: caps big winners, PF drops
- DCP Filter: kills 74% of 5R+ trades
