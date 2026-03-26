# Parking Lot

## CLOSED — ML entry filter research (experiments 5-7)
- ~~DCP filter~~ — KILLS 68-79% of 5R+ trades. Low DCP = deep pullback = big runners. Do NOT filter.
- ~~ML win/loss prediction~~ — 70 features, 4 models, AUC 0.53 OOS. No signal.
- ~~Visual pattern features~~ — Chart shape (normalized close paths), AUC 0.54. No signal.
- ~~Regime features~~ — VIX proxy, day range, gap. AUC 0.52. No signal.
- MFE/Gate prediction (AUC 0.61-0.66) — partially predictable but redundant with 3-bar gate

## CLOSED — MNQ 3min validation (experiments 8-9)
- ~~RE-ENTRY after BE~~ — MNQ 3min: marginal +8.4% daily but +22% DD. PF flat. Not worth complexity for prop firm.
- ~~Volume Node S/R~~ — Target mode HARMFUL (caps 5R+, PF drops 2.22→1.97). Filter mode only improves PF by reducing trades.

## Untested on MNQ
- 3 MNQ with DD control (reduce to 1 when in DD)
- Entry cutoff 13:00 or 13:30 instead of 14:00
- Wider stop_buffer (0.5, 0.6 ATR)
- Different EMA periods (15/40, 10/30)
- Skip-after-win = 0 (don't skip)
- Different be_trigger/be_stop combos beyond current 0.3/0.15
- Stress test: extra slippage, wider spread
- Adaptive contract count based on recent ATR
- Morning-only (9:30-11:00) filter
- 5-min bars instead of 3-min for lower cost%
