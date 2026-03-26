# ML Entry Filter — Experiment 5 Results

## Setup
- 37 features × 2827 IS trades × 2833 OOS trades
- Features: price/EMA, momentum, volatility, volume, bar patterns, time, channel, multi-TF

## Results

| Model | AUC_IS | AUC_OOS | Gap | Verdict |
|-------|--------|---------|-----|---------|
| LightGBM | 0.711 | 0.528 | +0.183 | NO signal |
| RandomForest | 0.773 | 0.535 | +0.239 | NO signal |
| R-regression | corr=0.325 | corr=0.039 | - | NO signal |

## Key Observations
1. LightGBM early-stopped at 19 rounds — couldn't improve OOS beyond 0.528
2. All feature correlations with R < |0.07|
3. Walk-forward by quarter: filter doesn't help in any quarter
4. At aggressive threshold (0.65), PF improves BUT kills 50% of 5R+ trades
5. IS models fit well (AUC 0.71-0.77) = pure noise fitting

## Verdict
Pre-entry features from 37 indicators CANNOT predict trade outcome OOS.
The edge is structural, not predictive.
