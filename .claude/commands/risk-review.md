---
argument-hint: "策略文件路径"
---
# Risk Review (Red Team)

Run adversarial risk review on: $ARGUMENTS

Apply the 8-point Kill Test:
1. ✅ Code audit: look-ahead, same-bar, cost model correct?
2. ✅ Statistical: sample size, multiple comparison, Bonferroni?
3. ✅ Overfitting: param sensitivity ±20%, IS/OOS ratio?
4. ✅ Walk-forward: all periods profitable?
5. ✅ Cost stress: survives 2x costs?
6. ✅ Random baseline: beats >95% of 1000 random strategies?
7. ✅ Regime split: profitable in all 6-month sub-periods?
8. ✅ Execution delay: survives 1-bar delay?

Also run destruction tests:
- Worst 10 days, worst drawdown period
- Maximum consecutive losses
- Cost bomb threshold (at what cost multiplier does it die?)

Write report to `docs/RISK_REVIEW_[strategy_name].md`

Acceptance criteria (from docs/RISK_POLICY.md):
- PF > 1.3 OOS, walk-forward >80% profitable, survives 2x cost, param range < 0.5
