---
argument-hint: "策略名称和简述，如: momentum-breakout 突破动量策略"
---
# New Strategy Research Flow

Launch a 4-agent red/blue team to research a new strategy:

1. **blue-quant**: Design the strategy based on: $ARGUMENTS
   - Write implementation to `src/strategy/implementations/`
   - Use shared engine from `src/backtest/engine.py`
   - Must use per-contract cost model (see CLAUDE.md)

2. **blue-backtest**: Validate with walk-forward + stress test
   - Use `src/backtest/walk_forward.py` framework
   - Compare against baseline (EMA20 touch PF=1.47)

3. **red-critic**: Audit code + statistical validity
   - Apply 8-point Kill Test checklist
   - Check for look-ahead, overfitting, cost errors

4. **red-breaker**: Destruction test
   - Worst-case scenarios, regime splits, cost bombs
   - Random baseline comparison (1000 permutations)

Data: `data/barchart_nq/NQ_1min_continuous_RTH.csv`
IS=2022-2023, OOS=2024-2026. Target: PF>1.5, DD<15%.
