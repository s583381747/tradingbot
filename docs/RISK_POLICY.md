# Risk Policy — NQ Futures Trading System

## Position Limits
- Max contracts per trade: 10 MNQ or 1 NQ
- Max concurrent positions: 1 (single strategy, no stacking)
- Max daily trades: 10 (circuit breaker)

## Loss Limits
- Per-trade max loss: 1R (defined by stop distance × contracts × point value)
- Daily loss limit: 2R cumulative → stop trading for the day
- Weekly loss limit: 5R → reduce to minimum size (1 MNQ) for remainder of week
- Monthly loss limit: 10R → pause and review strategy

## Drawdown Limits
- Topstep 50K: trailing DD < $2,000 → trade 1-2 MNQ max
- Personal $50K: trailing DD < $7,500 (15%) → trade 1 NQ max
- If DD exceeds 10% of account: reduce to half position size

## Execution Rules
- No entry after 14:00 ET
- Force close all positions by 15:58 ET
- No overnight holds (RTH only)
- Paper trade minimum 2 weeks before live

## Cost Assumptions (conservative)
- All costs per contract, never flat per trade
- Stop slippage: $1.00/contract MNQ, $1.25/contract NQ
- Spread: $0.50/contract MNQ, $5.00/contract NQ
- Commission: $2.46/contract round-trip
- Stop fill: worst of (stop price, bar open) for gap-through

## Strategy Acceptance Criteria
1. PF > 1.3 after costs on 4Y OOS data
2. Walk-forward: >80% of periods profitable
3. Survives 2x cost stress test (PF still > 1.0)
4. Survives 1-bar execution delay (PF decay < 20%)
5. Parameter sensitivity: PF range < 0.5 across ±20% perturbation
6. No look-ahead bias (code audit required)
7. 200+ trades minimum for statistical significance
