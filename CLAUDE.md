# CLAUDE.md — NQ Futures Quantitative Trading System

## Project Overview
NQ/MNQ futures intraday trading system for Topstep 50K prop firm evaluation.
Core instruments: NQ ($20/pt) and MNQ ($2/pt) on CME Globex.
Validated on 4.2Y real NQ continuous contract data (2022-2026).

## Current Status (2026-03-27)
**Edge discovery phase.** 15 experiments + 7 agent runs + red/blue team + tournament completed.
Strategy has real edge (raw PF=2.02) but MNQ cost structure is the #1 bottleneck.
Active team `propfirm-edge-discovery` running 3 scouts.

## Verified Baseline Numbers (audit-corrected costs)

| Config | PF | DD | $/day | Sharpe | Cost/R | Topstep |
|--------|-----|-----|-------|--------|--------|---------|
| EMA20 10min MNQ×2 | 1.465 | $1,753 | $26 | 2.20 | 4.4% | ✅ DD fits, ❌ too slow |
| EMA20 10min NQ×1 | 1.836 | $5,952 | $202 | 3.42 | 1.0% | ❌ DD too large |
| Vol Squeeze 30min NQ×1 | 1.533 | $9,725 | $66 | 2.33 | 0.6% | ❌ DD too large |

**None currently pass Topstep 50K fully (PF>1.5 + DD<$2K + $/day>$50).**

## Three Critical Findings

### 1. Entry Signal = Zero Alpha (Red Team Confirmed)
Random entry + current exit system = PF 1.484.
EMA20 touch entry + same exit = PF 1.465.
54.5% of random strategies beat EMA20 touch. p=0.50.
**All edge is in the EXIT SYSTEM: gate + BE + chandelier.**

### 2. Cost Structure is the #1 Bottleneck
| Timeframe | Cost/Risk | Net PF (MNQ×2) | Edge retained |
|-----------|-----------|-----------------|---------------|
| 3min | 8.4% | 1.193 | 27% of raw |
| 10min | 4.4% | 1.465 | 56% of raw |
| 30min NQ | 0.6% | 1.533 | 73% of raw |

Raw PF = 2.02. Costs eat 27-73% depending on timeframe.

### 3. Topstep 50K Math Problem
- MNQ×2 at 10min: DD=$1,753 fits, but $26/day → 150 days to $3K target
- Adding contracts: DD scales linearly, busts $2K at 3+ MNQ
- NQ×1: $202/day but DD=$5,952 → busts any Topstep tier
- **This is a mathematical constraint, not a strategy problem.**

## Technology Stack
- Python 3.13+, pandas, numpy, scikit-learn, lightgbm
- Data: NQ 1-min continuous RTH (Barchart, Panama Canal adjusted)
- Backtest: custom vectorized engine (`src/backtest/engine.py`)
- Costs: per-contract model with gap-through stop fill
- Agent Teams: enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`)

## Cost Model (MANDATORY — all agents must use)
All costs PER CONTRACT. Never flat per trade. This was a P0 bug that inflated all early results.
```python
MNQ: comm=$2.46 RT, spread=$0.50×nc, stop_slip=$1.00×nc, be_slip=$1.00×nc
NQ:  comm=$2.46 RT, spread=$5.00×nc, stop_slip=$1.25×nc, be_slip=$1.25×nc
```
Stop fill: gap-through — `fill_price = min(stop, bar_open)` for longs.
Shared engine: `src/backtest/engine.py`

## Data
- `data/barchart_nq/NQ_1min_continuous_RTH.csv` — 417K bars, 2021-12 to 2026-03
- Timezone: file is CT, add 1 hour for ET
- RTH: 09:30-16:00 ET
- IS = 2022-2023, OOS = 2024-2026 (forward walk, never reversed)

## Risk Hard Rules (ALL agents must obey)
- Topstep 50K: MaxDD < $2,000, profit target $3,000
- Personal $50K: MaxDD < 15% of account
- Daily loss limit: 2R
- No entry after 14:00 ET, force close at 15:58 ET
- No look-ahead bias — signals use only closed bar data
- New strategies: 4Y IS+OOS + walk-forward + red team Kill Test

## Red Team Kill Test (8 checks, must pass ALL)
1. Code audit: no look-ahead, no same-bar entry+exit, cost model correct
2. Statistical: sample size >200, Bonferroni correction applied
3. Overfitting: param sensitivity ±20% range < 0.5 PF
4. Walk-forward: >80% of 6-month periods profitable
5. Cost stress: survives 2x costs (PF still > 1.0)
6. Random baseline: beats >95% of 1000 random strategies
7. Regime split: profitable in all isolated 6-month sub-periods
8. Execution delay: survives 1-bar delay (PF decay < 20%)

## Agent Teams Setup
- Env: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` (global settings.json)
- Agents: `.claude/agents/` (blue-quant, blue-backtest, red-critic, red-breaker)
- Commands: `.claude/commands/` (/new-strategy, /full-backtest, /risk-review)
- Teams use `TeamCreate` → `Agent` spawn with `team_name` → `SendMessage` coordination
- Discoveries log to `docs/edge-discovery/DISCOVERY_LOG.md`

## Directory Structure
```
src/
├── backtest/engine.py          # shared backtest (correct costs)
├── backtest/walk_forward.py    # walk-forward validator
├── strategy/implementations/   # 6 strategies
├── edge-discovery/             # active scout outputs
├── risk/                       # risk modules
├── execution/                  # execution layer (TBD)
└── data/                       # data pipeline
docs/
├── STATUS.md                   # team status
├── STRATEGY_SPEC.md            # strategy specifications
├── RISK_POLICY.md              # risk limits
├── edge-discovery/DISCOVERY_LOG.md  # edge findings log
└── tournament/                 # tournament results
configs/
├── strategy_params.yaml
└── risk_limits.yaml
.lab/log.md                     # 15 experiment log
```

## Experiment History (15 experiments + 7 agents, see .lab/log.md)
### Phase 1: Parameter optimization (Exp 0-4)
1min→3min→10min, found gate/BE/chandelier optimal params

### Phase 2: Entry filter research (Exp 5-9)
ML 70 features (AUC=0.53), DCP, Volume S/R, Re-entry — all failed or marginal

### Phase 3: NQ real data + audit (Exp 10-15)
Real NQ data validation, cost bug discovered and fixed, 10min breakthrough

### Phase 4: Red/Blue team + Tournament (7 agents)
- Blue team: Squeeze wins (PF=1.53), ORB fails, MTF short interesting
- Red team: entry=zero alpha, strategy survives 2x costs
- Tournament: Vol Squeeze leads, Mean Rev eliminated, Momentum API error

### Phase 5: Edge Discovery (current)
Team propfirm-edge-discovery: exit-optimizer, tf-scout, entry-scout

## Known Dead Ends (DO NOT re-test)
- ML entry filters (AUC=0.53, 70 features, 4 models)
- Volume S/R targets (caps big winners, PF drops)
- DCP filter (kills 74% of 5R+ trades)
- Mean reversion on NQ (PF=1.089, NQ is trending)
- ORB 8 variants (OOS decays in 2024-2025)
- Dynamic position sizing (negative selection: more contracts in low-ATR = amplifies losses)
- 3min MNQ (cost/risk 8.4%, eats 73% of edge)
- skip_after_win, gate_mfe, be_trigger combos from Exp 11 (invalidated by cost bug)
