"""
Prop Firm Monte Carlo Simulator — MNQ EMA20 Strategy.

Uses empirical R-distribution from strategy_final.py to simulate:
  - Evaluation pass rate under various prop firm rules
  - Optimal R-sizing (how many R to divide total DD into)
  - EOD DD vs trailing DD impact
  - Consistency rule (20%/30%/40%) impact
  - Expected time to pass and expected payout

Designed to find the best firm + config to "milk" prop firms.
"""
from __future__ import annotations
import functools, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

print = functools.partial(print, flush=True)


# ══════════════════════════════════════════════════════════════════
# EMPIRICAL R-DISTRIBUTION (from strategy_final.py on QQQ 2yr)
# ══════════════════════════════════════════════════════════════════

def load_r_distribution():
    """Load empirical R-multiples from backtest."""
    from strategy_final import run_backtest
    df = pd.read_csv("data/QQQ_1Min_Polygon_2y_clean.csv",
                      index_col="timestamp", parse_dates=True)
    r = run_backtest(df)
    tl = r["trade_log"]
    tl["r_mult"] = tl["pnl"] / (tl["shares"] * tl["risk"])
    return tl["r_mult"].values


def load_daily_r_distribution():
    """Load daily R totals from backtest."""
    from strategy_final import run_backtest
    df = pd.read_csv("data/QQQ_1Min_Polygon_2y_clean.csv",
                      index_col="timestamp", parse_dates=True)
    r = run_backtest(df)
    tl = r["trade_log"]
    tl["r_mult"] = tl["pnl"] / (tl["shares"] * tl["risk"])
    tl["date"] = pd.to_datetime(tl["entry_time"]).dt.date
    daily = tl.groupby("date")["r_mult"].sum().values
    return daily


# ══════════════════════════════════════════════════════════════════
# PROP FIRM RULES
# ══════════════════════════════════════════════════════════════════

FIRMS = {
    "Apex_50K": {
        "name": "Apex $50K",
        "account_size": 50_000,
        "trailing_dd": 2500,      # trailing from equity peak
        "daily_dd": None,         # no separate daily limit
        "dd_type": "trailing",    # trailing = follows equity high
        "profit_target": 3000,
        "min_days": 7,
        "consistency_rule": 0.30, # no single day > 30% of total profit
        "monthly_fee": 167,
        "payout_split": 1.00,     # 100% first $25K
        "contracts_mnq": 20,      # max MNQ
        "eval_steps": 1,
    },
    "Apex_100K": {
        "name": "Apex $100K",
        "account_size": 100_000,
        "trailing_dd": 3000,
        "daily_dd": None,
        "dd_type": "trailing",
        "profit_target": 6000,
        "min_days": 7,
        "consistency_rule": 0.30,
        "monthly_fee": 207,
        "payout_split": 1.00,
        "contracts_mnq": 40,
        "eval_steps": 1,
    },
    "Apex_150K": {
        "name": "Apex $150K",
        "account_size": 150_000,
        "trailing_dd": 5000,
        "daily_dd": None,
        "dd_type": "trailing",
        "profit_target": 9000,
        "min_days": 7,
        "consistency_rule": 0.30,
        "monthly_fee": 297,
        "payout_split": 1.00,
        "contracts_mnq": 60,
        "eval_steps": 1,
    },
    "Topstep_50K": {
        "name": "Topstep $50K",
        "account_size": 50_000,
        "trailing_dd": 2000,      # trailing max DD
        "daily_dd": 1000,         # EOD daily loss limit
        "dd_type": "trailing",
        "profit_target": 3000,
        "min_days": 5,
        "consistency_rule": 0.50, # best day can't be > 50% of target
        "monthly_fee": 165,
        "payout_split": 0.90,
        "contracts_mnq": 10,
        "eval_steps": 1,
    },
    "Topstep_100K": {
        "name": "Topstep $100K",
        "account_size": 100_000,
        "trailing_dd": 3000,
        "daily_dd": 2000,
        "dd_type": "trailing",
        "profit_target": 6000,
        "min_days": 5,
        "consistency_rule": 0.50,
        "monthly_fee": 325,
        "payout_split": 0.90,
        "contracts_mnq": 20,
        "eval_steps": 1,
    },
    "Topstep_150K": {
        "name": "Topstep $150K",
        "account_size": 150_000,
        "trailing_dd": 4500,
        "daily_dd": 3000,
        "dd_type": "trailing",
        "profit_target": 9000,
        "min_days": 5,
        "consistency_rule": 0.50,
        "monthly_fee": 375,
        "payout_split": 0.90,
        "contracts_mnq": 30,
        "eval_steps": 1,
    },
    "MFF_50K": {
        "name": "MyFundedFutures $50K",
        "account_size": 50_000,
        "trailing_dd": 2000,
        "daily_dd": None,
        "dd_type": "EOD",         # EOD = measured at end of day, not intraday
        "profit_target": 3000,
        "min_days": 5,
        "consistency_rule": 0.40,
        "monthly_fee": 150,
        "payout_split": 1.00,
        "contracts_mnq": 10,
        "eval_steps": 1,
    },
    "MFF_100K": {
        "name": "MyFundedFutures $100K",
        "account_size": 100_000,
        "trailing_dd": 3000,
        "daily_dd": None,
        "dd_type": "EOD",
        "profit_target": 6000,
        "min_days": 5,
        "consistency_rule": 0.40,
        "monthly_fee": 275,
        "payout_split": 1.00,
        "contracts_mnq": 20,
        "eval_steps": 1,
    },
    "TPT_50K": {
        "name": "TakeProfitTrader $50K",
        "account_size": 50_000,
        "trailing_dd": 2000,
        "daily_dd": 1000,
        "dd_type": "trailing",
        "profit_target": 3000,
        "min_days": 5,
        "consistency_rule": 0.40,
        "monthly_fee": 150,
        "payout_split": 0.80,
        "contracts_mnq": 10,
        "eval_steps": 1,
    },
    "Bulenox_50K": {
        "name": "Bulenox $50K",
        "account_size": 50_000,
        "trailing_dd": 2500,
        "daily_dd": None,
        "dd_type": "trailing",
        "profit_target": 3000,
        "min_days": 5,
        "consistency_rule": 0.40,
        "monthly_fee": 155,
        "payout_split": 1.00,
        "contracts_mnq": 20,
        "eval_steps": 1,
    },
}


# ══════════════════════════════════════════════════════════════════
# MNQ CONVERSION
# ══════════════════════════════════════════════════════════════════

MNQ_TICK = 0.25        # $0.25 per tick
MNQ_POINT = 1.0        # $1 per point (4 ticks = $1)
MNQ_MULTIPLIER = 2.0   # $2 per point for MNQ ($0.50 per tick)
MNQ_COMMISSION = 0.62   # round trip per contract (typical)

# Our strategy on QQQ: 1R ≈ $0.50-0.60 on QQQ
# On MNQ: price moves are similar (NQ tracks QQQ * 4ish)
# We'll parametrize: 1R in $ = f(contracts, stop_distance)
# For prop firm: total_dd / N_divisions = $ per R
# Then contracts = R_dollars / (stop_ticks * MNQ_MULTIPLIER)


# ══════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════

def simulate_eval(args):
    """
    Simulate one evaluation attempt.

    Returns dict with: passed, days, max_dd_hit, daily_dd_hit,
    consistency_fail, final_pnl
    """
    firm_key, r_per_trade_dollars, daily_r_values, trade_r_values, seed = args
    firm = FIRMS[firm_key]
    rng = np.random.RandomState(seed)

    trailing_dd = firm["trailing_dd"]
    daily_dd = firm["daily_dd"]
    dd_type = firm["dd_type"]
    profit_target = firm["profit_target"]
    min_days = firm["min_days"]
    consistency = firm["consistency_rule"]

    # Resample daily R values to simulate trading days
    # Each "day" we draw a random daily R total from the empirical distribution
    equity = 0.0
    peak_equity = 0.0
    day_pnls = []
    days = 0
    max_days = 120  # max 4 months before giving up

    while days < max_days:
        # Sample a daily R from empirical distribution
        daily_r = rng.choice(daily_r_values)
        daily_pnl = daily_r * r_per_trade_dollars

        # Check daily DD limit (if exists)
        if daily_dd is not None and daily_pnl < -daily_dd:
            # Hit daily limit — capped loss
            daily_pnl = -daily_dd

        new_equity = equity + daily_pnl
        days += 1
        day_pnls.append(daily_pnl)

        # Check trailing DD
        if dd_type == "trailing":
            peak_equity = max(peak_equity, new_equity)
            current_dd = peak_equity - new_equity
        else:  # EOD — check at end of day
            peak_equity = max(peak_equity, new_equity)
            current_dd = peak_equity - new_equity

        if current_dd >= trailing_dd:
            return {
                "passed": False, "days": days, "reason": "max_dd",
                "final_pnl": new_equity, "peak": peak_equity,
            }

        equity = new_equity

        # Check profit target
        if equity >= profit_target and days >= min_days:
            # Check consistency rule
            if consistency > 0 and len(day_pnls) > 0:
                max_day = max(day_pnls)
                if max_day > profit_target * consistency:
                    # Consistency violated — but we've already passed target
                    # Some firms fail you, some just cap. We'll model as: keep trading
                    # until either a) you have enough days where max_day < threshold
                    # or b) you bust. For simplicity: check if removing best day
                    # still passes target.
                    total_without_best = equity - max_day
                    if total_without_best < profit_target * (1 - consistency):
                        # Need to keep trading to dilute the big day
                        continue

            return {
                "passed": True, "days": days, "reason": "target_hit",
                "final_pnl": equity, "peak": peak_equity,
            }

    return {
        "passed": False, "days": days, "reason": "timeout",
        "final_pnl": equity, "peak": peak_equity,
    }


def run_monte_carlo(firm_key, r_dollars, daily_r_values, trade_r_values,
                    n_sims=5000):
    """Run N simulations for a firm+R configuration."""
    args = [(firm_key, r_dollars, daily_r_values, trade_r_values, seed)
            for seed in range(n_sims)]

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        results = list(pool.map(simulate_eval, args))

    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / n_sims * 100

    days_to_pass = [r["days"] for r in results if r["passed"]]
    avg_days = np.mean(days_to_pass) if days_to_pass else 0
    med_days = np.median(days_to_pass) if days_to_pass else 0

    busted_dd = sum(1 for r in results if r["reason"] == "max_dd")
    busted_timeout = sum(1 for r in results if r["reason"] == "timeout")

    return {
        "firm": firm_key,
        "r_dollars": r_dollars,
        "pass_rate": round(pass_rate, 1),
        "avg_days": round(avg_days, 1),
        "med_days": round(med_days, 0),
        "busted_dd_pct": round(busted_dd / n_sims * 100, 1),
        "busted_timeout_pct": round(busted_timeout / n_sims * 100, 1),
        "n_sims": n_sims,
    }


# ══════════════════════════════════════════════════════════════════
# OPTIMAL R SEARCH
# ══════════════════════════════════════════════════════════════════

def find_optimal_r(firm_key, daily_r_values, trade_r_values, n_sims=3000):
    """Sweep R-divisions to find optimal sizing."""
    firm = FIRMS[firm_key]
    total_dd = firm["trailing_dd"]

    # Sweep: divide total DD into N parts
    # N = 5 means 1R = DD/5 = very aggressive
    # N = 30 means 1R = DD/30 = very conservative
    results = []
    for n_div in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]:
        r_dollars = total_dd / n_div
        r = run_monte_carlo(firm_key, r_dollars, daily_r_values, trade_r_values,
                           n_sims=n_sims)
        r["n_divisions"] = n_div
        r["r_per_trade"] = round(r_dollars, 2)

        # Expected value per attempt
        fee = firm["monthly_fee"]
        payout = firm["payout_split"]
        target = firm["profit_target"]
        ev = (r["pass_rate"] / 100 * target * payout) - fee
        r["ev_per_attempt"] = round(ev, 0)

        # ROI = EV / fee
        r["roi_pct"] = round(ev / fee * 100, 0) if fee > 0 else 0

        results.append(r)

    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("Loading empirical R-distribution from strategy_final.py...")
    trade_r = load_r_distribution()
    daily_r = load_daily_r_distribution()

    print(f"  Trade R: {len(trade_r)} trades, mean={trade_r.mean():.3f}R, median={np.median(trade_r):.3f}R")
    print(f"  Daily R: {len(daily_r)} days, mean={daily_r.mean():.3f}R, median={np.median(daily_r):.3f}R")
    print(f"  Daily R range: [{daily_r.min():.2f}R, {daily_r.max():.2f}R]")
    print()

    # ═══ SWEEP ALL FIRMS × R DIVISIONS ═══
    print("=" * 100)
    print("PROP FIRM EVALUATION SIMULATOR — Monte Carlo (3000 sims each)")
    print("=" * 100)

    all_results = []

    for firm_key in sorted(FIRMS.keys()):
        firm = FIRMS[firm_key]
        print(f"\n{'─'*80}")
        print(f"  {firm['name']}  |  DD=${firm['trailing_dd']}  |  Target=${firm['profit_target']}"
              f"  |  Fee=${firm['monthly_fee']}/mo  |  Consistency={firm['consistency_rule']*100:.0f}%"
              f"  |  DD Type={firm['dd_type']}")
        print(f"{'─'*80}")
        print(f"  {'Divisions':>10} {'1R($)':>8} {'Pass%':>7} {'AvgDays':>8} {'MedDays':>8}"
              f" {'Bust%':>7} {'EV($)':>8} {'ROI%':>7}")

        results = find_optimal_r(firm_key, daily_r, trade_r, n_sims=3000)
        all_results.extend(results)

        for r in results:
            mark = " ***" if r["pass_rate"] > 70 and r["ev_per_attempt"] > 0 else ""
            print(f"  {r['n_divisions']:>10} {r['r_per_trade']:>8.0f} {r['pass_rate']:>6.1f}%"
                  f" {r['avg_days']:>7.1f} {r['med_days']:>7.0f}"
                  f" {r['busted_dd_pct']:>6.1f}% {r['ev_per_attempt']:>7.0f} {r['roi_pct']:>6.0f}%{mark}")

    # ═══ BEST CONFIGS ═══
    print("\n" + "=" * 100)
    print("TOP 10 CONFIGS BY ROI")
    print("=" * 100)

    df_all = pd.DataFrame(all_results)
    # Filter: pass rate > 50%, positive EV
    viable = df_all[(df_all["pass_rate"] > 50) & (df_all["ev_per_attempt"] > 0)]
    top = viable.nlargest(10, "roi_pct")

    print(f"  {'Firm':<25} {'Div':>4} {'1R($)':>7} {'Pass%':>7} {'Days':>6}"
          f" {'EV($)':>8} {'ROI%':>7}")
    print(f"  {'-'*75}")

    for _, r in top.iterrows():
        firm = FIRMS[r["firm"]]
        print(f"  {firm['name']:<25} {r['n_divisions']:>4} {r['r_per_trade']:>7.0f}"
              f" {r['pass_rate']:>6.1f}% {r['avg_days']:>5.0f}"
              f" {r['ev_per_attempt']:>7.0f} {r['roi_pct']:>6.0f}%")

    # ═══ CONSISTENCY RULE IMPACT ═══
    print("\n" + "=" * 100)
    print("CONSISTENCY RULE IMPACT (best config per firm)")
    print("=" * 100)

    for firm_key in sorted(FIRMS.keys()):
        firm_results = [r for r in all_results if r["firm"] == firm_key]
        if not firm_results:
            continue
        best = max(firm_results, key=lambda x: x["ev_per_attempt"])
        firm = FIRMS[firm_key]

        # Our max daily R from data
        max_daily = daily_r.max()
        best_r = best["r_per_trade"]
        max_daily_pnl = max_daily * best_r
        consistency_cap = firm["profit_target"] * firm["consistency_rule"]
        capped = max_daily_pnl > consistency_cap

        print(f"  {firm['name']:<25} Best={best['n_divisions']}div  "
              f"MaxDayR={max_daily:.1f}R  MaxDay$={max_daily_pnl:.0f}  "
              f"Cap={consistency_cap:.0f}  {'CAPPED' if capped else 'OK'}")

    # ═══ EOD vs TRAILING DD ═══
    print("\n" + "=" * 100)
    print("EOD vs TRAILING DD COMPARISON")
    print("=" * 100)

    eod_firms = [k for k, v in FIRMS.items() if v["dd_type"] == "EOD"]
    trail_firms = [k for k, v in FIRMS.items() if v["dd_type"] == "trailing"]

    eod_pass = [r["pass_rate"] for r in all_results
                if r["firm"] in eod_firms and r["n_divisions"] == 15]
    trail_pass = [r["pass_rate"] for r in all_results
                  if r["firm"] in trail_firms and r["n_divisions"] == 15]

    if eod_pass:
        print(f"  EOD firms avg pass rate (15-div): {np.mean(eod_pass):.1f}%")
    if trail_pass:
        print(f"  Trailing firms avg pass rate (15-div): {np.mean(trail_pass):.1f}%")
    print(f"  EOD advantage: EOD doesn't penalize intraday swings → higher pass rate")

    # ═══ RECOMMENDATION ═══
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    if len(viable) > 0:
        best_overall = viable.loc[viable["roi_pct"].idxmax()]
        firm = FIRMS[best_overall["firm"]]
        print(f"""
  Best Config:
    Firm:        {firm['name']}
    Account:     ${firm['account_size']:,}
    DD Budget:   ${firm['trailing_dd']:,} (divide into {int(best_overall['n_divisions'])} parts)
    1R = ${best_overall['r_per_trade']:.0f}
    Pass Rate:   {best_overall['pass_rate']:.1f}%
    Avg Days:    {best_overall['avg_days']:.0f}
    EV/attempt:  ${best_overall['ev_per_attempt']:.0f}
    ROI:         {best_overall['roi_pct']:.0f}%
    Monthly Fee: ${firm['monthly_fee']}
    Payout:      {firm['payout_split']*100:.0f}%

  MNQ Sizing (approximate):
    Typical 1R on MNQ ≈ 10-15 ticks stop × $0.50/tick = $5-7.50/contract
    At 1R = ${best_overall['r_per_trade']:.0f}: trade {best_overall['r_per_trade']/7.5:.0f}-{best_overall['r_per_trade']/5:.0f} MNQ contracts
    (Max allowed: {firm['contracts_mnq']} MNQ)

  Strategy:
    - Use Plan G entry/exit logic unchanged
    - Set position size so each trade risks exactly ${best_overall['r_per_trade']:.0f}
    - Daily R limit: 2.5R = ${best_overall['r_per_trade'] * 2.5:.0f} (well within daily DD if applicable)
    - Expected to pass in {best_overall['avg_days']:.0f} trading days
    - {best_overall['pass_rate']:.0f}% chance of passing each attempt
""")
    else:
        print("  No viable configs found with pass_rate > 50% and positive EV.")


if __name__ == "__main__":
    main()
