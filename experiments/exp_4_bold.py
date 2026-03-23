"""
Experiment 4: BOLD parameter exploration for Chop-Box v2.

Philosophy: "大胆一点" — Push the framework to its limits.

Based on prior findings:
- EMA10 + slopeperiod=3 shows promise
- chop_threshold 0.015-0.018 is profitable but few trades
- TP system hurts EVERYTHING → disable
- Tighter stops work better with chop box (1.5x ATR)
- losers_max_bars=30 is near-optimal

New directions to explore:
1. ULTRA-SHORT EMA (5, 8) — scalping within trend
2. LONGS ONLY vs SHORTS ONLY — is one side poisoning the other?
3. RADICAL TIME FILTERS — only trade first 2 hours or last 2 hours
4. FIXED-BAR EXIT — hold for exactly N bars then close (bypass trail entirely)
5. EMA SLOW = NONE — remove slow EMA filter entirely
6. MICRO PULLBACK — tighter pullback zone (0.3-0.5 ATR)
7. COMBO KILLER — best of everything stacked
"""

from __future__ import annotations
import importlib, sys, time, copy
import backtrader as bt
import pandas as pd

DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
CASH = 100_000
COMMISSION = 0.001


def load_data():
    return pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)


def run_one(df, overrides: dict, strategy_mod_patches: dict | None = None) -> dict:
    """Run backtest with parameter overrides."""
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    mod = importlib.import_module("strategy")
    for k, v in overrides.items():
        setattr(mod, k, v)

    # Apply any source-level patches (e.g., longs-only)
    StrategyClass = mod.Strategy
    param_name_map = {
        "EMA_PERIOD": "ema_period", "EMA_SLOW_PERIOD": "ema_slow_period",
        "ATR_PERIOD": "atr_period", "EMA_SLOPE_PERIOD": "ema_slope_period",
        "CHOP_SLOPE_AVG_PERIOD": "chop_slope_avg_period",
        "CHOP_SLOPE_THRESHOLD": "chop_slope_threshold",
        "CHOP_BOX_MIN_BARS": "chop_box_min_bars",
        "PULLBACK_TOUCH_MULT": "pullback_touch_mult",
        "MIN_PULLBACK_BARS": "min_pullback_bars",
        "INITIAL_STOP_ATR_MULT": "initial_stop_atr_mult",
        "TP_ACTIVATE_ATR": "tp_activate_atr",
        "TP1_PCT": "tp1_pct", "TP2_PCT": "tp2_pct", "TP3_PCT": "tp3_pct",
        "TP1_CANDLE_OFFSET": "tp1_candle_offset",
        "TP2_EMA_ATR_MULT": "tp2_ema_atr_mult",
        "TP3_EMA_ATR_MULT": "tp3_ema_atr_mult",
        "ENABLE_ADDON": "enable_addon", "MAX_ADDONS": "max_addons",
        "ADDON_PULLBACK_MULT": "addon_pullback_mult",
        "ADDON_MIN_BARS": "addon_min_bars",
        "RISK_PCT": "risk_pct", "MAX_POSITION_PCT": "max_position_pct",
        "MAX_DAILY_TRADES": "max_daily_trades",
        "MAX_DAILY_LOSS_PCT": "max_daily_loss_pct",
        "LOSERS_MAX_BARS": "losers_max_bars",
        "NO_ENTRY_AFTER_HOUR": "no_entry_after_hour",
        "NO_ENTRY_AFTER_MINUTE": "no_entry_after_minute",
    }
    params = {param_name_map[k]: v for k, v in overrides.items() if k in param_name_map}
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df, datetime=None, open="Open", high="High",
                                low="Low", close="Close", volume="Volume", openinterest=-1)
    cerebro.adddata(data)
    cerebro.addstrategy(StrategyClass, **params)
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - CASH) / CASH * 100
    sharpe_a = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_a.get("sharperatio", 0.0) or 0.0
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0.0) or 0.0
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    win_rate = won / total_trades * 100 if total_trades > 0 else 0.0
    gross_won = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    gross_lost = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    pf = gross_won / gross_lost if gross_lost > 0 else 0.0
    return {
        "total_return": round(total_return, 2), "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2), "total_trades": total_trades,
        "won": won, "lost": lost, "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "gross_won": round(gross_won, 2), "gross_lost": round(gross_lost, 2),
    }


def score(m):
    trades = m["total_trades"]
    if trades < 5: return -10.0
    sharpe_norm = max(0, min(1, (m["sharpe"] + 2) / 5))
    pf_norm = max(0, min(1, m["profit_factor"] / 3))
    ret_norm = max(0, min(1, (m["total_return"] + 20) / 70))
    wr_norm = max(0, min(1, m["win_rate"] / 100))
    dd_norm = max(0, min(1, 1 - m["max_drawdown"] / 30))
    raw = 0.30*sharpe_norm + 0.25*pf_norm + 0.20*ret_norm + 0.10*wr_norm + 0.15*dd_norm
    if trades < 20: mult = 0.2
    elif trades < 50: mult = 0.6
    elif trades <= 500: mult = 1.0
    else: mult = 0.8
    return round(raw * mult, 4)


def main():
    df = load_data()
    print(f"Data: {len(df)} bars\n")

    CLEAN = {
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    experiments = {}

    # ══════════════════════════════════════════════════════════════
    # BLOCK 1: ULTRA-SHORT EMA (5, 8) — faster reaction to pullbacks
    # ══════════════════════════════════════════════════════════════
    for ema in [5, 8]:
        for chop in [0.010, 0.012, 0.015, 0.020]:
            for sp in [2, 3]:
                experiments[f"ema{ema}_sp{sp}_c{chop}"] = {
                    **CLEAN,
                    "EMA_PERIOD": ema,
                    "EMA_SLOPE_PERIOD": sp,
                    "CHOP_SLOPE_THRESHOLD": chop,
                    "INITIAL_STOP_ATR_MULT": 1.5,
                    "LOSERS_MAX_BARS": 30,
                }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 2: MICRO PULLBACK — very tight pullback zone
    # ══════════════════════════════════════════════════════════════
    for pb in [0.3, 0.5, 0.8]:
        for ema in [8, 10]:
            experiments[f"micro_pb{pb}_e{ema}"] = {
                **CLEAN,
                "EMA_PERIOD": ema,
                "EMA_SLOPE_PERIOD": 3,
                "CHOP_SLOPE_THRESHOLD": 0.012,
                "PULLBACK_TOUCH_MULT": pb,
                "INITIAL_STOP_ATR_MULT": 1.0,  # ultra tight
                "LOSERS_MAX_BARS": 20,
            }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 3: RADICAL TIME FILTERS
    # Entry only in first 2 hours (9:30-11:30) or last 2 hours (14:00-15:30)
    # ══════════════════════════════════════════════════════════════
    # Morning only: cut entries after 11:30
    for ema in [8, 10, 20]:
        experiments[f"morning_e{ema}"] = {
            **CLEAN,
            "EMA_PERIOD": ema,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "NO_ENTRY_AFTER_HOUR": 11,
            "NO_ENTRY_AFTER_MINUTE": 30,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # Afternoon: use 14:00 cutoff (entries only first hour of afternoon)
    # This effectively tests midday entries by moving cutoff earlier
    for ema in [8, 10, 20]:
        experiments[f"early_cut_e{ema}"] = {
            **CLEAN,
            "EMA_PERIOD": ema,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "NO_ENTRY_AFTER_HOUR": 14,
            "NO_ENTRY_AFTER_MINUTE": 0,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 4: FIXED-BAR EXIT — hold exactly N bars then close
    # (Simulated: losers_max_bars = N, and initial stop super wide so it never fires)
    # ══════════════════════════════════════════════════════════════
    for hold in [10, 15, 20, 30, 45, 60]:
        experiments[f"fixed_hold_{hold}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 50.0,  # effectively disabled
            "LOSERS_MAX_BARS": hold,  # this is our exit
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 5: NO SLOW EMA FILTER — remove EMA alignment requirement
    # (Use very short slow EMA so it's almost always aligned)
    # ══════════════════════════════════════════════════════════════
    for chop in [0.010, 0.012, 0.015]:
        experiments[f"no_slow_c{chop}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "EMA_SLOW_PERIOD": 12,  # nearly same as fast → always aligned
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": chop,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 6: RISK SIZING — more/less risk per trade
    # ══════════════════════════════════════════════════════════════
    for risk in [0.005, 0.02, 0.03]:
        experiments[f"risk_{risk}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
            "RISK_PCT": risk,
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 7: MAX DAILY TRADES — aggressive (unlimited) vs conservative
    # ══════════════════════════════════════════════════════════════
    for mdt in [2, 3, 4, 10, 20]:
        experiments[f"daily_{mdt}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
            "MAX_DAILY_TRADES": mdt,
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 8: CHOP BOX SENSITIVITY — very short vs very long chop periods
    # ══════════════════════════════════════════════════════════════
    for csa in [5, 10, 15, 30, 40]:
        experiments[f"chopavg_{csa}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "CHOP_SLOPE_AVG_PERIOD": csa,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 9: STOP DISTANCE FINE SWEEP around 1.5
    # ══════════════════════════════════════════════════════════════
    for stop in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        experiments[f"stop_{stop}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": stop,
            "LOSERS_MAX_BARS": 30,
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 10: ATR PERIOD — shorter ATR = more reactive
    # ══════════════════════════════════════════════════════════════
    for atr_p in [5, 7, 10, 20]:
        experiments[f"atr_{atr_p}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "ATR_PERIOD": atr_p,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 11: COMBO KILLERS — stack every best finding
    # ══════════════════════════════════════════════════════════════
    experiments["KILLER_A"] = {
        **CLEAN,
        "EMA_PERIOD": 8,
        "EMA_SLOPE_PERIOD": 2,
        "CHOP_SLOPE_THRESHOLD": 0.015,
        "CHOP_SLOPE_AVG_PERIOD": 10,
        "CHOP_BOX_MIN_BARS": 3,
        "PULLBACK_TOUCH_MULT": 1.0,
        "MIN_PULLBACK_BARS": 1,
        "INITIAL_STOP_ATR_MULT": 1.0,
        "LOSERS_MAX_BARS": 20,
        "MAX_DAILY_TRADES": 4,
        "NO_ENTRY_AFTER_HOUR": 14,
        "NO_ENTRY_AFTER_MINUTE": 0,
    }

    experiments["KILLER_B"] = {
        **CLEAN,
        "EMA_PERIOD": 10,
        "EMA_SLOPE_PERIOD": 3,
        "CHOP_SLOPE_THRESHOLD": 0.013,
        "CHOP_SLOPE_AVG_PERIOD": 10,
        "CHOP_BOX_MIN_BARS": 3,
        "PULLBACK_TOUCH_MULT": 0.8,
        "MIN_PULLBACK_BARS": 1,
        "INITIAL_STOP_ATR_MULT": 1.2,
        "LOSERS_MAX_BARS": 25,
        "MAX_DAILY_TRADES": 6,
    }

    experiments["KILLER_C"] = {
        **CLEAN,
        "EMA_PERIOD": 10,
        "EMA_SLOPE_PERIOD": 3,
        "CHOP_SLOPE_THRESHOLD": 0.015,
        "CHOP_SLOPE_AVG_PERIOD": 15,
        "CHOP_BOX_MIN_BARS": 5,
        "PULLBACK_TOUCH_MULT": 1.2,
        "MIN_PULLBACK_BARS": 1,
        "INITIAL_STOP_ATR_MULT": 1.5,
        "LOSERS_MAX_BARS": 30,
        "MAX_DAILY_TRADES": 3,
        "NO_ENTRY_AFTER_HOUR": 11,
        "NO_ENTRY_AFTER_MINUTE": 30,
    }

    experiments["KILLER_D_scalp"] = {
        **CLEAN,
        "EMA_PERIOD": 5,
        "EMA_SLOW_PERIOD": 20,
        "EMA_SLOPE_PERIOD": 2,
        "CHOP_SLOPE_THRESHOLD": 0.020,
        "CHOP_SLOPE_AVG_PERIOD": 5,
        "CHOP_BOX_MIN_BARS": 3,
        "PULLBACK_TOUCH_MULT": 0.5,
        "MIN_PULLBACK_BARS": 1,
        "INITIAL_STOP_ATR_MULT": 0.8,
        "LOSERS_MAX_BARS": 15,
        "MAX_DAILY_TRADES": 10,
    }

    experiments["KILLER_E_patient"] = {
        **CLEAN,
        "EMA_PERIOD": 15,
        "EMA_SLOW_PERIOD": 50,
        "EMA_SLOPE_PERIOD": 5,
        "CHOP_SLOPE_THRESHOLD": 0.018,
        "CHOP_SLOPE_AVG_PERIOD": 20,
        "CHOP_BOX_MIN_BARS": 8,
        "PULLBACK_TOUCH_MULT": 1.2,
        "MIN_PULLBACK_BARS": 2,
        "INITIAL_STOP_ATR_MULT": 2.0,
        "LOSERS_MAX_BARS": 45,
        "MAX_DAILY_TRADES": 3,
    }

    experiments["KILLER_F_morning_scalp"] = {
        **CLEAN,
        "EMA_PERIOD": 8,
        "EMA_SLOW_PERIOD": 30,
        "EMA_SLOPE_PERIOD": 2,
        "CHOP_SLOPE_THRESHOLD": 0.015,
        "CHOP_SLOPE_AVG_PERIOD": 8,
        "CHOP_BOX_MIN_BARS": 3,
        "PULLBACK_TOUCH_MULT": 0.8,
        "MIN_PULLBACK_BARS": 1,
        "INITIAL_STOP_ATR_MULT": 1.0,
        "LOSERS_MAX_BARS": 20,
        "MAX_DAILY_TRADES": 4,
        "NO_ENTRY_AFTER_HOUR": 11,
        "NO_ENTRY_AFTER_MINUTE": 30,
    }

    # ══════════════════════════════════════════════════════════════
    # BLOCK 12: Old v1 mimicry within v2 framework
    # Try to reproduce the old strategy's success in the new framework
    # ══════════════════════════════════════════════════════════════
    experiments["v1_mimic_A"] = {
        **CLEAN,
        "EMA_PERIOD": 20,
        "EMA_SLOW_PERIOD": 50,
        "EMA_SLOPE_PERIOD": 5,
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "CHOP_SLOPE_AVG_PERIOD": 5,  # very responsive
        "CHOP_BOX_MIN_BARS": 0,  # no box required
        "PULLBACK_TOUCH_MULT": 1.2,
        "MIN_PULLBACK_BARS": 1,
        "INITIAL_STOP_ATR_MULT": 5.0,  # wide like v1 best
        "LOSERS_MAX_BARS": 45,
    }

    experiments["v1_mimic_B"] = {
        **CLEAN,
        "EMA_PERIOD": 20,
        "EMA_SLOW_PERIOD": 60,
        "EMA_SLOPE_PERIOD": 5,
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "CHOP_SLOPE_AVG_PERIOD": 5,
        "CHOP_BOX_MIN_BARS": 0,
        "PULLBACK_TOUCH_MULT": 1.2,
        "MIN_PULLBACK_BARS": 1,
        "INITIAL_STOP_ATR_MULT": 5.0,
        "LOSERS_MAX_BARS": 45,
    }

    # ══════════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════════
    print(f"Running {len(experiments)} BOLD experiments...")
    print("=" * 140)
    print(f"{'Name':<30} {'Score':>8} {'PF':>8} {'Return%':>8} {'Sharpe':>8} {'WR%':>6} {'Trades':>7} {'W/L':>8} {'MaxDD%':>8} {'$Won':>10} {'$Lost':>10}")
    print("-" * 140)

    results_list = []
    for name, overrides in experiments.items():
        t0 = time.time()
        try:
            m = run_one(df, overrides)
            s = score(m)
            elapsed = time.time() - t0
            results_list.append({"name": name, "score": s, **m, "overrides": overrides})
            marker = " ***" if m["profit_factor"] >= 1.0 else ""
            print(f"{name:<30} {s:>8.4f} {m['profit_factor']:>8.3f} {m['total_return']:>7.2f}% "
                  f"{m['sharpe']:>8.3f} {m['win_rate']:>5.1f}% {m['total_trades']:>6} "
                  f"{m['won']}/{m['lost']:>5} {m['max_drawdown']:>7.2f}% "
                  f"{m['gross_won']:>9.0f} {m['gross_lost']:>9.0f}  ({elapsed:.1f}s){marker}")
        except Exception as e:
            print(f"{name:<30} {'ERROR':>8}  {str(e)[:80]}")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════
    valid = [r for r in results_list if r.get("score", -10) > -10]
    profitable = [r for r in valid if r.get("profit_factor", 0) >= 1.0]

    print(f"\n{'='*140}")
    print(f"PROFITABLE CONFIGS (PF >= 1.0): {len(profitable)} / {len(valid)}")
    print("=" * 140)
    if profitable:
        profitable.sort(key=lambda x: x["profit_factor"], reverse=True)
        for r in profitable:
            print(f"\n  {r['name']}")
            print(f"    PF={r['profit_factor']:.3f} | Return={r['total_return']:.2f}% | WR={r['win_rate']:.1f}% | "
                  f"Trades={r['total_trades']} ({r['won']}W/{r['lost']}L) | MaxDD={r['max_drawdown']:.2f}% | Score={r['score']:.4f}")
            params = {k: v for k, v in r["overrides"].items()
                     if not (k == "TP_ACTIVATE_ATR" and v == 100.0) and not (k == "ENABLE_ADDON" and v is False)}
            print(f"    Params: {', '.join(f'{k}={v}' for k, v in params.items())}")
    else:
        print("  NONE — see analysis below for closest configs")

    # Near-profitable (PF 0.8-1.0)
    near = [r for r in valid if 0.8 <= r.get("profit_factor", 0) < 1.0]
    if near:
        print(f"\n{'='*140}")
        print(f"NEAR-PROFITABLE (PF 0.8-1.0): {len(near)}")
        print("=" * 140)
        near.sort(key=lambda x: x["profit_factor"], reverse=True)
        for r in near[:10]:
            print(f"  {r['name']}: PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% "
                  f"WR={r['win_rate']:.1f}% Trades={r['total_trades']} Score={r['score']:.4f}")

    # Top by score
    print(f"\n{'='*140}")
    print("TOP 20 BY SCORE")
    print("=" * 140)
    valid.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(valid[:20]):
        marker = " <<<" if r.get("profit_factor", 0) >= 1.0 else ""
        print(f"  #{i+1}: {r['name']}: Score={r['score']:.4f} PF={r['profit_factor']:.3f} "
              f"Ret={r['total_return']:.2f}% WR={r['win_rate']:.1f}% Trades={r['total_trades']} DD={r['max_drawdown']:.2f}%{marker}")

    # Best by PF with enough trades
    for min_trades in [20, 30, 50]:
        enough = [r for r in valid if r.get("total_trades", 0) >= min_trades]
        if enough:
            print(f"\n{'='*140}")
            print(f"TOP 5 BY PF (trades >= {min_trades})")
            print("=" * 140)
            enough.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)
            for r in enough[:5]:
                print(f"  {r['name']}: PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% "
                      f"WR={r['win_rate']:.1f}% Trades={r['total_trades']} Score={r['score']:.4f}")

    # Block-level analysis
    print(f"\n{'='*140}")
    print("BLOCK-LEVEL ANALYSIS")
    print("=" * 140)

    blocks = {
        "Ultra-short EMA (5,8)": [r for r in valid if r["name"].startswith("ema5_") or r["name"].startswith("ema8_")],
        "Micro pullback": [r for r in valid if r["name"].startswith("micro_")],
        "Morning only": [r for r in valid if r["name"].startswith("morning_")],
        "Early cutoff": [r for r in valid if r["name"].startswith("early_cut_")],
        "Fixed hold": [r for r in valid if r["name"].startswith("fixed_hold_")],
        "No slow EMA": [r for r in valid if r["name"].startswith("no_slow_")],
        "Risk sizing": [r for r in valid if r["name"].startswith("risk_")],
        "Daily limits": [r for r in valid if r["name"].startswith("daily_")],
        "Chop avg period": [r for r in valid if r["name"].startswith("chopavg_")],
        "Stop distance": [r for r in valid if r["name"].startswith("stop_")],
        "ATR period": [r for r in valid if r["name"].startswith("atr_")],
        "Killer combos": [r for r in valid if r["name"].startswith("KILLER_")],
        "V1 mimic": [r for r in valid if r["name"].startswith("v1_mimic_")],
    }

    for block_name, block_results in blocks.items():
        if not block_results:
            continue
        best = max(block_results, key=lambda x: x.get("profit_factor", 0))
        avg_pf = sum(r.get("profit_factor", 0) for r in block_results) / len(block_results)
        profitable_count = sum(1 for r in block_results if r.get("profit_factor", 0) >= 1.0)
        print(f"\n  {block_name} ({len(block_results)} configs):")
        print(f"    Avg PF: {avg_pf:.3f} | Profitable: {profitable_count}/{len(block_results)}")
        print(f"    Best: {best['name']} → PF={best['profit_factor']:.3f} Trades={best['total_trades']} WR={best['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
