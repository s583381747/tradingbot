"""Sweep trail width to find where R:R actually lets profits run."""
from __future__ import annotations
import importlib, sys, os, time, functools
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import backtrader as bt
import pandas as pd

print = functools.partial(print, flush=True)
DATA = os.path.abspath("data/QQQ_1Min_Polygon_2y_clean.csv")


def run_one(args):
    trail_mult, stop_atr, activate, data_path = args
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    if "strategy_gen8" in sys.modules:
        del sys.modules["strategy_gen8"]
    mod = importlib.import_module("strategy_gen8")
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df, datetime=None, open="Open", high="High",
                                low="Low", close="Close", volume="Volume", openinterest=-1)
    cerebro.adddata(data)
    cerebro.addstrategy(mod.Strategy,
                        trail_ema_atr_mult=trail_mult,
                        stop_buffer_atr=stop_atr,
                        trail_activate_atr=activate)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    results = cerebro.run()
    ta = results[0].analyzers.ta.get_analysis()
    tt = ta.get("total", {}).get("total", 0)
    w = ta.get("won", {}).get("total", 0)
    l = ta.get("lost", {}).get("total", 0)
    gw = ta.get("won", {}).get("pnl", {}).get("total", 0) or 0
    gl = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0) or 0)
    pf = gw / gl if gl > 0 else 0
    ret = (cerebro.broker.getvalue() - 100000) / 1000
    aw = gw / w if w > 0 else 0
    al = gl / l if l > 0 else 0
    return {
        "trail": trail_mult, "stop": stop_atr, "act": activate,
        "trades": tt, "won": w, "lost": l,
        "pf": round(pf, 3), "ret": round(ret, 2),
        "aw": round(aw, 1), "al": round(al, 1),
        "rr": round(aw / al, 2) if al > 0 else 0,
    }


def main():
    tasks = []
    # Trail width sweep (structural stop buffer=0.3)
    for trail in [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        tasks.append((trail, 0.3, 1.0, DATA))

    # Buffer sweep (trail=4.0)
    for buf in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        tasks.append((4.0, buf, 1.0, DATA))

    # Activate sweep (trail=4.0, buf=0.3)
    for act in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        tasks.append((4.0, 0.3, act, DATA))

    # Combined: wide trail + tight stop
    for trail in [3.0, 4.0, 5.0, 6.0]:
        for buf in [0.2, 0.3, 0.5]:
            tasks.append((trail, buf, 1.0, DATA))

    print(f"Running {len(tasks)} configs on {cpu_count()} CPUs...")
    print(f"{'Trail':<8}{'Stop':<8}{'Act':<8}{'Trades':<8}{'W/L':<10}{'PF':<8}{'Ret%':<8}{'AvgW':<8}{'AvgL':<8}{'R:R':<8}")
    print("-" * 82)

    results = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            m = " ***" if r["pf"] >= 1.0 else ""
            print(f"{r['trail']:<8}{r['stop']:<8}{r['act']:<8}{r['trades']:<8}"
                  f"{r['won']}/{r['lost']:<8}{r['pf']:<8}{r['ret']:<8}"
                  f"${r['aw']:<7}${r['al']:<7}{r['rr']:<8}{m}")

    print(f"\nProfitable (PF>=1.0):")
    for r in sorted([x for x in results if x["pf"] >= 1.0], key=lambda x: x["pf"], reverse=True):
        print(f"  Trail={r['trail']} Stop={r['stop']} Act={r['act']} → PF={r['pf']} R:R={r['rr']} Trades={r['trades']}")


if __name__ == "__main__":
    main()
