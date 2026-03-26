"""
NQ Combination Sweep — combine top single-param winners.

Top winners from sweep:
1. gate_tighten=0.0 (score 18.56) — BE on gate fail instead of small loss
2. skip_after_win=0 (score 15.13) — take every signal, +96$/d
3. skip_after_win=2 (score 17.87) — skip 2, DD=$528
4. be_trigger=0.20/stop=0.15 (score 16.60)
5. gate_mfe=0.25 (score 14.49)
"""
from exp_nq_tune import *


def main():
    print("Loading NQ data...")
    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)
    nq_is = nq[nq.index >= "2024-01-01"]
    nq_oos = nq[(nq.index >= "2022-01-01") & (nq.index < "2024-01-01")]

    # Baseline
    ri = run(nq_is, DEFAULT); ro = run(nq_oos, DEFAULT)
    bs = score(ro)
    print(f"\nBASELINE: IS PF={ri['pf']:.3f} $/d={ri['dpnl']:+.1f} DD=${ri['dd']:.0f} 5R+={ri['b5']} | "
          f"OOS PF={ro['pf']:.3f} $/d={ro['dpnl']:+.1f} DD=${ro['dd']:.0f} 5R+={ro['b5']} | Score={bs}")

    combos = [
        # Name, overrides
        ("A: gate_tighten=0.0", {"gate_tighten": 0.0}),
        ("B: gate_tighten=-0.05", {"gate_tighten": -0.05}),
        ("C: skip=0", {"skip_after_win": 0}),
        ("D: skip=2", {"skip_after_win": 2}),
        ("E: be_trig=0.20/stop=0.15", {"be_trigger_r": 0.20, "be_stop_r": 0.15}),
        ("F: gate_mfe=0.25", {"gate_mfe": 0.25}),

        # ═══ COMBINATIONS ═══
        ("G: A+C (gate0.0 + skip0)", {"gate_tighten": 0.0, "skip_after_win": 0}),
        ("H: A+D (gate0.0 + skip2)", {"gate_tighten": 0.0, "skip_after_win": 2}),
        ("I: A+E (gate0.0 + be0.20)", {"gate_tighten": 0.0, "be_trigger_r": 0.20, "be_stop_r": 0.15}),
        ("J: A+F (gate0.0 + mfe0.25)", {"gate_tighten": 0.0, "gate_mfe": 0.25}),
        ("K: A+E+F (gate0.0 + be0.20 + mfe0.25)", {"gate_tighten": 0.0, "be_trigger_r": 0.20, "be_stop_r": 0.15, "gate_mfe": 0.25}),
        ("L: A+C+E (gate0.0 + skip0 + be0.20)", {"gate_tighten": 0.0, "skip_after_win": 0, "be_trigger_r": 0.20, "be_stop_r": 0.15}),
        ("M: A+D+E (gate0.0 + skip2 + be0.20)", {"gate_tighten": 0.0, "skip_after_win": 2, "be_trigger_r": 0.20, "be_stop_r": 0.15}),
        ("N: B+C (gate-0.05 + skip0)", {"gate_tighten": -0.05, "skip_after_win": 0}),
        ("O: B+D (gate-0.05 + skip2)", {"gate_tighten": -0.05, "skip_after_win": 2}),
        ("P: B+E (gate-0.05 + be0.20)", {"gate_tighten": -0.05, "be_trigger_r": 0.20, "be_stop_r": 0.15}),
        ("Q: A+C+E+F (all top)", {"gate_tighten": 0.0, "skip_after_win": 0, "be_trigger_r": 0.20, "be_stop_r": 0.15, "gate_mfe": 0.25}),
        ("R: A+D+E+F (all but skip=2)", {"gate_tighten": 0.0, "skip_after_win": 2, "be_trigger_r": 0.20, "be_stop_r": 0.15, "gate_mfe": 0.25}),

        # Explore gate_tighten between -0.05 and 0.0 with best combos
        ("S: gate=-0.03 + be0.20", {"gate_tighten": -0.03, "be_trigger_r": 0.20, "be_stop_r": 0.15}),
        ("T: gate=-0.03 + be0.20 + mfe0.25", {"gate_tighten": -0.03, "be_trigger_r": 0.20, "be_stop_r": 0.15, "gate_mfe": 0.25}),
        ("U: gate=0.0 + be0.20 + mfe0.30", {"gate_tighten": 0.0, "be_trigger_r": 0.20, "be_stop_r": 0.15, "gate_mfe": 0.30}),

        # Also test be_trigger=0.20/stop=0.10 (more room)
        ("V: gate0.0 + be_trig=0.20/stop=0.10", {"gate_tighten": 0.0, "be_trigger_r": 0.20, "be_stop_r": 0.10}),
        ("W: gate0.0 + be_trig=0.20/stop=0.10 + mfe0.25", {"gate_tighten": 0.0, "be_trigger_r": 0.20, "be_stop_r": 0.10, "gate_mfe": 0.25}),
    ]

    print(f"\n{'Name':>45} {'IS_PF':>7} {'IS_DD':>7} {'IS_$/d':>8} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4} {'IS/OOS':>7}")
    print(f"{'baseline':>45} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ri['dpnl']:>+8.1f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {bs:>7.2f} {ro['b5']:>4} {ro['pf']/ri['pf']*100:>6.1f}%")

    best_score = bs; best_name = "baseline"
    for name, overrides in combos:
        s = {**DEFAULT, **overrides}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        ratio = ro["pf"] / ri["pf"] * 100 if ri["pf"] > 0 else 0
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_name = name
        print(f"{name:>45} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ri['dpnl']:>+8.1f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4} {ratio:>6.1f}%{tag}")

    print(f"\n{'='*100}")
    print(f"BEST COMBO: {best_name} → score={best_score:.2f}")
    print(f"{'='*100}")

    # Run best in detail
    print(f"\n--- BEST COMBO DETAIL ---")
    for name, overrides in combos:
        if name == best_name:
            s = {**DEFAULT, **overrides}
            print(f"  Params: {overrides}")
            ri = run(nq_is, s); ro = run(nq_oos, s)
            print(f"  IS:  PF={ri['pf']:.3f} $/d={ri['dpnl']:+.1f} DD=${ri['dd']:.0f} N={ri['n']} 5R+={ri['b5']}")
            print(f"  OOS: PF={ro['pf']:.3f} $/d={ro['dpnl']:+.1f} DD=${ro['dd']:.0f} N={ro['n']} 5R+={ro['b5']}")
            print(f"  Score: {score(ro):.2f}")
            print(f"  IS/OOS PF ratio: {ro['pf']/ri['pf']*100:.1f}%")
            dd_ok = "✅" if ro["dd"] <= 2000 else "❌"
            print(f"  Topstep 50K DD: ${ro['dd']:.0f} / $2,000 → {dd_ok}")
            break

    # Also show top 5 by score
    print(f"\n--- TOP 5 BY SCORE ---")
    results = []
    for name, overrides in combos:
        s = {**DEFAULT, **overrides}
        ri = run(nq_is, s); ro = run(nq_oos, s)
        results.append((name, score(ro), ro["pf"], ro["dd"], ro["dpnl"], ro["b5"], ri["pf"]))
    results.sort(key=lambda x: x[1], reverse=True)
    for name, sc, pf, dd, dpnl, b5, is_pf in results[:5]:
        print(f"  {sc:>7.2f} | OOS PF={pf:.3f} DD=${dd:.0f} $/d={dpnl:+.1f} 5R+={b5} IS/OOS={pf/is_pf*100:.0f}% | {name}")


if __name__ == "__main__":
    main()
