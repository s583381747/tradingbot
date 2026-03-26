"""
Entry Mechanism Visualizer — shows exactly how touch → bounce → trigger works.

Generates an HTML file with interactive candlestick charts showing:
  - EMA20/EMA50
  - Touch bar, bounce bar, trigger bar highlighted
  - Signal line, stop level
  - Open vs sig gap on trigger bar
"""
from __future__ import annotations
import functools, datetime as dt, json
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch, check_bounce, calc_signal_line

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

PARAMS = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "signal_offset": 0.05, "stop_buffer": 0.3,
    "signal_valid_bars": 3,
    "no_entry_after": dt.time(15, 30),
    "daily_loss_r": 2.5,
}


def collect_examples(df):
    """Collect entry signal examples with context bars."""
    p = PARAMS
    df = add_indicators(df, p)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_p = df["Open"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    timestamps = df.index
    n = len(df)

    examples = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0
    current_date = None

    while bar < n - 200:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]:
            bar += 1; continue
        d_date = dates[bar]
        if current_date != d_date:
            current_date = d_date; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]:
            bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend != 1:
            bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        bb = bar + 1
        if bb >= n: bar += 1; continue
        if not check_bounce(trend, close[bb], high[bar], low[bar]):
            bar += 1; continue

        sl = calc_signal_line(trend, high[bar], low[bar], a,
                              p["signal_offset"], p["stop_buffer"])
        if sl is None: bar += 1; continue
        sig, stop, risk = sl

        for j in range(1, p["signal_valid_bars"] + 1):
            cb = bb + j
            if cb >= n: break
            if high[cb] >= sig:
                gap = open_p[cb] - sig
                ctx_start = max(0, bar - 8)
                ctx_end = min(n, cb + 5)

                ctx_bars = []
                for i in range(ctx_start, ctx_end):
                    ctx_bars.append({
                        "t": str(timestamps[i]),
                        "o": round(float(open_p[i]), 2),
                        "h": round(float(high[i]), 2),
                        "l": round(float(low[i]), 2),
                        "c": round(float(close[i]), 2),
                        "e20": round(float(ema[i]), 2),
                        "e50": round(float(ema_s[i]), 2),
                    })

                examples.append({
                    "ti": bar - ctx_start,
                    "bi": bb - ctx_start,
                    "tri": cb - ctx_start,
                    "sig": round(float(sig), 2),
                    "stop": round(float(stop), 2),
                    "otr": round(float(open_p[cb]), 2),
                    "gap": round(float(gap), 4),
                    "gpr": round(float(gap / risk * 100), 1),
                    "risk": round(float(risk), 2),
                    "date": str(dates[bar]),
                    "time": str(times[bar]),
                    "bars": ctx_bars,
                })
                bar = cb + 1
                break
        else:
            bar += 1; continue

    return examples


def pick_examples(examples):
    picked = {}
    by_gap = sorted(examples, key=lambda x: x["gap"])

    intrabar = [e for e in by_gap if e["gap"] < -0.01]
    if intrabar:
        picked["intrabar"] = intrabar[len(intrabar) // 2]

    small = [e for e in by_gap if 0.01 < e["gap"] < 0.05]
    if small:
        picked["small_gap"] = small[len(small) // 2]

    medium = [e for e in by_gap if 0.05 < e["gap"] < 0.15]
    if medium:
        picked["medium_gap"] = medium[len(medium) // 2]

    large = [e for e in by_gap if 0.15 < e["gap"] < 0.30]
    if large:
        picked["large_gap"] = large[len(large) // 2]

    huge = [e for e in by_gap if e["gap"] > 0.30]
    if huge:
        picked["huge_gap"] = huge[len(huge) // 2]

    return picked


def generate_html(picked):
    cat_labels = {
        "intrabar": "Intrabar trigger — Open < sig, NO gap issue",
        "small_gap": "Small gap — $0.01-$0.05",
        "medium_gap": "Medium gap — $0.05-$0.15 (typical, 23% of triggers)",
        "large_gap": "Large gap — $0.15-$0.30 (25% of triggers)",
        "huge_gap": "Huge gap — $0.30+ (22% of triggers)",
    }

    panels_json = json.dumps([
        {
            "id": cat,
            "label": cat_labels.get(cat, cat),
            "data": ex,
        }
        for cat, ex in picked.items()
    ])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Entry Mechanism — Gap Visualization</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; margin:0; padding:20px;
         background:#1a1a2e; color:#e0e0e0; }}
  h1 {{ text-align:center; color:#00d4ff; }}
  .sub {{ text-align:center; color:#888; margin-bottom:20px; font-size:14px; }}
  .panel {{ background:#16213e; border-radius:12px; padding:20px; margin-bottom:25px; }}
  .panel h2 {{ color:#00d4ff; margin:0 0 8px; font-size:17px; }}
  .tags {{ display:flex; gap:12px; flex-wrap:wrap; font-size:12px; margin-bottom:8px; }}
  .tags span {{ background:#0f3460; padding:3px 10px; border-radius:5px; }}
  .chart {{ height:420px; }}
  .expl {{ background:#0f3460; border-radius:10px; padding:18px; margin-bottom:25px;
           line-height:1.7; font-size:14px; }}
  .expl code {{ background:#1a1a2e; padding:2px 6px; border-radius:4px; color:#00d4ff; }}
  .warn {{ background:#f443361a; border-left:3px solid #f44336; padding:10px 15px;
           margin:12px 0; border-radius:0 8px 8px 0; }}
  .legend {{ display:flex; gap:14px; justify-content:center; margin:16px 0; flex-wrap:wrap;
             font-size:13px; }}
  .li {{ display:flex; align-items:center; gap:5px; }}
  .lc {{ width:14px; height:14px; border-radius:3px; display:inline-block; }}
</style>
</head>
<body>

<h1>Entry Mechanism — Gap Visualization</h1>
<p class="sub">QQQ EMA20 Trend Following — Touch → Bounce → Trigger flow</p>

<div class="expl">
  <strong>Entry flow (Long):</strong><br>
  1. <span class="lc" style="background:#2196F3"></span> <b>Touch bar</b>: Low reaches EMA20<br>
  2. <span class="lc" style="background:#9C27B0"></span> <b>Bounce bar</b>: Close &gt; touch bar High<br>
  3. <span class="lc" style="background:#FF9800"></span> <b>Trigger bar</b>: High &ge; signal line
     (<code>touch_high + $0.05</code>)<br>
  4. Backtest enters at <b>sig</b> (cyan dashed line)<br>
  <div class="warn">
    <b>Problem:</b> 68% of trigger bars open <i>above</i> sig.
    A real stop-buy fills at Open, not sig.
    The <span style="color:#f44336">red shaded area</span> = money the backtest gives away for free.
  </div>
</div>

<div class="legend">
  <span class="li"><span class="lc" style="background:#2196F3"></span> Touch</span>
  <span class="li"><span class="lc" style="background:#9C27B0"></span> Bounce</span>
  <span class="li"><span class="lc" style="background:#FF9800"></span> Trigger</span>
  <span class="li"><span class="lc" style="background:#00BCD4"></span> Signal (sig)</span>
  <span class="li"><span class="lc" style="background:#f44336"></span> Stop loss</span>
  <span class="li"><span class="lc" style="background:#4CAF50"></span> EMA20</span>
  <span class="li"><span class="lc" style="background:#FF980088"></span> EMA50</span>
</div>

<div id="root"></div>

<script>
const panels = {panels_json};

panels.forEach((p, idx) => {{
  // create DOM
  const sec = document.createElement('div');
  sec.className = 'panel';
  const gapVal = p.data.gap;
  const gapColor = gapVal <= 0.005 ? '#4CAF50' : (gapVal < 0.15 ? '#FF9800' : '#f44336');
  sec.innerHTML = `<h2>${"${p.label}"}</h2>
    <div class="tags">
      <span>Date: ${{p.data.date}} ${{p.data.time}}</span>
      <span>Signal: $${{p.data.sig}}</span>
      <span>Trigger Open: $${{p.data.otr}}</span>
      <span style="color:${{gapColor}};font-weight:bold">Gap: $${{p.data.gap.toFixed(4)}} (${{p.data.gpr}}% of 1R)</span>
      <span>Risk: $${{p.data.risk}}</span>
    </div>
    <div id="ch${{idx}}" class="chart"></div>`;
  document.getElementById('root').appendChild(sec);

  const d = p.data;
  const bars = d.bars;
  const times = bars.map(b => b.t.split(' ')[1] || b.t);  // show time only
  const timeFull = bars.map(b => b.t);

  // candlestick trace
  const candlestick = {{
    x: timeFull,
    open: bars.map(b => b.o),
    high: bars.map(b => b.h),
    low: bars.map(b => b.l),
    close: bars.map(b => b.c),
    type: 'candlestick',
    increasing: {{ line: {{ color: '#26a69a' }}, fillcolor: '#26a69a' }},
    decreasing: {{ line: {{ color: '#ef5350' }}, fillcolor: '#ef5350' }},
    name: 'QQQ',
    showlegend: false,
  }};

  // EMA traces
  const ema20 = {{
    x: timeFull, y: bars.map(b => b.e20),
    mode: 'lines', line: {{ color: '#4CAF50', width: 2 }},
    name: 'EMA20', showlegend: false,
  }};
  const ema50 = {{
    x: timeFull, y: bars.map(b => b.e50),
    mode: 'lines', line: {{ color: '#FF9800', width: 1.5, dash: 'dot' }},
    name: 'EMA50', showlegend: false,
  }};

  // Signal line
  const sigLine = {{
    x: [timeFull[d.ti], timeFull[Math.min(d.tri + 2, bars.length - 1)]],
    y: [d.sig, d.sig],
    mode: 'lines', line: {{ color: '#00BCD4', width: 2, dash: 'dash' }},
    showlegend: false,
  }};

  // Stop line
  const stopLine = {{
    x: [timeFull[d.ti], timeFull[Math.min(d.tri + 2, bars.length - 1)]],
    y: [d.stop, d.stop],
    mode: 'lines', line: {{ color: '#f44336', width: 1.5, dash: 'dot' }},
    showlegend: false,
  }};

  // Open marker on trigger bar
  const openMarker = {{
    x: [timeFull[d.tri]],
    y: [d.otr],
    mode: 'markers',
    marker: {{ color: '#f44336', size: 10, symbol: 'diamond' }},
    name: 'Real fill (Open)',
    showlegend: false,
  }};

  // Shapes: highlight bars + gap rect
  const shapes = [];

  // Touch bar highlight
  shapes.push({{
    type: 'rect',
    x0: timeFull[Math.max(0, d.ti)], x1: timeFull[Math.min(d.ti, bars.length-1)],
    y0: 0, y1: 1, yref: 'paper',
    fillcolor: 'rgba(33,150,243,0.12)', line: {{ width: 0 }},
  }});
  // Bounce bar
  shapes.push({{
    type: 'rect',
    x0: timeFull[d.bi], x1: timeFull[d.bi],
    y0: 0, y1: 1, yref: 'paper',
    fillcolor: 'rgba(156,39,176,0.12)', line: {{ width: 0 }},
  }});
  // Trigger bar
  shapes.push({{
    type: 'rect',
    x0: timeFull[d.tri], x1: timeFull[d.tri],
    y0: 0, y1: 1, yref: 'paper',
    fillcolor: 'rgba(255,152,0,0.12)', line: {{ width: 0 }},
  }});

  // Gap rectangle (sig to open on trigger bar)
  if (d.gap > 0.005) {{
    const gapLo = d.sig;
    const gapHi = d.otr;
    // Extend slightly around trigger bar
    const x0i = Math.max(0, d.tri - 1);
    const x1i = Math.min(d.tri + 1, bars.length - 1);
    shapes.push({{
      type: 'rect',
      x0: timeFull[x0i], x1: timeFull[x1i],
      y0: gapLo, y1: gapHi,
      fillcolor: 'rgba(244,67,54,0.25)',
      line: {{ color: 'rgba(244,67,54,0.6)', width: 1.5 }},
    }});
  }}

  const annotations = [
    {{
      x: timeFull[d.ti], y: bars[d.ti].l,
      text: 'TOUCH', showarrow: true, arrowhead: 2,
      ax: 0, ay: 30, font: {{ color: '#2196F3', size: 12, family: 'Arial Black' }},
      arrowcolor: '#2196F3',
    }},
    {{
      x: timeFull[d.bi], y: bars[d.bi].h,
      text: 'BOUNCE', showarrow: true, arrowhead: 2,
      ax: 0, ay: -30, font: {{ color: '#9C27B0', size: 12, family: 'Arial Black' }},
      arrowcolor: '#9C27B0',
    }},
    {{
      x: timeFull[d.tri], y: bars[d.tri].h,
      text: 'TRIGGER', showarrow: true, arrowhead: 2,
      ax: 35, ay: -30, font: {{ color: '#FF9800', size: 12, family: 'Arial Black' }},
      arrowcolor: '#FF9800',
    }},
    {{
      x: timeFull[Math.max(0, d.ti - 1)], y: d.sig,
      text: 'sig=$' + d.sig.toFixed(2),
      showarrow: false,
      font: {{ color: '#00BCD4', size: 11 }},
      xanchor: 'right', xshift: -5,
    }},
  ];

  // Gap label
  if (d.gap > 0.005) {{
    annotations.push({{
      x: timeFull[Math.min(d.tri + 1, bars.length - 1)],
      y: (d.sig + d.otr) / 2,
      text: '<b>GAP $' + d.gap.toFixed(3) + '</b><br>' + d.gpr + '% of 1R',
      showarrow: true, arrowhead: 0,
      ax: 55, ay: 0,
      font: {{ color: '#f44336', size: 12 }},
      arrowcolor: '#f44336',
      bordercolor: '#f44336', borderwidth: 1, borderpad: 4,
      bgcolor: 'rgba(244,67,54,0.15)',
    }});

    // Real fill marker label
    annotations.push({{
      x: timeFull[d.tri], y: d.otr,
      text: 'Real fill $' + d.otr.toFixed(2),
      showarrow: true, arrowhead: 2,
      ax: -60, ay: -20,
      font: {{ color: '#f44336', size: 10 }},
      arrowcolor: '#f44336',
    }});
  }}

  // Y range: tight around price action
  const allY = bars.flatMap(b => [b.h, b.l, b.e20, b.e50]).concat([d.sig, d.stop]);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yPad = (yMax - yMin) * 0.15;

  const layout = {{
    paper_bgcolor: '#16213e',
    plot_bgcolor: '#0f3460',
    font: {{ color: '#e0e0e0' }},
    xaxis: {{
      gridcolor: '#1a3a5c',
      rangeslider: {{ visible: false }},
      tickangle: -45,
      tickfont: {{ size: 9 }},
    }},
    yaxis: {{
      gridcolor: '#1a3a5c',
      tickformat: '$.2f',
      range: [yMin - yPad, yMax + yPad],
    }},
    margin: {{ l: 65, r: 40, t: 10, b: 55 }},
    shapes: shapes,
    annotations: annotations,
    showlegend: false,
  }};

  Plotly.newPlot('ch' + idx,
    [candlestick, ema20, ema50, sigLine, stopLine, openMarker],
    layout, {{ responsive: true }});
}});
</script>
</body>
</html>"""


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars")

    print("Collecting entry examples...")
    examples = collect_examples(df)
    print(f"Found {len(examples)} long entry examples")

    print("Picking representative examples...")
    picked = pick_examples(examples)
    for cat, ex in picked.items():
        print(f"  {cat}: gap=${ex['gap']:.4f} ({ex['gpr']:.1f}% of 1R) on {ex['date']}")

    print("Generating HTML...")
    html = generate_html(picked)

    outpath = "entry_visual.html"
    with open(outpath, "w") as f:
        f.write(html)
    print(f"\nSaved to {outpath}")
    print("Open in browser to view.")


if __name__ == "__main__":
    main()
