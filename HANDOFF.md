# 项目交接说明 — QQQ EMA20 Trend Following Strategy

## 一句话概要

QQQ 1 分钟级别 EMA20 回调弹跳趋势跟随策略，2 年 SIP 数据回测 PF=2.55、+33%，经过外部审计 + 微观结构压力测试验证，待实盘接入。

---

## 1. 策略逻辑（完整描述）

### 入场

```
条件：做多（做空镜像）

1. 趋势确认:  Close > EMA20 > EMA50
2. 回调触碰:  当前 bar 的 Low 在 EMA20 上方 0.15×ATR 以内（wick 几乎亲到均线）
3. 弹跳确认:  下一根 bar 的 Close > 触碰 bar 的 High
4. 信号挂单:  在触碰 bar 的 High + $0.05 挂 stop buy order
5. 有效期:    信号 3 bars 内未触发则作废
6. 不在 15:30 后新开仓，15:58 强平所有持仓
```

### 止损

```
初始止损:  触碰 bar 的 Low - 0.3×ATR
风险 (1R): |入场价 - 止损价| ≈ $0.50-0.60
```

### 出场（分仓架构）

```
Lock (20% 仓位):
  - 固定止盈 0.3:1 R:R（约 $0.15-0.18）
  - 触发后止损移到保本 (breakeven)

Runner (80% 仓位):
  - Chandelier 追踪止损: highest_high(40 bars) - 1.0×ATR
  - 持仓 40 bar 后开始追踪，只升不降
  - 无固定止盈上限，让利润奔跑
```

### 风控

```
仓位:  每笔风险 = 1% 权益，最大仓位 = 25% 权益
日限:  当日累计亏损 ≥ 2.5R 后停止交易至次日
佣金:  IBKR $0.005/share
```

### 策略本质

高频小赌注 + 不对称收益分布：
- 20% 交易止损 (-1R)
- 64% 交易微赚微亏 (0~0.1R，lock 赚了但 runner 在 BE 出)
- 14% 交易 runner 盈利 (1R~5R)
- 2% 交易大赢 (5R+)，贡献 ~40% 总利润

---

## 2. 性能验证

### 回测结果（Plan F，含全部 bug fix）

| 指标 | 值 |
|---|---|
| PF | 2.55 |
| 2 年回报 | +33.26% |
| 胜率 | 77.1% |
| 交易总数 | 4,431 (8.9/天) |
| Avg Win / Avg Loss | $16.01 / $21.16 |

### Walk-Forward

| 时段 | PF | 回报 | 交易数 |
|---|---|---|---|
| Y1 (2024-03 → 2025-03) | 2.443 | +14.81% | 2,231 |
| Y2 (2025-03 → 2026-03) | 2.651 | +16.06% | 2,200 |

Y2 > Y1，无退化。4 个半年窗口全盈利，std=0.10。

### 压力测试（Nightmare 场景）

全部最悲观假设叠加（strict lock fill + BE $0.03 滑点 + entry bar bug fix + $0.05 入场滑点）：

| 指标 | Baseline | Nightmare |
|---|---|---|
| PF | 2.55 | 1.63 |
| Return | +33.3% | +15.6% |
| Y1→Y2 | 2.44→2.65 | 1.48→1.78 |

Nightmare 下仍盈利，Y2 仍优于 Y1。

### 实盘模拟

`live_sim.py`：1-bar 延迟 + 随机滑点 ($0.015 mean)，40 天模拟 +1.66%，20 次 Monte Carlo 全盈利 (PF≈1.55)。

---

## 3. 关键文件清单

### 生产代码（需要关注的）

| 文件 | 行数 | 用途 |
|---|---|---|
| `strategy_final.py` | 407 | **核心** — 策略逻辑 + pandas 回测引擎 |
| `live_sim.py` | 347 | 实盘模拟器（1-bar latency + slippage） |
| `stress_test.py` | 513 | 6 场景微观结构压力测试 |
| `exit_comparison.py` | 302 | 6 套出场方案 walk-forward 对比 |
| `walk_forward.py` | 231 | Walk-forward 验证框架 |
| `chart_viewer.py` | 344 | Dash 交互式可视化 (localhost:8051) |
| `download_polygon.py` | 218 | Polygon.io SIP 数据下载 |

### 研究工具（可选读）

| 文件 | 用途 |
|---|---|
| `sweep_runner.py` | 7 种 trail 方法对比（73 配置） |
| `sweep_locks.py` | Lock/runner 分仓配置优化（352 配置） |
| `sweep_pf.py` | 参数邻域稳定性扫描 |
| `exp_phase1_analysis.py` | Phase 1 组件独立统计分析 |

### 文档

| 文件 | 用途 |
|---|---|
| `STRESS_TEST_REPORT.md` | 微观结构压力测试完整报告 |
| `AUDIT_PACKAGE.md` | 发给外部 AI 审计的完整包 |
| `AUDIT_RESPONSE.md` | 对 Gemini 审计 3 个 bug 的回应 |
| `ALIGNMENT_SPEC.md` | pandas ↔ backtrader 对齐规范 |
| `.dev/journal/decisions.md` | 11 个重要技术决策及理由 |
| `.dev/journal/changelog.md` | 时间线变更日志 |

### 历史文件（可忽略）

`strategy_gen{0-16}.py` — 16 代迭代版本，保留作为参考。全部基于 backtrader，已被 pandas 引擎替代。

### 数据

| 文件 | 说明 |
|---|---|
| `data/QQQ_1Min_Polygon_2y_clean.csv` | **生产数据** — 193,926 bars，SIP 质量 |
| `data/QQQ_1Min_Polygon_2y_raw.csv` | 原始下载，含盘前盘后 |
| `data/QQQ_1Min_2y_clean.csv` | IEX 旧数据（已弃用，13% zero-range bars） |

---

## 4. 核心技术决策（为什么这样做）

### D1: 数据源 IEX → Polygon SIP
IEX 数据有 13% zero-range bars (H=L)，volume 只有 SIP 的 2%。在 IEX 上的"盈利"是假象。

### D2: 佣金 0.1% → $0.005/share
backtrader 默认 `commission=0.001` 意味着 0.1% of value = $0.50/share on QQQ $500。真实 IBKR 佣金 $0.005/share，差 **100 倍**。这是 Gen 0-15 全部亏损的根因。

### D6: 回测引擎 backtrader → pandas
backtrader 的 Stop order 有 1 bar fill 延迟，导致交易数从 5000 降到 850。pandas 精确匹配交易逻辑。

### D5+D8: 出场架构 Lock+Trail 胜过一切
- 不分仓 (100% chandelier): PF=1.13
- 分仓 (20% lock + 80% trail): PF=2.55
- Fixed RR vs Trailing: trailing 每笔 +71% 期望值
- 352 种 lock 配置测试后，1-Lock(0.3R, 20%) 最简且最优

### D4: Touch tolerance 1.0 → 0.15 ATR
宽 tolerance 让太远的信号入场，R:R 全面下降 ~10pp。0.15 ATR 要求 wick 几乎碰到 EMA20。

完整决策列表见 `.dev/journal/decisions.md`（D1-D11）。

---

## 5. 已知风险和局限

1. **BE 滑点是真实成本** — 64% 的交易在保本出场，实盘每笔约亏 $0.02-0.05 滑点。压力测试显示影响约 -11% PF。
2. **执行延迟敏感** — 需要低延迟 VPS (AWS us-east-1)，不能从本地跑。
3. **全自动化必须** — 每天 ~9 笔交易，手动不可行。
4. **容量限制** — 1 分钟级别 scalping，适合 $100K 级别资金，不适合大资金。
5. **单标的风险** — 只在 QQQ 验证，未跨标的测试（SPY 待做）。

---

## 6. 待完成事项（按优先级）

### P0: 实盘前必须

- [ ] **Pine Script 同步** — `tradingview_strategy.pine` 还是旧版本，需要同步 Plan F 逻辑用于手动监控/验证
- [ ] **SPY 交叉验证** — 用完全相同的参数（不重新调优）在 SPY 上跑，验证策略泛化性
- [ ] **live_sim.py 更新** — 同步 Plan F 参数 (chandelier 40/1.0) 和 daily 2.5R limit

### P1: 实盘接入

- [ ] **IBKR API 集成** — 用 `ib_insync` 库连接 Interactive Brokers
- [ ] **US East VPS** — AWS us-east-1 或 Equinix NY，延迟 < 10ms
- [ ] **实时数据源** — Polygon WebSocket 或 IBKR 原生 streaming
- [ ] **订单管理** — signal line → stop buy order, lock → limit order, chandelier → trailing stop
- [ ] **风控层** — 日亏损限制、连接断开保护、异常波动暂停

### P2: 增强

- [ ] **多标的扩展** — SPY, IWM, 行业 ETF
- [ ] **参数自适应** — ATR 或 volatility regime 驱动的动态参数
- [ ] **实盘 dashboard** — 实时监控交易、P&L、风控状态

---

## 7. 如何运行

```bash
cd /Users/mac/project/qqq/tradingbot

# 回测
python3 strategy_final.py

# 压力测试（6 场景并行）
python3 stress_test.py

# 实盘模拟（40天 + Monte Carlo）
python3 live_sim.py --days 40 --runs 20

# 可视化
python3 chart_viewer.py
# 然后打开 http://localhost:8051

# Walk-forward 验证
python3 walk_forward.py

# 出场方案对比
python3 exit_comparison.py
```

环境：Python 3.13, pandas, numpy, plotly, dash。无需 GPU。所有计算密集型代码用 `ProcessPoolExecutor(cpu_count())`。

---

## 8. 代码架构

```
strategy_final.py 的核心函数：

prepare_data(df) → df
  添加 EMA20, EMA50, ATR 指标

run_backtest(df, capital=100000, params=None) → dict
  返回 {pf, return_pct, trades, win_rate, trade_log, equity_curve, ...}

  内部流程：
  while bar < n:
    1. 跳过 NaN / 收盘前 / 日亏损超限
    2. 检查趋势 (EMA20 > EMA50)
    3. 检查 wick touch EMA20
    4. 检查 bounce (close > prev high)
    5. 设置 signal line + stop
    6. 等待 signal trigger (3 bars)
    7. Bug #1: 入场 bar 止损检查
    8. 仓位管理循环:
       - Force close at 15:58
       - Bug #2: 同 bar 止损/锁利冲突
       - Stop check
       - Lock: 20% at 0.3R → BE
       - Bug #3: Chandelier trail (排除当前 bar)
    9. 记录交易 + 更新日 R 亏损
```

---

*交接日期: 2026-03-23 | 最后验证: PF=2.55, Y1=2.44→Y2=2.65*
