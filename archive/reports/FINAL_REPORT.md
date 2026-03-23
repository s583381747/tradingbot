# Final Autonomous Iteration Report

## Data
- **Source**: Polygon.io SIP (consolidated, 100% market volume)
- **Period**: 2024-03-22 → 2026-03-20 (500 trading days)
- **Quality**: 193,926 bars, 388 bars/day, **0 zero-range bars** (vs IEX 13.3%)
- **Avg volume/bar**: 102,797 (vs IEX 1,611 — 64x more)
- Buy & Hold QQQ: **+30.81%**

## Generation Comparison

| Gen | Change | PF | Return | Trades | T/day | WR% | R:R | MaxDD | Best Q | p-value |
|-----|--------|-----|--------|--------|-------|-----|-----|-------|--------|---------|
| **0** | v3 baseline | 0.777 | -1.45% | 83 | 0.17 | 25.3 | 2.3 | 3.07% | 3/8 | 0.55 |
| 1 | Bug fixes (no daily reset, flicker, wick) | 0.609 | -6.98% | 240 | 0.48 | 30.0 | 1.4 | 7.20% | 0/8 | 0.47 |
| 2 | Volume confirm + wick + loose chop | 0.319 | -5.64% | 121 | 0.24 | 20.7 | 1.2 | 5.65% | 0/9 | 0.54 |
| 3 | SIP ATR recalibration + wide stop | 0.694 | -1.99% | 85 | 0.17 | 25.9 | 2.0 | 3.05% | 4/8 | 0.45 |
| 4 | Resistance breakout filter | 0.335 | -0.15% | 3 | 0.01 | 33.3 | 0.7 | 0.25% | 0/7 | 0.66 |
| **5** | Tight stop + fast loser exit | **0.806** | -1.20% | 85 | 0.17 | 22.4 | **2.8** | **2.62%** | 3/8 | 0.44 |

## Key Findings

### 1. IEX 数据制造了虚假 edge
| 指标 | IEX | Polygon SIP |
|------|-----|-------------|
| Gen 0 PF | 1.063 (盈利) | **0.777 (亏损)** |
| Gen 0 Trades | 64 | 83 |
| Zero-range bars | 13.3% | 0% |

IEX 的 13% H=L bars 人为缩小了 ATR、减少了交易触发，让策略看起来盈利。换到真实数据后 edge 消失。

### 2. 所有代的 p-value 都 > 0.4 — 无统计显著性
没有任何一代能排除"随机运气"假设。

### 3. Walk-forward 全部失败
没有任何一代在 Y1 和 Y2 都达到 PF>1.0。

### 4. 核心诊断：WR 始终低于盈亏平衡线
- 策略 R:R 在 2.0-2.8 之间 → 需要 WR 26-33% 才盈利
- 实际 WR 始终 22-26% → 差 3-5 个百分点
- 这不是参数问题，是**入场信号质量不足**

### 5. "Bug fixes" 实际上是有用的过滤器
- 每日重置 chop box → 防止隔夜 gap 影响
- Trend flicker 重置 → 自然淘汰弱趋势中的入场
- Close-based 回撤 → 比 wick 更严格，过滤噪音
- 这些不是 bug，是策略能 PF=0.8 的原因

### 6. 季度规律：Q4 强、Q1 弱
- 2024Q4: PF=4.8-6.1（所有代都盈利）
- 2025Q1: PF=0.19-0.22（所有代都大亏）
- 策略在强趋势市场有效，在震荡/反转市场失效

## 诚实结论

**EMA20 回撤策略在 1 分钟 QQQ SIP 数据上没有可靠的交易 edge。**

原因分析：
1. 1 分钟级别噪音太大，EMA20 的方向性信号被淹没
2. QQQ 作为高流动性 ETF，1 分钟的价格效率很高，简单技术形态被快速套利
3. 25% 胜率需要 3:1 以上的 R:R 才能盈利，但日内时间有限（6.5 小时），大波段难以形成

## 下一步建议

### 如果要继续 1 分钟 QQQ：
1. **完全重新设计入场** — EMA 回撤不够，需要更强的信号（微观结构、订单流、bid-ask spread 变化）
2. **多时间框架** — 用 15 分钟确认趋势，1 分钟精确入场
3. **事件驱动** — 结合经济数据发布、开盘/收盘特定时段

### 如果愿意换时间框架：
1. **5 分钟** — 减少噪音，EMA20 更有意义
2. **15 分钟** — 信号质量更高，每天仍有 2-3 个入场机会

### 如果换标的：
1. **个股** — 趋势性更强，EMA 回撤策略更适合
2. **期货（ES/NQ）** — 成本更低，杠杆可控

## 文件索引
```
strategy_gen0.py  — 基线（v3 参数）
strategy_gen1.py  — Bug fix（降级到 PF=0.61）
strategy_gen2.py  — 量价确认（PF=0.32）
strategy_gen3.py  — SIP ATR 校准（PF=0.69）
strategy_gen4.py  — 阻力突破（3 笔太少）
strategy_gen5.py  — 激进砍亏（PF=0.81 最佳）
reports/gen{0-5}_report.txt — 各代审计报告
data/QQQ_1Min_Polygon_2y_clean.csv — SIP 清洗数据
data/QQQ_1Min_Polygon_2y_raw.csv — SIP 原始数据
data/QQQ_1Min_2y_clean.csv — IEX 数据（对比参考）
```
