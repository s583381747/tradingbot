# EMA20 Trend Following Strategy — Audit Package

## 请审计以下内容：
1. 策略逻辑是否有 look-ahead bias
2. 回测引擎是否有 bug
3. 统计结论是否可靠（过拟合风险）
4. 实盘可行性评估

---

## 一、策略概述

**品种**: QQQ (NASDAQ-100 ETF)
**时间框架**: 1 分钟
**数据**: Polygon.io SIP, 2024-03-22 → 2026-03-20, 193,926 bars, 500 trading days
**佣金**: IBKR $0.005/share

### 入场逻辑
1. **趋势**: close > EMA20 > EMA50 (做多) 或 close < EMA20 < EMA50 (做空)
2. **回撤触及**: candle wick 碰到 EMA20 (容差 0.15×ATR，不穿太深 >0.5×ATR)
3. **弹跳确认**: 下一根 bar close > touch bar 的 high (做多)
4. **信号线入场**: 在 touch bar high + $0.05 挂 stop buy, 3 bar 有效

### 止损
- 初始 stop = touch bar low - 0.3×ATR

### 出场（方案 F，最优）
- **Lock (20% 仓位)**: 盈利 0.3:1 R:R 时 fixed 止盈 → stop 移到 breakeven
- **Runner (80% 仓位)**: Chandelier trail = highest_high(40) - 1.0×ATR，只往上移

### 风控
- 每笔风险 1% 权益, 最大仓位 25%, 15:30 停止入场, 15:58 强制平仓

---

## 二、回测结果

### 方案 F (推荐): Fixed 0.3R Lock + Chandelier 40/1.0

| 指标 | Full 2Y | Y1 (IS) | Y2 (OOS) |
|---|---|---|---|
| PF | 2.38 | 2.34 | 2.43 |
| Return | +45.4% | +20.5% | +20.7% |
| WR | 77.0% | 76.5% | 77.4% |
| MaxDD | 0.65% | 0.48% | 0.65% |
| Trades | 6525 | 3274 | 3249 |
| Trades/day | 13.1 | 13.1 | 13.0 |

### Walk-forward 半年滚动

| Period | PF |
|---|---|
| H1-2024 | 2.31 |
| H2-2024 | 2.35 |
| H1-2025 | 2.57 |
| H2-2025 | 2.36 |
| **全部盈利, std=0.10** | |

### 方案对比 (全部经 walk-forward 验证)

| 方案 | 描述 | Full PF | Y1→Y2 Δ PF | 半年全盈利 | 过拟合风险 |
|---|---|---|---|---|---|
| A | Fixed 0.3R + Chand 30/1.5 | 2.23 | +0.04 | ✓ | LOW |
| B | Trail(10/0.8) + Trail(40/1.0) | 1.96 | +0.03 | ✓ | LOW |
| C | Trail(15/1.0) + Chand 30/1.5 | 1.85 | +0.02 | ✓ | LOW |
| D | 100% Chand 30/1.5 (不分仓) | 1.13 | -0.06 | ✗ | MEDIUM |
| E | Trail(10/0.8) + Chand 30/1.5 | 1.85 | +0.01 | ✓ | LOW |
| **F** | **Fixed 0.3R + Trail 40/1.0** | **2.38** | **+0.09** | **✓** | **LOW** |

### R 分布

| R Range | % |
|---|---|
| -1.0R (止损) | 20% |
| 0 ~ 0.1R (lock 微赚) | 64% |
| 1R+ (runner 跑出来) | 13% |
| 5R+ (大赢家) | 1.7% |

### 滑点敏感度

| Slippage | PF | Return |
|---|---|---|
| $0.00 | 2.28 | +31.2% |
| $0.02 | 1.98 | +25.1% |
| $0.05 | 1.78 | +21.0% |

### 实盘模拟 (1-bar latency + random slippage)

| 指标 | 值 |
|---|---|
| PF | 1.55 |
| Return (40天) | +1.66% |
| Monte Carlo 50次 | 50/50 全盈利 |

---

## 三、已知问题

1. **Pandas vs Backtrader 差异**: pandas engine 支持同一 bar 内信号检测+入场，backtrader 有 1-bar fill delay。pandas 结果是上限。
2. **Bounce 确认用 close[i+1] > high[i]**: 需要 bar 完成才知道 close。实盘中用 stop order 挂在 high + offset 即可。
3. **Signal trigger 假设 fill at signal price**: 实际可能有滑点。slippage 测试显示 $0.05 仍盈利。
4. **每天 ~13 笔**: 必须全自动化执行。

---

## 四、审计重点

请特别关注：
1. 回测引擎 `run_backtest()` 中的 bar-by-bar 循环是否有 look-ahead bias
2. Touch/Bounce/Signal 三步检测的时序是否正确（是否用了未来数据）
3. 止损检查是否在每根 bar 都执行（是否有跳过的情况）
4. Lock 止盈计算是否使用了入场时固定的 risk（而非当前 ATR）
5. Chandelier trail 的 highest_high 计算是否只用了过去数据
6. 仓位计算在权益减少时是否正确缩小
7. 佣金是否在每次 partial exit 都扣除

---

## 五、完整代码

见附带的 `strategy_final.py` — 单文件，~280 行，包含完整回测引擎。
运行方式: `python3 strategy_final.py`
数据: `data/QQQ_1Min_Polygon_2y_clean.csv` (Polygon SIP, 193,926 bars)
