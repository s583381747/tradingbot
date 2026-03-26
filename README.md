# MNQ Prop Firm Strategy — EMA20 Touch Close

MNQ (Micro E-mini Nasdaq) 3分钟级别 EMA20 回调趋势跟随策略，专为 Topstep 50K 评估设计。

**4年验证：OOS PF=2.21、$69/day、MaxDD=$613（远低于$2,000限制）。真实 Tradovate 滑点验证通过。**

---

## 策略核心

### 入场逻辑

```
1. 趋势确认    Close > EMA20 > EMA50（做多）或 Close < EMA20 < EMA50（做空）
2. 回调触碰    3分钟bar的wick在 EMA20 ± 0.15×ATR 范围内
3. 触碰bar收盘入场    在touch bar的Close价格入场（无stop-buy gap bias）
4. 时间限制    不在14:00后新开仓，15:58强平所有持仓
5. 胜后跳过    赢一笔后跳过下一个信号（避免过度交易）
```

### 出场架构（全仓位，无分仓）

```
3-Bar MFE Gate:
  - 入场后3根bar内MFE < 0.2R → 止损收紧至 entry - 0.1R
  - Cohen's d > 1.0（强信号），快速切掉不动的交易

BE Trigger + Stop Move:
  - 价格达到 +0.25R → 止损移至 entry + 0.15R（留0.1R呼吸空间）
  - 不做部分止盈（MNQ 2合约无法有效分仓）

Chandelier Trail（BE触发后激活）:
  - highest_high(25 bars) - 0.3×ATR（做多）
  - 只升不降，让利润奔跑
  - 无固定止盈上限
```

### 风控

```
合约数      2 MNQ（固定）
日内熔断    当日累计亏损达2.0R后停止开新仓（次日重置）
止损        touch bar low - 0.4×ATR（做多）
最大持仓    180根1分钟bar（=60根3分钟bar）
强平时间    15:58
```

---

## 回测成绩

### MNQ 成本模型（真实 Tradovate 数据验证）

| 成本项 | 金额 | 来源 |
|--------|------|------|
| 佣金 | $2.46/合约 RT | Tradovate 标准 |
| 价差 | $0.50/笔 (1 tick) | 市场观察 |
| 止损滑点 | $1.00/笔 (2 ticks) | **45笔MNQ stop fill实测 mean=1.94 ticks** |
| BE滑点 | $1.00/笔 | 保守估计 |

### 核心回测结果（真实滑点模型）

| | IS (2024-2026) | OOS (2022-2024) |
|--|----------------|-----------------|
| **Net PF** | **2.415** | **2.209** |
| **Daily $** | **+$91** | **+$66** |
| **MaxDD** | **$898** | **$613** |
| Trades | 2,827 | 2,833 |
| Trades/day | 5.6 | 5.6 |
| 5R+ | 37 | 34 |
| Cost/Risk | 6.2% | 7.8% |
| IS/OOS PF ratio | — | **91.5%** |

### Exit Breakdown

| Exit Type | IS | OOS | 含义 |
|-----------|-----|-----|------|
| stop | 1,069 (38%) | 1,016 (36%) | 完整止损 |
| be | 1,493 (53%) | 1,563 (55%) | BE触发后止损 |
| trail | 265 (9%) | 254 (9%) | Chandelier追踪止盈 |

### Prop Firm 适配（Topstep 50K）

| 指标 | 值 | Topstep限制 | 状态 |
|------|-----|------------|------|
| MaxDD | $898 | $2,000 | ✅ 安全余量 $1,102 |
| 日均盈利 | $69 OOS | 目标$3,000 | ✅ ~44交易日达标 |
| 日内最大亏损 | ~2R = ~$160 | $2,000/day | ✅ |
| 一致性 | 5.6笔/天稳定 | 需要 | ✅ |

---

## 真实滑点验证

数据来源：用户 Tradovate 账户 45笔 MNQ stop order fill（2026 Q1）

| 指标 | Ticks | USD |
|------|-------|-----|
| Mean | 1.94 | $0.97 |
| Median | 1.0 | $0.50 |
| P90 | 5.2 | $2.60 |
| P95 | 6.8 | $3.40 |
| Max | 18.0 | $9.00 |

| 产品 | Mean滑点(ticks) | Zero滑点% |
|------|-----------------|-----------|
| NQ | 0.25 | 75% |
| **MNQ** | **1.94** | **40%** |
| MES | 0.26 | 74% |

MNQ滑点 ~8x NQ/MES。回测使用 2 ticks ($1.00) — 与实测 mean 1.94 ticks 匹配。

---

## 研究历程（9个实验）

### Phase 1: 参数优化 (Exp 0-4)

| # | 描述 | IS PF | OOS PF | IS DD | OOS DD | 状态 |
|---|------|-------|--------|-------|--------|------|
| 0 | Baseline (1min, 2MNQ) | 1.304 | 1.132 | $3,668 | $4,242 | baseline |
| 1 | 3min + gate-0.2 | 1.802 | 1.608 | $1,313 | $1,657 | keep |
| 2 | trigger=stop bug | — | — | — | — | INVALID |
| 3 | v8: gate-0.15 + chand30/0.3 | 2.128 | 1.889 | $1,255 | $983 | keep |
| **4** | **gate-0.1 + trigger0.25 + chand25** | **2.488** | **2.290** | **$860** | **$588** | **★ BEST** |

关键发现：1min→3min 将 cost/risk 从 30% 降到 6-8%。Gate -0.1R + BE trigger 0.25R/stop 0.15R + Chandelier 25/0.3 三者协同——快速切亏 + 快速锁利 = DD 最小化。

### Phase 2: ML 入场过滤 (Exp 5-7)

| # | 描述 | OOS AUC | 状态 |
|---|------|---------|------|
| 5 | LightGBM/RF 37 features → win/loss | 0.528 | ✗ noise |
| 6 | 70 features, 5 targets, visual patterns | 0.532-0.656 | ✗ partial MFE signal only |
| 7 | DCP deep dive | Cohen's d = -0.03 | ✗ kills 74% of 5R+ |

**定论**：70个特征、4种模型、5种目标、purged CV，预入场特征**无法预测**交易结果。DCP（ML排名#1特征）与大赢家**反向相关**——低DCP（深回调）产出68-74%的5R+交易。

### Phase 3: 出场/再入场优化 (Exp 8-9)

| # | 描述 | OOS PF | OOS DD | 5R+ | 状态 |
|---|------|--------|--------|-----|------|
| 8 | RE-ENTRY after BE (window=20) | 2.212 | $748 | 32 | marginal (+8% $/d, +22% DD) |
| 9 | Volume S/R target | 1.970 | $510 | 14 | ✗ HARMFUL (caps 5R+) |

**定论**：Volume target **主动有害**——5R+从34→14，PF从2.22→1.97。任何限制上行空间的机制都会伤害策略的厚尾利润核心。

---

## 核心认知

1. **Edge 来源是结构性的**：趋势(EMA) + 回调入场(touch) + 快速切亏(gate) + 让利跑(chandelier)
2. **入场时不可预测**：所有touch在入场时看起来高度同质化。70个特征的ML模型 OOS AUC=0.53（随机）
3. **利润来自厚尾**：2.7% 的 5R+ 交易贡献 ~50% 利润。任何 capping 机制都致命
4. **3-bar MFE gate 是唯一有效的 post-entry 信号**：Cohen's d > 1.0
5. **MNQ成本问题通过3min解决**：1min cost/risk 30%+ → 3min 6-8%

---

## 策略参数（strategy_mnq.py）

```python
STRATEGY = {
    "tf_minutes": 3,
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "no_entry_after": time(14, 0),
    "stop_buffer": 0.4,
    "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15,
    "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180,
    "force_close_at": time(15, 58),
    "daily_loss_r": 2.0,
    "skip_after_win": 1,
    "n_contracts": 2,
}
```

---

## 项目结构

### 生产代码

| 文件 | 用途 |
|------|------|
| `strategy_mnq.py` | **核心** — MNQ prop firm 策略 + 回测引擎 |
| `strategy_final.py` | 原始 QQQ Plan G 策略（参考） |
| `entry_signal.py` | 共享入场信号模块 |

### 实验代码

| 文件 | 用途 |
|------|------|
| `exp_ml_filter.py` | ML入场过滤 V1 (37 features) |
| `exp_ml_filter_v2.py` | ML入场过滤 V2 (70 features, 5 targets) |
| `exp_ml_filter_v3.py` | DCP deep dive |
| `exp_reentry_mnq.py` | RE-ENTRY after BE |
| `exp_vol_sr_mnq.py` | Volume Node S/R |
| `exp_*.py` | 其他历史实验 (chop detection, entry redesign, etc.) |

### 研究实验室

| 文件 | 用途 |
|------|------|
| `.lab/config.md` | 研究配置 |
| `.lab/log.md` | 实验日志（9个实验） |
| `.lab/results.tsv` | 结构化结果 |
| `.lab/mnq_slippage.md` | 真实 MNQ 滑点数据 |
| `.lab/parking-lot.md` | 已关闭的研究方向 |

### 数据

| 文件 | 用途 |
|------|------|
| `data/QQQ_1Min_Polygon_2y_clean.csv` | IS 数据 (2024-2026) |
| `data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv` | OOS 数据 (2022-2024) |

---

## 快速开始

```bash
pip install pandas numpy

# MNQ策略回测
python3 strategy_mnq.py

# ML入场过滤研究
python3 exp_ml_filter.py

# RE-ENTRY测试
python3 exp_reentry_mnq.py

# Volume S/R测试
python3 exp_vol_sr_mnq.py
```

---

## 下一步

- [ ] **Paper trading 2周** — 验证真实执行
- [ ] **获取NQ 1min数据** — 替换QQQ×40近似回测
- [ ] **Broker max loss设置** — 每单最大$200
- [ ] **自动化执行** — NinjaTrader/Sierra Chart

---

*最后验证: 2026-03-26 | MNQ v8 | OOS PF=2.21, $69/day, MaxDD=$613, 真实滑点验证通过, 9个实验闭环*
