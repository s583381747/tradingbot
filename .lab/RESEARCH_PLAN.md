# Research Plan v2 — Touch Close Strategy Optimization

*CTO 审计后修订版*

---

## 1. 现状总结

### 策略画像
```
入场:  Touch bar close（价格触碰 EMA20 时收盘价入场）
止损:  touch_low - 0.3×ATR
锁利:  5% 仓位 @ 0.1R → 移 BE
Runner: 95% 仓位, Chandelier 40/0.5 追踪
PF = 1.617 (IS), 1.575 (OOS), 1.594 (4年)
```

### 盈利结构
```
                  笔数    占比     总 R     角色
initial_stop     1337   23.7%   -1404R    全额亏损
be_stop          4000   70.8%    -139R    微亏（均值 MFE 1.20R）
trail_stop        312    5.5%   +2366R    全部利润
─────────────────────────────────────────────────
净计              5649            +823R    PF = 1.533 (R-based)
```

### 已证实
- ✅ Touch close 入场解决了 gap 问题，Long/Short 都盈利
- ✅ Runner 越大越好（单调关系，334 分仓 PF=1.02）
- ✅ 策略是极端尾部驱动（top 1% = 108% of profit）

### 已证伪
- ❌ 15 个 chop 指标在 **bar 级别** 无法区分单笔赢/输（Cohen's d < 0.1）
- ❌ 多次止盈、固定 TP、ATR trail 全部劣于当前架构

### 未验证（CTO 审计发现）
- ❓ 日级别 chop regime 是否有效？（bar 级别无效 ≠ 日级别无效）
- ❓ BE 出场后价格怎么走了？（放宽 BE 是否有意义）
- ❓ Bar 41 出场后价格怎么走了？（chandelier 是否"过早止出"）
- ❓ 被 BE 杀死的赢家 — 到底有多少是"真误杀"vs"正确止出"？

---

## 2. 执行计划

### 总体框架

```
Phase 0: 前置分析（3 个关键问题 + 时间过滤）— 决定后续方向
Phase 1: 单变量出场实验（基于 Phase 0 的结论选择性执行）
Phase 2: 联合优化（承认 BE/Chandelier 耦合，做 grid search）
Phase 3: Trade-during 适应性管理（有前置统计分析门槛）
Phase 4: 完整验证
```

---

### Phase 0: 前置分析（先回答再动手）

**所有后续 Phase 是否执行、怎么执行，取决于 Phase 0 的结论。**

#### 0A. BE 出场后价格走向分析

**问题**: 4000 笔 BE 出场交易，出场后价格怎么走了？

**方法**:
- 对每笔 BE 交易，记录出场后 10/20/40/60 bars 的价格走向
- 计算：如果不在 BE 止出，而是继续持仓到 bar+N，结果是多少 R？
- 分桶：出场后继续有利方向 vs 继续反向 的比例

**决策标准**:
- 如果 > 40% 的 BE 交易出场后 20 bar 内回到 +1R → Phase 1A（放宽 BE）有意义
- 如果 > 70% 出场后继续反向 → BE 是正确的，Phase 1A 缩减为仅测 BE+buffer

#### 0B. Bar 41 出场后价格走向分析

**问题**: 312 笔 trail stop 中位持仓 41 bar，这是"过早止出"还是"正确捕捉顶部"？

**方法**:
- 对每笔 trail stop 交易，记录出场后 10/20/40 bars 的价格走向
- 计算：如果不在此处止出，还能多赚多少 R？或会亏回多少？

**决策标准**:
- 如果 > 40% 的 trail stop 出场后趋势继续 → Phase 1B（改善 chandelier）有意义
- 如果 > 70% 出场后趋势确实结束 → Chandelier 已经最优，Phase 1B 缩减

#### 0C. 日级别 Chop Regime 分析

**问题**: bar 级别 chop 指标无效，但整天维度的 chop 是否能识别？

**方法**:
- 按交易日分组，计算每天的: 平均 ADX、日 range/ATR、日内 EMA 交叉次数、日 PF
- 将日子分为 "趋势日"（ADX > 25, range > 2ATR）vs "chop 日"（ADX < 20, range < 1.5ATR）
- 比较两类日子的 PF、5R+ 频率、平均 R

**决策标准**:
- 趋势日 PF > 2.0 且 chop 日 PF < 1.0 → 日级别过滤有价值，纳入 Phase 2
- 差异 < 20% → 日级别也没用，彻底放弃 chop 过滤

#### 0D. 时间过滤（快速实验，结果 bake in）

最简单、最不可能过拟合的改善，先做并固化到 baseline。

| 实验 | 描述 | 已知数据 |
|------|------|---------|
| 0D-1 | 不在 14:00 后入场 | 14:xx PF=0.674, 168 笔 |
| 0D-2 | 不在 13:30 后入场 | 更保守 |
| 0D-3 | Win 后跳过 1 笔 | win-after-win PF=0.411, 316 笔 |
| 0D-4 | 0D-1 + 0D-3 组合 | |

**约束**: 5R+ 不减少 > 10%，trades/day > 5。
**如果有效**: 更新 baseline，后续所有实验在新 baseline 上进行。

#### Phase 0 产出

写一份 `0_analysis_report.md`，包含：
1. 三个前置问题的数据和结论
2. 每个后续 Phase 的 go/no-go 决定
3. 更新后的 baseline（含时间过滤）

---

### Phase 1: 单变量出场实验

**前置条件**: Phase 0 完成，根据 0A/0B/0C 结论决定哪些子项执行。

#### 1A. BE 替代方案

**仅在 Phase 0A 结论为"放宽有意义"时执行完整版。否则仅测 1A-4 和 1A-5。**

| 实验 | 描述 | 实盘复杂度 |
|------|------|-----------|
| 1A-1 | 不移 BE（lock 后止损保持原位）**← 最重要，架构级决策** | 简单 |
| 1A-2 | BE - 0.3R（移到亏 0.3R 位置） | 简单 |
| 1A-3 | BE - 0.5R | 简单 |
| 1A-4 | BE + 0.1R（微利位置） | 简单 |
| 1A-5 | BE + 0.2R | 简单 |
| 1A-6 | MFE-based: 到过 1R 后才移 BE | 中等 |
| 1A-7 | MFE-based: 到过 2R 后才移 BE | 中等 |
| 1A-8 | Time-based: 10 bar 后移 BE | 简单 |
| 1A-9 | Time-based: 20 bar 后移 BE | 简单 |
| 1A-10 | 渐进: 每 5 bar 止损上移 0.1R（从 -1R 开始） | 中等 |

**量化目标**: trail stop 数从 312 增到 > 400，同时 PF > 1.5。
**约束**: 5R+ ≥ 180，Max Trailing DD ≤ baseline × 1.5。
**失败标准**: 所有方案 PF < 1.5 → BE 是必要的，保留当前架构。

#### 1B. Chandelier 优化

**仅在 Phase 0B 结论为"存在过早止出"时执行完整版。否则仅测 1B-7/1B-8。**

| 实验 | 描述 | 实盘复杂度 |
|------|------|-----------|
| 1B-1 | 立刻启动 chandelier, mult=2.0 | 简单 |
| 1B-2 | 立刻启动, mult=1.5 | 简单 |
| 1B-3 | 立刻启动, mult=1.0 | 简单 |
| 1B-4 | Bar 10 启动, mult=1.0 | 简单 |
| 1B-5 | Bar 20 启动, mult=0.8 | 简单 |
| 1B-6 | Bar 20 启动, mult=0.5 | 简单 |
| 1B-7 | 渐进: 2.0×ATR → 0.5×ATR over 60 bars（线性收紧） | 中等 |
| 1B-8 | 渐进: 1.5×ATR → 0.5×ATR over 40 bars | 中等 |
| 1B-9 | 双阶段: bar 1-20 用 EMA20 止损, bar 20+ 用 chandelier | 复杂 |
| 1B-10 | MFE-based: 到过 2R 后才启动 chandelier | 中等 |

**量化目标**: winner 持仓时间 std > 15 bars（当前几乎为 0），trail stop 均值 R > 5R。
**约束**: 5R+ ≥ 180，PF > 1.5，实盘复杂度优先选简单/中等。
**失败标准**: 所有方案 PF < 1.5 → 当前 chandelier 已最优。

#### 1C. Lock 机制重新评估

**依赖 1A 结果。如果 1A-1（不移 BE）胜出，lock 可能完全不需要。**

| 实验 | 描述 |
|------|------|
| 1C-1 | 无 lock, 无 BE（纯 chandelier + 初始止损） |
| 1C-2 | 无 lock, 到 1R 后移到 -0.5R |
| 1C-3 | 无 lock, 渐进止损（每 5 bar 上移 0.1R） |

---

### Phase 2: 联合优化

**Phase 1A 和 1B 是耦合的**——改变 BE 会改变活到 chandelier 的交易集，所以单变量结论在组合时可能失效。

**方法**:
1. 从 1A 选 top 3 个 BE 配置（含 baseline）
2. 从 1B 选 top 3 个 chandelier 配置（含 baseline）
3. 从 0D 选有效的时间过滤
4. 做 3×3 = 9 个 BE×Chandelier grid search
5. 如果 0C 日级别 chop 有效，每个 grid 点再叠加 chop 过滤（9→18）
6. 如果 1C 表明不需要 lock，额外跑无 lock 版本

**量化目标**: PF > 1.8 (IS)
**约束**: Max DD ≤ baseline × 1.5, 5R+ ≥ 180, Long PF > 1.0, Short PF > 1.0

**过拟合检查**: Phase 2 top 配置必须在 OOS 上重现 IS 改善的至少 50%。
- 例：IS 从 1.6 → 1.9 (+0.3)，OOS 必须至少 1.6 → 1.75 (+0.15)
- 不满足 → 标记为过拟合，降级或丢弃

---

### Phase 3: Trade-During 适应性管理

**硬性前置条件**: 先做统计分析，通过门槛后才做实验。

#### 3-Pre: 统计分析

对三类交易（initial_stop / be_stop / trail_stop），比较入场后 3/5/10 bar 的行为：
- MFE 分布（KS test）
- 有利方向移动速度
- 首次到达 0.3R/0.5R/1R 的 bar 数

**决策门槛**:
- 任何指标在三类之间 KS test p < 0.01 且效应量 Cohen's d > 0.3 → 该指标可用于 early exit
- 所有指标 p > 0.05 或 d < 0.15 → Phase 3 跳过

#### 3-Exp: 条件实验（仅在 3-Pre 通过后执行）

| 实验 | 描述 | 前置分析依据 |
|------|------|-------------|
| 3-1 | 入场后 N bar 未过 X×R，主动平仓 | MFE 速度差异 |
| 3-2 | 价格穿回 EMA20，平仓 | 趋势失效信号 |
| 3-3 | EMA20 斜率反向，收紧止损 | 趋势减弱 |
| 3-4 | ATR 暴涨 > 2×入场 ATR，收紧止损 | regime 变化 |

**约束**: trail stop 交易的 count 和均值 R 不能下降 > 10%。

---

### Phase 4: 完整验证

对 Phase 2 的最优配置（最多 2 个候选）：

#### 4A. Walk-Forward
- IS Y1 vs Y2（Polygon 2024-2026）
- 季度 PF（8-9 个季度）
- 半年滚动（7 个窗口）

#### 4B. Out-of-Sample
- Barchart 2022-2024 全期
- OOS Y1 vs Y2
- IS vs OOS PF 偏差 < 20%

#### 4C. Stress Test
- BE bleed ($0.03)
- Entry slip ($0.03)
- Strict lock fill
- Nightmare（全叠加）
- Nightmare walk-forward Y1 vs Y2

#### 4D. 参数稳定性
- 每个关键参数 ±20% sweep
- 所有点 PF > 1.2

#### 4E. Risk 指标
- Max Trailing DD (R)
- Loss streak P95 和 Max
- 日内最大亏损
- 连续亏损月数

#### 通过标准
```
硬性（必须全部满足）:
  IS PF > 1.5
  OOS PF > 1.3
  Nightmare PF > 1.15
  Nightmare Y2 PF > 1.0
  Long PF > 1.0 (IS + OOS)
  Short PF > 1.0 (IS + OOS)
  参数 ±20% 全部 PF > 1.2
  Max DD ≤ baseline × 1.5
  OOS 重现 IS 改善 ≥ 50%

软性（期望满足）:
  IS PF > 1.8
  OOS PF > 1.5
  季度盈利率 > 75%
  Nightmare PF > 1.3
  实盘复杂度: 简单或中等
```

---

## 3. 技术实现

### 测量脚本 `.lab/measure.py`

每个实验统一输出：
```
PF | Ret% | Trades | T/day | WR% | Long_PF | Short_PF | 5R+ |
init_stop_n | init_stop_R | be_stop_n | be_stop_R | trail_stop_n | trail_stop_R |
Max_DD_R | Nightmare_PF
```

### 实验引擎 `exp_research.py`

基于 `exp_touch_validate.py`，新增可配置项：
```python
exit_config = {
    # BE 机制
    "be_type": "exact",       # exact / offset / mfe_based / time_based / progressive / none
    "be_offset": 0.0,         # BE 偏移量（R 单位）
    "be_mfe_threshold": 0,    # MFE-based: 到过 N×R 后才移 BE
    "be_time_threshold": 0,   # Time-based: N bars 后才移 BE
    "be_progressive_step": 0, # 每 N bars 上移 0.1R

    # Chandelier
    "chand_start_bar": 40,    # 启动 bar
    "chand_start_mult": 0.5,  # 初始 mult
    "chand_end_mult": 0.5,    # 最终 mult（渐进时不同）
    "chand_tighten_bars": 0,  # 从 start 到 end 的过渡 bar 数（0=固定）

    # Lock
    "lock_rr": 0.1,
    "lock_pct": 0.05,

    # 时间过滤
    "no_entry_after": time(15, 30),
    "skip_after_win": 0,

    # Trade-during
    "early_exit_bars": 0,     # N bars 后未过 threshold 则平仓
    "early_exit_threshold": 0, # R threshold

    # 实盘复杂度标注
    "complexity": "simple",   # simple / medium / complex
}
```

### 执行纪律

每个实验：
1. git commit（含配置描述）
2. 运行 measure.py → 记录 primary + secondary + Max DD
3. Keep/Discard 判断 → log.md + results.tsv
4. Discard → git reset --hard HEAD~1
5. 每 10 个实验 re-validate current best

---

## 4. 预计实验数量

```
Phase 0: 3 分析 + 4 实验 = 7
Phase 1A: 3-10（取决于 0A 结论）
Phase 1B: 3-10（取决于 0B 结论）
Phase 1C: 0-3（取决于 1A 结论）
Phase 2: 9-18 grid
Phase 3: 0-4（取决于 3-Pre 结论）
Phase 4: 1-2 validation runs
──────────────────────────────
Total: 25-55 experiments
```

---

## 5. 风险和缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Phase 0 三个前置分析全部否定 | 低 | Phase 1 大幅缩减 | 执行 minimal Phase 1（仅 1A-1, 1A-4, 1B-7），然后接受 PF≈1.6 |
| BE 替代方案增加 Max DD | 高 | 实盘不可用 | 每个实验记录 Max DD，硬约束 ≤ 1.5× baseline |
| 过拟合到 IS | 中 | OOS 不重现 | Phase 2 top 必须 OOS 重现 50%+ |
| Phase 1A/1B 耦合导致组合失效 | 中 | Phase 2 结果不如预期 | 预期 Phase 2 结果与 Phase 1 不同，做 full grid |
| Phase 3 伤害 trail stop | 中 | 核心利润源被削弱 | 硬性前置 KS test 门槛 + trail stop 约束 |
| 复杂方案实盘无法执行 | 低 | 研究浪费 | 每个实验标注复杂度，优先简单方案 |

---

## 6. 不做什么

1. 不再做 bar 级别入场 chop 过滤（已证伪）
2. 不做多次止盈 / 334 分仓（已证伪）
3. 不用 ATR trail / 固定 TP（已证伪）
4. 不增加入场复杂度（touch close 已最优）
5. 不做 > 3 个参数同时变化的实验（过拟合风险）

---

## 7. 决策树

```
Phase 0A: BE 出场后价格走向
  ├── > 40% 回到 +1R → 执行完整 Phase 1A（10 个实验）
  └── > 70% 继续反向 → 仅测 1A-4, 1A-5（BE+buffer），跳过 1A-1~3

Phase 0B: Trail stop 出场后价格走向
  ├── > 40% 趋势继续 → 执行完整 Phase 1B（10 个实验）
  └── > 70% 趋势结束 → 仅测 1B-7, 1B-8（渐进 chandelier），跳过其他

Phase 0C: 日级别 chop
  ├── 趋势日 PF > 2.0 且 chop 日 PF < 1.0 → 纳入 Phase 2 grid
  └── 差异 < 20% → 彻底放弃 chop 过滤

Phase 0D: 时间过滤
  ├── 任何配置 PF > 1.65 且 5R+ ≥ 185 → Bake in 到新 baseline
  └── 无显著改善 → 保持原 baseline

Phase 1A 最优 + Phase 1B 最优 → Phase 2 联合 grid
Phase 2 top → OOS 检查 → 重现 ≥ 50%? → Phase 4 验证
                         → 不重现 → 标记过拟合，使用次优配置
```
