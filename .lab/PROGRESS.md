# Research Progress — Touch Close Strategy Optimization

*Updated after all parallel experiments completed*

## 当前最佳策略 (v2.1)
```
入场:  Touch bar close (EMA20 回调触碰即入场)
过滤:  14:00 cutoff + skip-after-win
管理:  入场后 3 bar，若 MFE < 0.3R → 止损收紧到 -0.3R
锁利:  5% 仓位 @ 0.1R → 移 BE
追踪:  95% runner, Chandelier 40/0.5

4 年: PF=1.596, +33.6%, $100K→$133,596, 年化 7.5%, MaxDD 1.58%
IS:   PF=1.655, +16.0%
OOS:  PF=1.537, +15.2%
Long PF=1.49, Short PF=1.73
```

---

## OOS 验证通过的改善（可叠加候选）

| # | 方向 | 最佳配置 | IS PF | OOS PF | 效果 |
|---|------|---------|-------|--------|------|
| 1 | **Close Position** ★ | DCP≥0.5 + slope5>0 | 1.872 | 1.788 | touch bar 收盘强+趋势性收盘 |
| 2 | **RE-ENTRY** | w10/cd1/re1 | 1.695 | 1.595 | BE 后 10bar 内再 touch 再入场 |
| 3 | **Volume Node S/R** | vol_node < 1.0 ATR | 1.715 | 1.617 | 在高成交量价位附近入场 |
| 4 | **Gate 加仓** | G=0.3R Add+50% | 1.658 | 1.582 | 3bar 确认后追加 50% 仓位 |
| 5 | **Dynamic sizing** | RecentPF | +13.7% Ret/DD | +13.7% Ret/DD | 仅适用于期货/杠杆 |

---

## 完整实验记录

### 入场机制
1. ✅ Touch close 入场解决 gap 问题 (PF 1.19→1.275→1.617)
2. ❌ 更大 offset 降低 PF
3. ❌ 去掉 bounce 确认 PF=1.02
4. ❌ Retest limit 直接亏钱

### 出场架构
5. ✅ Runner 越大越好，5%@0.1R lock 最优
6. ❌ 334 分仓 PF=1.02
7. ❌ 固定 TP 全亏，ATR trail 不如 chandelier
8. ❌ BE 替代方案（放宽/MFE/time）全部劣于 exact BE
9. ❌ Chandelier 提前启动截断大赢家

### Chop 过滤（4 轮全部失败）
10. ❌ 15 个 bar 级别指标 Cohen's d < 0.1
11. ❌ 日级别 regime look-ahead 去除后无效
12. ❌ Price box 实时检测在噪声范围
13. ❌ 结构性趋势健康 8 个指标 Cohen's d < 0.1
14. ❌ Volume 10 个指标 Cohen's d < 0.15

### 有效发现
15. ✅ 时间过滤: 14:00 cutoff + skip-after-win (MaxDD 减半)
16. ✅ 3-bar MFE gate → tighten (Cohen's d > 1.0, IS+OOS 一致)
17. ✅ Close Position DCP: touch bar 收盘位置+趋势 (IS 1.872, OOS 1.788) ★
18. ✅ RE-ENTRY after BE: w10/cd1/re1 (IS 1.695, OOS 1.595)
19. ✅ Volume Node S/R: 高成交量价位 (IS 1.715, OOS 1.617)
20. ✅ Gate pass 加仓: G=0.3R +50% (IS 1.658, OOS 1.582)
21. ✅ Dynamic sizing RecentPF (仅期货): Ret/DD +13.7%
22. ❌ HTF 过滤（3-30min）全部降低 IS PF
23. ❌ QQQ 动态仓位被 position cap 限死

---

## 待做: 叠加测试
将 #17 + #18 + #19 + #20 组合，做 4 年验证 + stress test

## 文件清单
```
exp_entry_mode.py, exp_gap_analysis.py, exp_entry_redesign.py
exp_entry_rethink.py, exp_entry_visual.py
exp_touch_exit.py, exp_touch_multi_exit.py, exp_touch_validate.py
exp_edge_analysis.py, exp_chop_detect.py, exp_chop_realtime.py
exp_chop_audit.py, exp_pricebox.py, exp_trend_alive.py
exp_volume.py, exp_early_exit.py, exp_mtf.py
exp_phase0.py, exp_phase1.py
exp_reentry.py (旧入场,废弃), exp_reentry_v2.py ✅
exp_gate_add.py ✅, exp_sr_filter.py ✅
exp_close_position.py ✅, exp_dynamic_size.py, exp_dynamic_size_nocap.py
exp_dynamic_size_v2.py ✅
```
