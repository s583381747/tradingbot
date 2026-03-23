# Pandas → Backtrader Alignment Spec

## Pandas 逻辑（已验证盈利）

```
FOR each bar i:
  1. TREND CHECK:
     - bull = close[i] > ema20[i] AND ema20[i] > ema50[i]
     - bear = close[i] < ema20[i] AND ema20[i] < ema50[i]

  2. TOUCH CHECK (tight: 0.15 ATR tolerance):
     - long touch: low[i] <= ema20[i] + 0.15*atr[i]
                   AND low[i] >= ema20[i] - 0.5*atr[i]
     - short touch: high[i] >= ema20[i] - 0.15*atr[i]
                    AND high[i] <= ema20[i] + 0.5*atr[i]

  3. BOUNCE CHECK (next bar confirms):
     - long bounce: close[i+1] > high[i]
     - short bounce: close[i+1] < low[i]

  4. SIGNAL LINE (if touch + bounce):
     - long: signal_price = high[i] + $0.05    ← TOUCH bar's high, NOT bounce bar
             stop_price = low[i] - 0.3*atr[i]  ← TOUCH bar's low
     - short: signal_price = low[i] - $0.05
              stop_price = high[i] + 0.3*atr[i]
     - risk = |signal_price - stop_price|       ← FIXED at entry time
     - entry_atr = atr[i]                       ← FIXED at entry time

  5. SIGNAL TRIGGER (check bars i+2, i+3, i+4):
     - long: triggered if high[j] >= signal_price → entry at signal_price
     - short: triggered if low[j] <= signal_price

  6. EXIT (bar by bar from entry):
     a. STOP CHECK FIRST every bar:
        - long: if low[k] <= runner_stop → close all, PnL based on runner_stop
        - short: if high[k] >= runner_stop
     b. LOCK1 at 0.5:1 R:R (using FIXED risk from step 4):
        - target = entry + 0.5 * risk * direction
        - if hit: lock1_pnl = 0.30 * 0.5 * risk
        - move runner_stop to breakeven (entry price)
     c. LOCK2 at 1.5:1 R:R:
        - target = entry + 1.5 * risk * direction
        - if hit: lock2_pnl = 0.20 * 1.5 * risk
     d. RUNNER TRAIL (after lock1 done, bar >= 5):
        - 10-bar trailing low - 0.3 * entry_atr (FIXED atr)
        - only ratchets in favorable direction
```

## 5 个必须对齐的点

1. **prices 来自 touch bar (i), 不是 bounce bar (i+1) 或 exit-zone bar**
2. **risk 和 atr 在入场时固定，不随时间变化**
3. **每个 bar 都检查止损，不因 pending order 跳过**
4. **允许同时持多个仓位（pyramiding）或独立处理每个信号**
5. **runner trail 用 entry_atr (固定), 不用 current atr**
