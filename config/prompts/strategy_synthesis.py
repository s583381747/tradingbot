"""
Prompt templates for knowledge synthesis and code generation.

SYNTHESIS_PROMPT  — Combine multiple video analyses into a coherent trading system.
CODEGEN_PROMPT    — Convert a TradingStrategy into backtrader Python code.
"""

# ---------------------------------------------------------------------------
# Synthesis: merge all video analyses into a trading system document
# ---------------------------------------------------------------------------
SYNTHESIS_PROMPT = """\
# 角色与目标

你是一位资深量化交易系统架构师。你收到了从多个交易教学视频中提取的策略分析结果（JSON 格式）。你的任务是将这些碎片化的信息整合成一份结构化的「交易系统手册」。

## 输入数据

以下是所有视频分析结果：

{video_analyses_json}

## 整合原则

1. **频率优先**：如果同一策略在多个视频中反复出现，将其列为核心策略。出现次数越多，权重越高。
2. **新视频优先**：当不同视频中的规则存在矛盾时（例如参数不同、条件冲突），优先采用较新视频的版本，并在「版本说明」中记录变更。
3. **合并同类项**：不同视频中对同一指标的不同用法应合并到同一策略条目下，按场景区分。
4. **丢弃低质量**：量化评分低于 4 分（满分 10 分）的策略片段，仅保留在「低优先级 / 待确认」部分，不纳入主系统。
5. **补全参数**：如果核心策略缺少关键参数，根据行业惯例给出建议默认值，但必须标注「假设值 - 需回测验证」。

## 输出格式

请严格按照以下 JSON 结构输出。不要输出 JSON 以外的内容。

```json
{{
  "trading_system": {{
    "name": "系统名称（根据核心策略自动命名）",
    "version": "1.0",
    "source_video_count": 0,
    "creation_date": "{creation_date}",
    "applicable_markets": ["适用市场列表"],
    "primary_timeframes": ["主要交易周期"],
    "system_overview": "系统整体描述（3-5句话）"
  }},
  "core_strategies": [
    {{
      "strategy_id": "S001",
      "name": "策略名称",
      "source_videos": ["出现该策略的视频标题列表"],
      "frequency_count": 0,
      "description": "策略完整描述",
      "market": "适用市场",
      "timeframe": "适用周期",
      "indicators": [
        {{
          "name": "指标名称",
          "parameters": {{}},
          "role": "该指标在策略中的作用"
        }}
      ],
      "entry_rules": {{
        "long": [
          {{
            "rule_id": "L1",
            "condition": "条件描述",
            "indicator": "指标",
            "operator": "操作符",
            "value": "值",
            "is_assumed": false
          }}
        ],
        "short": [
          {{
            "rule_id": "S1",
            "condition": "条件描述",
            "indicator": "指标",
            "operator": "操作符",
            "value": "值",
            "is_assumed": false
          }}
        ],
        "combination": "AND|OR|SEQUENCE"
      }},
      "exit_rules": {{
        "take_profit": [
          {{
            "method": "方法",
            "value": "值",
            "is_assumed": false
          }}
        ],
        "stop_loss": [
          {{
            "method": "方法",
            "value": "值",
            "is_assumed": false
          }}
        ],
        "trailing_stop": {{
          "enabled": false,
          "method": "方法",
          "value": "值"
        }}
      }},
      "position_sizing": {{
        "method": "方法",
        "value": "值",
        "is_assumed": false
      }},
      "risk_management": {{
        "max_loss_per_trade_pct": 0.0,
        "max_daily_loss_pct": 0.0,
        "max_concurrent_positions": 1
      }},
      "quantifiability_score": 0,
      "implementation_readiness": "ready|needs_assumptions|not_implementable",
      "version_notes": "版本说明、来源视频的差异记录"
    }}
  ],
  "contradictions_resolved": [
    {{
      "topic": "矛盾主题",
      "video_a": "视频A的说法",
      "video_b": "视频B的说法",
      "resolution": "采用哪个及原因",
      "video_dates": {{
        "a": "视频A日期",
        "b": "视频B日期"
      }}
    }}
  ],
  "low_priority_fragments": [
    {{
      "description": "低优先级策略片段",
      "source_video": "来源视频",
      "reason_excluded": "排除原因",
      "quantifiability_score": 0
    }}
  ],
  "missing_information": [
    {{
      "strategy_id": "关联策略ID",
      "missing_item": "缺失信息描述",
      "impact": "对回测的影响程度（高/中/低）",
      "suggested_default": "建议默认值",
      "suggestion_basis": "建议依据"
    }}
  ],
  "implementation_priority": [
    {{
      "rank": 1,
      "strategy_id": "策略ID",
      "strategy_name": "策略名称",
      "reason": "优先实现的理由",
      "estimated_complexity": "低|中|高"
    }}
  ]
}}
```

## 注意事项

- 同一个讲解者在不同视频中可能用不同名称指代同一策略，注意合并
- 保留所有原始中文术语
- 对于「假设值」，选择业内最保守的默认参数
- implementation_priority 按照：量化评分 > 出现频率 > 复杂度（越简单越优先）排序
"""

# ---------------------------------------------------------------------------
# Code generation: TradingStrategy -> backtrader code
# ---------------------------------------------------------------------------
CODEGEN_PROMPT = """\
# 角色与目标

你是一位资深 Python 量化开发工程师，专精 backtrader 框架。你的任务是将下面的交易策略定义转换为可运行的 backtrader 策略代码。

## 策略定义

```json
{strategy_json}
```

## 代码生成要求

### 第一步：伪代码

先用中文写出策略的伪代码逻辑，每行一个步骤。这一步用于确认你理解了策略规则。格式：

```
伪代码：
1. 初始化：加载XX指标，参数为...
2. 每根K线：检查...
3. 入场条件：当... 且... 时，做多/做空
4. 出场条件：当... 时，平仓
5. 风控：...
```

### 第二步：backtrader 代码

然后输出完整的 Python 代码。要求：

1. **类结构**：继承 `bt.Strategy`，使用 `params` 元组定义所有可调参数
2. **指标初始化**：在 `__init__` 中使用 backtrader 内置指标或自定义指标
3. **信号逻辑**：在 `next` 中实现入场/出场判断
4. **仓位管理**：使用 `self.broker.getvalue()` 计算仓位大小
5. **风控逻辑**：实现止损止盈（优先用 `self.sell(exectype=bt.Order.Stop)` 等订单类型）
6. **日志**：关键操作使用 `self.log()` 记录
7. **类型注解**：添加 Python 类型注解
8. **参数化**：所有魔法数字都必须是 params 中的参数
9. **附带运行脚本**：在同一输出中提供 `if __name__ == "__main__"` 入口，包含数据加载和 cerebro 配置

### 代码模板（必须遵循的结构）

```python
import backtrader as bt
import datetime


class {strategy_class_name}(bt.Strategy):
    \\"\\"\\"
    {strategy_description}

    Source: Auto-generated from video analysis pipeline.
    Quantifiability score: {quantifiability_score}/10
    \\"\\"\\"

    params = (
        # -- 指标参数 --
        # (param_name, default_value),
        # -- 交易参数 --
        # ('stake_pct', 0.95),  # 仓位比例
        # -- 风控参数 --
        # ('stop_loss_pct', 0.02),  # 止损比例
        # ('take_profit_pct', 0.04),  # 止盈比例
    )

    def __init__(self) -> None:
        # 保存收盘价引用
        self.dataclose = self.datas[0].close
        # 订单跟踪
        self.order: bt.Order | None = None
        self.buy_price: float | None = None
        self.buy_comm: float | None = None
        # TODO: 初始化指标

    def log(self, txt: str, dt: datetime.date | None = None) -> None:
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{{dt.isoformat()}} {{txt}}")

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY  @ {{order.executed.price:.2f}}, cost={{order.executed.value:.2f}}, comm={{order.executed.comm:.2f}}")
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(f"SELL @ {{order.executed.price:.2f}}, cost={{order.executed.value:.2f}}, comm={{order.executed.comm:.2f}}")
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order failed: {{order.getstatusname()}}")
        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        if not trade.isclosed:
            return
        self.log(f"TRADE P&L gross={{trade.pnl:.2f}}, net={{trade.pnlcomm:.2f}}")

    def next(self) -> None:
        if self.order:
            return  # 有未完成订单，跳过
        # TODO: 实现策略逻辑


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy({strategy_class_name})

    # 数据加载 - 使用 Yahoo Finance 数据
    # TODO: 根据策略适用市场选择合适的标的和时间段
    data = bt.feeds.YahooFinanceCSVData(
        dataname="data.csv",
        fromdate=datetime.datetime(2020, 1, 1),
        todate=datetime.datetime(2024, 12, 31),
    )
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    print(f"Starting Portfolio Value: {{cerebro.broker.getvalue():.2f}}")
    results = cerebro.run()
    strat = results[0]
    print(f"Final Portfolio Value:    {{cerebro.broker.getvalue():.2f}}")

    # 输出分析结果
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print(f"Sharpe Ratio:  {{sharpe.get('sharperatio', 'N/A')}}")
    print(f"Max Drawdown:  {{dd.max.drawdown:.2f}}%")
    print(f"Total Trades:  {{trades.total.total}}")

    cerebro.plot()
```

### 重要约束

- 如果策略定义中有标注 `is_assumed: true` 的参数，在代码注释中标注 `# ASSUMED - 需回测验证`
- 如果量化评分低于 6 分，在类文档字符串中加入警告
- 所有字符串使用 f-string 格式化
- 不要使用 `self.buy()` 的 `size` 参数做仓位管理，统一用 `sizer` 或手动计算
- 确保代码可以直接运行（无语法错误）

## 输出

先输出伪代码，然后输出完整 Python 代码。两部分用 `---` 分隔。
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_synthesis_prompt(video_analyses_json: str, creation_date: str) -> str:
    """Fill in the SYNTHESIS_PROMPT with video analysis data.

    Args:
        video_analyses_json: JSON string containing all video analyses
            (list of first/second pass results).
        creation_date: ISO date string for the system document.
    """
    return SYNTHESIS_PROMPT.format(
        video_analyses_json=video_analyses_json,
        creation_date=creation_date,
    )


def format_codegen_prompt(
    strategy_json: str,
    strategy_class_name: str,
    strategy_description: str,
    quantifiability_score: int,
) -> str:
    """Fill in the CODEGEN_PROMPT with strategy details.

    Args:
        strategy_json: JSON string of the TradingStrategy object from synthesis.
        strategy_class_name: PascalCase class name for the backtrader strategy.
        strategy_description: One-line description for the class docstring.
        quantifiability_score: 0-10 score from the analysis pipeline.
    """
    return CODEGEN_PROMPT.format(
        strategy_json=strategy_json,
        strategy_class_name=strategy_class_name,
        strategy_description=strategy_description,
        quantifiability_score=quantifiability_score,
    )
