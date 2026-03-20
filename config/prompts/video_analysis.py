"""
Prompt templates for Gemini video analysis.

Two-pass architecture:
  1. FIRST_PASS_PROMPT  — Full video scan: identify topics, strategies, timestamps.
  2. SECOND_PASS_PROMPT — Deep extraction on specific strategy segments identified in pass 1.

Prompts are in Chinese (matching source video language) with English structure comments.
Template variables use {curly_brace} format for str.format() / f-string interpolation.
"""

# ---------------------------------------------------------------------------
# Pass 1: Full video overview analysis
# ---------------------------------------------------------------------------
FIRST_PASS_PROMPT = """\
# 角色与目标

你是一位专业的量化交易研究员，正在观看一段中文交易教学视频。你的任务是从视频中提取所有可量化的交易策略信息，为后续的自动化回测做准备。

## 关键原则

1. **区分信息来源**：明确标注每条信息是来自「画面」（图表标注、指标参数截图）还是「口述」（讲解者说的话）。画面信息优先级高于口述。
2. **保留中文术语**：所有交易术语保持原始中文，括号内可附英文对照。例如：均线（MA）、布林带（Bollinger Bands）、缠论（Chan Theory）。
3. **参数精确性**：如果视频画面中能看到指标参数（如 MA 周期、RSI 参数），必须记录精确数值，不要猜测。如果看不清，标注「画面不清晰，需确认」。
4. **量化可行性评估**：对每个策略点评估是否可以被编程实现。完全主观的判断（如"感觉要涨"）标记为「不可量化」。

## 输出格式

请严格按照以下 JSON 结构输出。不要输出 JSON 以外的内容。

```json
{{
  "video_metadata": {{
    "title": "视频标题（从画面或标题栏读取）",
    "estimated_duration_minutes": 0,
    "language": "zh-CN",
    "content_type": "教学|实盘|复盘|指标介绍|综合",
    "overall_topic": "视频主题一句话概括"
  }},
  "strategies_overview": [
    {{
      "strategy_name": "策略名称",
      "description": "策略简要描述（2-3句话）",
      "market_applicable": "适用市场（股票/期货/外汇/加密货币/通用）",
      "timeframe": "适用周期（1分钟/5分钟/15分钟/1小时/日线/周线/不限）",
      "first_mention_timestamp": "HH:MM:SS",
      "quantifiability": "高|中|低",
      "quantifiability_reason": "为什么给这个评级"
    }}
  ],
  "technical_indicators": [
    {{
      "indicator_name": "指标中文名",
      "indicator_name_en": "指标英文名",
      "parameters": {{
        "param_name": "param_value（从画面读取的精确值）"
      }},
      "source": "画面|口述|两者",
      "usage_context": "该指标在策略中如何使用",
      "timestamp": "HH:MM:SS"
    }}
  ],
  "chart_annotations": [
    {{
      "timestamp": "HH:MM:SS",
      "annotation_type": "趋势线|支撑阻力|形态标注|箭头信号|文字标注|其他",
      "description": "标注内容和含义",
      "tradeable_signal": true,
      "signal_description": "如果可交易，描述信号条件"
    }}
  ],
  "key_timestamps": [
    {{
      "timestamp": "HH:MM:SS",
      "event": "核心策略讲解|指标参数设置|实盘演示|入场示范|出场示范|风控讲解",
      "description": "该时间点的关键内容",
      "priority": "高|中|低"
    }}
  ],
  "raw_quotes": [
    {{
      "timestamp": "HH:MM:SS",
      "quote": "讲解者的原话（与交易规则直接相关的）",
      "context": "这句话在讲什么"
    }}
  ],
  "analysis_notes": "分析者的补充说明、疑问、或需要在第二轮深入提取的内容"
}}
```

## 注意事项

- 如果视频中出现多个策略，全部提取，按出现顺序排列
- 注意视频中「正面教材」和「反面教材」的区别，标注清楚
- 如果讲解者提到其他视频或「之前讲过」的内容，在 analysis_notes 中记录
- 时间戳格式统一用 HH:MM:SS
- 如果视频画质低导致参数看不清，如实说明，不要编造数字
"""

# ---------------------------------------------------------------------------
# Pass 2: Deep extraction for specific strategy segments
# ---------------------------------------------------------------------------
SECOND_PASS_PROMPT = """\
# 角色与目标

你是一位量化交易策略工程师。你之前已经看过这个视频的整体分析，现在需要对其中的特定策略片段进行深度提取，目标是获取足够精确的规则，使其可以直接转化为回测代码。

## 第一轮分析摘要

{first_pass_summary}

## 需要深入分析的策略

{target_strategy}

## 关注的时间段

从 {start_time} 到 {end_time}

## 提取要求

请仔细观看指定时间段的视频内容，提取以下信息。如果某项信息视频中没有提及，标注「未提及」而非留空。

### 输出格式

请严格按照以下 JSON 结构输出。不要输出 JSON 以外的内容。

```json
{{
  "strategy_detail": {{
    "name": "策略名称",
    "version_note": "如果讲解者提到版本或改进，记录在此",
    "core_logic": "策略核心逻辑的详细描述（5-10句话，越详细越好）"
  }},
  "entry_conditions": {{
    "long_entry": [
      {{
        "condition_id": 1,
        "description": "做多条件描述",
        "indicator": "使用的指标",
        "operator": ">|<|==|crosses_above|crosses_below|between",
        "value": "阈值或比较对象",
        "source": "画面|口述",
        "confidence": "高|中|低",
        "confidence_reason": "置信度原因"
      }}
    ],
    "short_entry": [
      {{
        "condition_id": 1,
        "description": "做空条件描述",
        "indicator": "使用的指标",
        "operator": ">|<|==|crosses_above|crosses_below|between",
        "value": "阈值或比较对象",
        "source": "画面|口述",
        "confidence": "高|中|低",
        "confidence_reason": "置信度原因"
      }}
    ],
    "logical_combination": "AND|OR|SEQUENCE（多个条件之间的关系）",
    "sequence_description": "如果是 SEQUENCE，描述条件触发的顺序"
  }},
  "exit_conditions": {{
    "take_profit": [
      {{
        "method": "固定点数|固定比例|指标信号|跟踪止盈|目标价位",
        "value": "具体数值",
        "description": "详细说明",
        "source": "画面|口述"
      }}
    ],
    "stop_loss": [
      {{
        "method": "固定点数|固定比例|指标信号|结构止损|时间止损",
        "value": "具体数值",
        "description": "详细说明",
        "source": "画面|口述"
      }}
    ],
    "other_exit": [
      {{
        "condition": "其他出场条件描述",
        "source": "画面|口述"
      }}
    ]
  }},
  "indicator_parameters": [
    {{
      "indicator": "指标名称",
      "parameter_name": "参数名",
      "value": "参数值",
      "source": "画面|口述",
      "screenshot_readable": true,
      "notes": "补充说明"
    }}
  ],
  "position_sizing": {{
    "method": "固定手数|固定比例|凯利公式|ATR 动态|未提及",
    "details": "具体说明",
    "max_position_pct": "最大仓位比例（如提及）",
    "scaling": "是否分批建仓，如何加减仓"
  }},
  "risk_management": {{
    "max_loss_per_trade_pct": "单笔最大亏损比例",
    "max_daily_loss_pct": "每日最大亏损比例",
    "max_concurrent_positions": "最大同时持仓数",
    "other_rules": ["其他风控规则"]
  }},
  "quantifiability_assessment": {{
    "score": 0,
    "max_score": 10,
    "breakdown": {{
      "entry_clarity": "0-2分：入场条件清晰度",
      "exit_clarity": "0-2分：出场条件清晰度",
      "parameter_completeness": "0-2分：参数完整度",
      "objectivity": "0-2分：客观性（vs 主观判断）",
      "reproducibility": "0-2分：可复现性"
    }},
    "missing_for_full_score": ["达到满分还需要的信息"],
    "ambiguities": ["模糊之处，需要做假设的地方"],
    "assumptions_needed": [
      {{
        "aspect": "需要假设的方面",
        "suggested_default": "建议的默认值",
        "reasoning": "建议理由"
      }}
    ]
  }},
  "code_ready_summary": {{
    "can_implement": true,
    "blocking_issues": ["如果不能实现，列出阻塞问题"],
    "pseudocode_sketch": "用自然语言写出策略伪代码，每行一个步骤"
  }}
}}
```

## 注意事项

- 仔细观察画面中的图表：K线形态、指标线的位置关系、标注文字
- 如果讲解者在画面上画了标记，描述标记的位置和含义
- 区分「讲解者明确说的规则」和「从示例中推断的规则」，后者标注 confidence: "低"
- 对于看不清的画面参数，标注 screenshot_readable: false
- 量化评分要严格：只有完全客观、参数完整的策略才能得 8 分以上
"""

# ---------------------------------------------------------------------------
# Helper: format second pass prompt with context from first pass
# ---------------------------------------------------------------------------
def format_second_pass_prompt(
    first_pass_summary: str,
    target_strategy: str,
    start_time: str = "00:00:00",
    end_time: str = "99:99:99",
) -> str:
    """Fill in the SECOND_PASS_PROMPT template with analysis context.

    Args:
        first_pass_summary: JSON string or summary from first pass analysis.
        target_strategy: Name/description of the strategy to deep-dive on.
        start_time: Video start timestamp (HH:MM:SS) for the segment.
        end_time: Video end timestamp (HH:MM:SS) for the segment.
            Use "99:99:99" to mean "until end of video".
    """
    return SECOND_PASS_PROMPT.format(
        first_pass_summary=first_pass_summary,
        target_strategy=target_strategy,
        start_time=start_time,
        end_time=end_time,
    )
