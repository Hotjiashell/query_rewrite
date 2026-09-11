"""Prompt templates used by the self-evolution pipeline.

直接编辑本文件即可定制默认的初始、分析器和优化器提示词；运行时仍可通过
``run_evolution`` / ``PromptEov`` 参数覆盖。
"""

INITIAL_PROMPT = """
你是一个智能客服。对话内容反应了用户的某些问题，你需要根据对话内容构造一个query，用来检索能够解决用户问题的案例。
将你的结果用```json```代码块包裹，格式如下：
```json
{
    "query": "你生成的query"
}
```
对话内容:
{dialogue}
""".strip()

ANALYZER_PROMPT = """
请分析以下 top-10 未召回案例，找出当前检索 query 提示词的问题，并提出具体、可执行的改进建议。
无需输出 JSON，可使用自然语言、分点或 Markdown。

未召回案例：
{bad_cases}
""".strip()

OPTIMIZER_PROMPT = """
请根据分析报告优化初始提示词，使其能够更好地生成案例检索 query。
请保留有效约束，吸收报告中的改进建议，并只输出完整的新提示词文本，不要解释修改过程。

初始提示词：
{initial_prompt}

分析报告：
{analysis}
""".strip()
