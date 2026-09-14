# PromptEov

```python
from promptEov import run_evolution
from prompt import BASELINE_PROMPT
from openai import OpenAI

client = OpenAI()
def llm(text):
    return client.chat.completions.create(model='your-model', messages=[{'role':'user','content':text}]).choices[0].message.content
run_evolution('data/dialog_example.json', BASELINE_PROMPT, llm=llm, iterations=3)
```

每轮在 `promptEov/runs/iteration_N.json` 保存 prompt、逐条检索结果、top-10 命中标记、bad cases、分析报告和新 prompt。每个 badcase 单独调用一次分析器，多个 badcase 的分析会并行执行，再按原顺序汇总后交给优化器；可通过 `analysis_concurrency` 调整并行度。分析器输出按原文保存，不要求 JSON；优化器基于当前轮 prompt 继续优化。可通过 `generate_query(dialogue_prompt, dialogue)` 与 `retrieve(query)` 注入自定义实现。
