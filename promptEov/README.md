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

每轮在 `promptEov/runs/iteration_N.json` 保存 prompt、逐条检索结果、top-10 命中标记、bad cases、分析报告和新 prompt。分析器输出按原文保存，不要求 JSON；优化器使用固定的初始 prompt（`initial_prompt`），而不是上一轮 prompt。可通过 `generate_query(dialogue_prompt, dialogue)` 与 `retrieve(query)` 注入自定义实现。
