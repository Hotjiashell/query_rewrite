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

也可以直接使用命令行运行：

```bash
export OPENAI_API_KEY='your-api-key'
python -m promptEov \
  --dataset data/dialog_example.json \
  --model your-model-name \
  --base-url https://your-llm.example/v1 \
  --iterations 3 \
  --output-dir promptEov/runs \
  --analysis-concurrency 4
```

如果项目根目录存在 `config.json`，可以直接复用其中的 `llm`、`query_generation.input_path` 和 `retrieval` 配置，不再重复填写模型、API Key、数据集路径和检索地址：

```bash
export OPENAI_API_KEY='your-api-key'
python -m promptEov --config config.json
```

命令行参数会覆盖配置文件中的同名设置。`embedding` 如果由检索服务内部使用，PromptEov 不会直接调用它；若配置文件中存在 `embedding` 段，会由检索服务自行处理，客户端无需额外传参。

如果要使用自定义初始提示词，增加 `--initial-prompt-file path/to/prompt.txt`。可用 `--temperature`、`--retrieval-url`、`--retrieval-timeout` 调整模型和检索服务参数。

每轮在 `promptEov/runs/iteration_N.json` 保存 prompt、逐条检索结果、top-10 命中标记、bad cases、分析报告和新 prompt。每个 badcase 单独调用一次分析器，多个 badcase 的分析会并行执行，再按原顺序汇总后交给优化器；可通过 `analysis_concurrency` 调整并行度。分析器输出按原文保存，不要求 JSON；优化器基于当前轮 prompt 继续优化。可通过 `generate_query(dialogue_prompt, dialogue)` 与 `retrieve(query)` 注入自定义实现。
