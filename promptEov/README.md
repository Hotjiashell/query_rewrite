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
  --analysis-concurrency 4 \
  --analysis-batch-size 50
```

如果项目根目录存在 `config.json`，可以直接复用其中的 `llm`、`query_generation.input_path` 和 `retrieval` 配置，不再重复填写模型、API Key、数据集路径和检索地址：

```bash
export OPENAI_API_KEY='your-api-key'
python -m promptEov --config config.json
```

命令行参数会覆盖配置文件中的同名设置。`embedding` 如果由检索服务内部使用，PromptEov 不会直接调用它；若配置文件中存在 `embedding` 段，会由检索服务自行处理，客户端无需额外传参。

运行时会在 stderr 显示每轮的样本处理、badcase 分析和 prompt 优化进度，同时在输出目录写入 `evolution.log`。日志包含运行、迭代、批次、LLM 调用耗时和异常信息。作为 Python 库调用时默认也显示进度；如需关闭，可传入 `progress=False`：

```python
run_evolution(..., progress=False)
```

如果要使用自定义初始提示词，增加 `--initial-prompt-file path/to/prompt.txt`。可用 `--temperature`、`--llm-timeout`、`--retrieval-url`、`--retrieval-timeout` 调整模型和检索服务参数。

LLM 请求默认超时为 60 秒，分析器和优化器请求共用这个设置。也可以在 `config.json` 的 `llm` 段配置，例如：

```json
{
  "llm": {
    "timeout": 180
  }
}
```

命令行参数 `--llm-timeout 180` 会覆盖配置文件。`retrieval.timeout` 只控制检索请求，不影响 prompt 分析和优化请求。需要注意：该参数只能放宽客户端等待时间；如果 LLM 网关、反向代理或服务端自身有更短的超时，仍然需要同时调整服务端配置。由于每批默认会将 50 条分析汇总后交给优化器，优化请求通常比单条分析更慢；如果仍然超时，可先将 `analysis_batch_size` 调小，例如 20 或 10。

每轮在 `promptEov/runs/iteration_N.json` 保存 prompt、逐条检索结果、top-10 命中标记、bad cases、分批分析报告和新 prompt。badcase 按 `analysis_batch_size` 分批，默认每累计 50 条分析就优化一次 prompt；最后不足 50 条的尾批也会执行。每个 badcase 单独调用一次分析器，批内多个 badcase 的分析会并行执行，再按原顺序汇总后交给优化器；可通过 `analysis_concurrency` 和 `analysis_batch_size` 调整并行度。优化器提示词要求将结果放在 `<result></result>` 中，引擎会提取标签内的 prompt；未带标签时兼容使用完整返回文本。可通过 `generate_query(dialogue_prompt, dialogue)` 与 `retrieve(query)` 注入自定义实现。
