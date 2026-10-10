# Huawei 单 query baseline 评估

固定调用 `api.get_query(dialogue)`，从返回的 JSON 对象中读取
`baseline_user_query` 字符串。不使用本项目的 LLM、提示词或多 query 策略。
检索、输入格式、结果格式及 Recall@1/3/5/10 的口径复用根目录 `evaluate.py`。

**目前 `api.py` 是占位函数，需要先补齐实际接口调用。** 约定如下：

```python
def get_query(dialog):
    # 在此请求实际服务，并返回解析后的 JSON 对象。
    return {"baseline_user_query": "接口生成的单条 query"}
```

实际网络请求的地址、鉴权和超时在 `api.py` 中设置。

在项目根目录运行完整评估：

```bash
python huawei_baseline/evaluate_baseline.py all \
  --config config.json \
  --input data/dialogs.json \
  --query-output huawei_baseline/results/queries.json \
  --output huawei_baseline/results/evaluation.json
```

也可以分两步执行，第二步直接复用生成结果，不会再次调用 query 接口：

```bash
python huawei_baseline/evaluate_baseline.py generate \
  --config config.json \
  --input data/dialogs.json \
  --output huawei_baseline/results/queries.json

python huawei_baseline/evaluate_baseline.py retrieve \
  --config config.json \
  --input huawei_baseline/results/queries.json \
  --output huawei_baseline/results/evaluation.json
```

支持等价的模块调用 `python -m huawei_baseline.evaluate_baseline all ...`。
默认阶段为 `all`。默认样本并发数为 1，可用 `--concurrency` 同时设置两阶段，
或用 `--query-concurrency` / `--retrieval-concurrency` 单独覆盖。
检索地址和超时从 `config.json` 的 `retrieval` 部分读取，也可以用
`--retrieval-url` / `--timeout` 覆盖。单 query 直接使用检索服务的原始排序，
不进行融合或相关性过滤。

可选配置（CLI 优先；输入默认回退到 `query_generation.input_path`）：

```json
{
  "huawei_baseline": {
    "input_path": "data/dialogs.json",
    "query_output_path": "huawei_baseline/results/queries.json",
    "output_path": "huawei_baseline/results/evaluation.json",
    "concurrency": 1
  }
}
```

输入为 JSON 数组，每条包含 `chat_content`、`caseID`，以及可选 `call_sno`。
生成结果保存 query、原始对话、目标案例 ID 和逐条错误；最终结果保存检索
案例 ID、标题、分数、命中排名及汇总召回率。生成或检索失败不会中断其余
样本，失败样本保留在召回率分母中，计为未命中。命令返回 0 表示结果文件
已写入，逐条成功或失败情况请查看 `summary` / `metrics` 和 `records`。

运行测试：`python -m unittest huawei_baseline.test_evaluate_baseline -v`。
