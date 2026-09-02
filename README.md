# Query Rewrite Baseline Evaluation

This repository evaluates a query rewriting strategy against the case
retrieval service. The first strategy is a one-shot baseline using
`prompt.BASELINE_PROMPT`.

## Installation

```bash
pip install -r requirements.txt
```

## Configure the two stages

Edit [config.json](config.json) before running. Query generation and retrieval
are independent stages, with independent input/output paths and concurrency:

```json
{
  "llm": {
    "base_url": "https://your-llm.example/v1",
    "model_name": "your-model-name",
    "api_key_env": "OPENAI_API_KEY"
  },
  "query_generation": {
    "input_path": "data/dialog_example.json",
    "output_path": "results/generated_queries.json",
    "concurrency": 4
  },
  "retrieval": {
    "input_path": "results/generated_queries.json",
    "output_path": "results/baseline.json",
    "url": "http://10.67.43.14:8276/run_case_retrieval",
    "timeout": 30,
    "concurrency": 8
  }
}
```

By default the key is read from the environment variable named by
`llm.api_key_env`:

```bash
export OPENAI_API_KEY="your-api-key"
python generate_queries.py
python retrieve_cases.py
```

The first command only calls the LLM and writes
`query_generation.output_path`. The second command reads only
`retrieval.input_path`, calls the retrieval service, and writes the final
evaluation. It never calls the LLM, so you can rerun retrieval with a new URL,
timeout, or concurrency without regenerating queries.

For a short-lived local setup, `llm.api_key` is also supported, but keeping a
secret in the configuration file is not recommended. Regardless of its source,
the API key is never written to the result artifact.

The same stages are also exposed as subcommands:

```bash
python evaluate.py generate
python evaluate.py retrieve
```

If you want one command for the complete pipeline, use `all`. It still writes
the intermediate query file, then reads that file for retrieval, and uses the
two configured concurrency values independently:

```bash
python evaluate.py all
```

In `all` mode, `query_generation.output_path` is always used as the retrieval
input for that run, ensuring retrieval consumes the queries just generated.
The final result is written to `retrieval.output_path`.

命令行运行时会在终端实时显示当前阶段的完成数、百分比、成功数和失败数；
进度信息输出到 stderr，最终汇总输出到 stdout。作为 Python 库调用时，
`progress` 默认关闭，可按需传入 `progress=True`。

Use another settings file with `--config`, or override a specific setting for
one stage:

```bash
python generate_queries.py --config configs/experiment-a.json --concurrency 4
python retrieve_cases.py --concurrency 8 --output results/experiment-a.json
```

For the LLM settings, resolution priority is command-line option, then
config-file value, then environment variable. The baseline includes
`extra_body={"chat_template_kwargs": {"enable_thinking": false}}` on every
model request.

`query_generation.concurrency` limits concurrent LLM calls, while
`retrieval.concurrency` limits concurrent retrieval calls.
`retrieval.timeout` applies to each retrieval request.

No external request is made by installing dependencies or running tests. An
evaluation run does call both the configured LLM and the retrieval endpoint.

## Input contract

The input must be a JSON array with these required fields per item:

```json
{
  "call_sno": "00000001",
  "chat_content": "客服：您好\\n用户：电脑连不上网",
  "caseID": "KT0000001"
}
```

`call_sno` is optional metadata. `caseID` is the ground-truth ID used to
calculate Recall@1, Recall@3, Recall@5, and Recall@10.

## Query artifact

The first stage writes a `generated_queries` JSON artifact with a record for
every input item. Each record retains `sample_index`, `call_sno`,
`expected_case_id`, `query`, `status`, and `error`. It does not write the full
dialogue or API key.

An input or model error marks only that record as `failed`; all other samples
continue. The second stage reads this artifact. Failed query records are kept
in the final output with `retrieval_status: "skipped"` and count as a miss;
they do not cause a new model call.

## Retrieval artifact

The second-stage output is a JSON object containing a sanitized configuration,
aggregate metrics, and one record per query artifact entry. API keys and full
case content are never written. Each record stores:

- the generated `query`;
- ordered `retrieval_trace` entries with only `rank`, `case_id`, and
  `case_title`;
- `query_status`, `retrieval_status`, `matched_rank`, and any per-sample
  query/retrieval error.

The retriever collects every numbered key (`top1`, `top2`, and so on) in
numeric order. It does not assume a fixed result count, so a response with 5,
7, 10, or another number of returned candidates evaluates correctly. Failed
model or retrieval requests are retained and count as a miss in recall, while
`successful_samples` lets you inspect service health separately.

## Add an improved strategy

`QueryGenerator` in `gen_query.py` is the sole extension point:

```python
from gen_query import QueryGenerator


class ImprovedQueryGenerator(QueryGenerator):
    def generate(self, dialogue: str) -> str:
        # Call your rewritten-query pipeline here.
        return "query for case retrieval"
```

The generation command uses the implementation automatically. The dataset
loading, query-file persistence, retrieval tracing, and Recall@K calculation
can all remain unchanged.

## Tests

```bash
python -m unittest discover -s tests -v
```
