# Query Rewrite Baseline Evaluation

This repository evaluates a query rewriting strategy against the case
retrieval service. The first strategy is a one-shot baseline using
`prompt.BASELINE_PROMPT`.

## Installation

```bash
pip install -r requirements.txt
```

## Run the baseline

Supply an OpenAI-compatible URL, model name, and API key either as arguments
or environment variables. The baseline includes
`extra_body={"chat_template_kwargs": {"enable_thinking": false}}` on every
model request.

```bash
export OPENAI_BASE_URL="https://your-llm.example/v1"
export OPENAI_MODEL="your-model-name"
export OPENAI_API_KEY="your-api-key"

python evaluate.py \
  --input data/dialog_example.json \
  --output results/baseline.json \
  --concurrency 4
```

Equivalent command-line configuration:

```bash
python evaluate.py \
  --input data/dialog_example.json \
  --output results/baseline.json \
  --base-url "https://your-llm.example/v1" \
  --model "your-model-name" \
  --api-key "your-api-key" \
  --concurrency 4
```

The current retrieval service URL comes from `search.py`. Override it with
`--retrieval-url` when needed. `--timeout` defaults to 30 seconds and applies
to each retrieval request.

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

## Result artifact

The output is a JSON object containing a sanitized configuration, aggregate
metrics, and one record per input sample. API keys and full case content are
never written. Each record stores:

- the generated `query`;
- ordered `retrieval_trace` entries with only `rank`, `case_id`, and
  `case_title`;
- `matched_rank`, `status`, and any per-sample error.

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

Pass the instance to `Evaluator(generator=..., retriever=...)` in
`evaluate.py`. The dataset loading, concurrent execution, retrieval tracing,
artifact persistence, and Recall@K calculation can all remain unchanged.

## Tests

```bash
python -m unittest discover -s tests -v
```
