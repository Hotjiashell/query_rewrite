# Query Rewrite Evaluation

This repository evaluates a query rewriting strategy against the case
retrieval service. It currently includes five prompt methods:

- `baseline`: one-shot generation using `prompt.BASELINE_PROMPT`;
- `method_v1`: the improved prompt in `prompt.METHOD_V1_PROMPT`;
- `multi_query`: generates up to three queries per dialogue using
  `prompt.MULTI_QUERY_PROMPT`, retrieves each in parallel, and fuses the
  results (see [Multi-query retrieval fusion](#multi-query-retrieval-fusion));
- `custom`: a one-shot method backed by your own prompt template file (see
  [Using a custom prompt file](#using-a-custom-prompt-file));
- `custom_multi`: the multi-query counterpart of `custom` — your own prompt
  template file, but expected to return up to three queries and fused the
  same way as `multi_query` (see
  [Using a custom prompt file](#using-a-custom-prompt-file)).

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
    "method": "baseline",
    "input_path": "data/dialog_example.json",
    "output_path": "results/generated_queries.json",
    "concurrency": 4
  },
  "retrieval": {
    "input_path": "results/generated_queries.json",
    "output_path": "results/baseline.json",
    "url": "http://10.67.43.14:8276/run_case_retrieval",
    "timeout": 30,
    "concurrency": 8,
    "fusion_method": "round_robin",
    "top_k": 10
  }
}
```

`retrieval.fusion_method` and `retrieval.top_k` only take effect for samples
whose query artifact record has more than one `queries` entry (i.e. generated
with `multi_query`); single-query records always retrieve and evaluate
exactly as before.

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
python retrieve_cases.py --fusion-method score --top-k 5
```

For the LLM settings, resolution priority is command-line option, then
config-file value, then environment variable. All three prompt methods include
`extra_body={"chat_template_kwargs": {"enable_thinking": false}}` on every
model request.

## LLM reranking

After retrieval, `rerank_cases.py` can ask the model to judge the returned
cases against the full dialogue and reorder them. The current retrieval
artifact provides `case_id`, `case_title`, and original rank as the default
evidence sent to the model. The retrieval similarity score is deliberately
ignored during reranking and omitted from the reranking output. If an input
artifact has an optional `content` field in a trace entry, it is also included
in the prompt.

Add or adjust the `rerank` section in `config.json`:

```json
{
  "rerank": {
    "input_path": "results/baseline.json",
    "output_path": "results/baseline_reranked.json",
    "concurrency": 8,
    "candidate_limit": 0
  }
}
```

Then run:

```bash
python rerank_cases.py --config config.json
```

The input must be the `retrieval_evaluation` artifact produced by
`evaluate.py`. `candidate_limit` is optional; `0` reranks every returned
candidate, while a positive value limits the model prompt to the first N
candidates. Command-line values override the config file, so a one-off run
can be started with:

```bash
python rerank_cases.py \
  --input results/baseline.json \
  --output results/baseline_reranked.json \
  --concurrency 8
```

The model must return one JSON object with a complete `ranking` array. Every
candidate `case_id` must occur exactly once, in the desired order, with a
`relevance` score from 1 to 5 and a short `reason`. The script disables model
thinking using the same `extra_body` setting as query generation, displays
progress on stderr, and runs samples concurrently. A model timeout, malformed
JSON response, missing candidate, or extra candidate only marks that sample as
failed; its `reranked_trace` falls back to the original retrieval order. The
output includes both `source_metrics` and reranked `metrics`, plus
`reranked_matched_rank` for each sample.

## Serial latency and recall benchmark

To measure the end-to-end latency of query generation, retrieval, and
reranking, use `latency_benchmark.py`. It processes samples in input order and
processes different samples serially. Multiple queries belonging to the same
`multi_query` sample are still retrieved in parallel:

```bash
python latency_benchmark.py \
  --config config.json \
  --test-num 100 \
  --output results/latency_benchmark.json
```

The benchmark uses `query_generation.input_path`, the configured query method,
retrieval settings, and the shared LLM settings from `config.json`. A
`benchmark` section can override the input, output, sample count, method,
retrieval URL/timeout, fusion settings, and rerank candidate limit:

```json
{
  "benchmark": {
    "input_path": "data/dialog_example.json",
    "output_path": "results/latency_benchmark.json",
    "test_num": 100,
    "candidate_limit": 0
  }
}
```

`--test-num 0` means all input samples. The two reported times start before
query generation for each sample:

- `metrics.latency.time_to_retrieval.average_seconds`: average time until the
  retrieval stage returns;
- `metrics.latency.time_to_rerank.average_seconds`: average cumulative time
  until the reranking stage returns.

The per-sample fields are `time_to_retrieval_sec` and
`time_to_rerank_sec`. The artifact also includes `metrics.retrieval` and
`metrics.rerank`, each with Recall@1, Recall@3, Recall@5, and Recall@10. A
latency average only includes samples that reached that stage; the artifact
records the corresponding completed-sample count.

## Multi-query evaluation with Laya filtering

`evaluate_laya.py` is a separate multi-query pipeline:
custom-prompt generation -> retrieve each query -> deduplicate by `case_id`
-> judge each unique case against the dialogue with Laya -> filter -> fuse.
It reuses the existing retrieval endpoint and generated-query artifact format.
Generation always uses `custom_multi`; there is no built-in prompt selection.
One query in a `queries` array is accepted when the custom generator only
finds one useful search direction; single-query artifacts without that array
are rejected per sample.

Use Python 3.11+ (the existing evaluation code uses `datetime.UTC`), then install:

```bash
python -m pip install -r requirements-laya.txt
```

Use an existing query file without calling the query-generation LLM:

```bash
python evaluate_laya.py retrieve \
  --config config.json \
  --query-file results/custom_multi_queries.json \
  --output results/custom_multi_laya.json \
  --device cuda \
  --threshold 0.5 \
  --fusion-method round_robin \
  --top-k 10
```

The query file is the `generated_queries` JSON artifact produced by
`evaluate.py generate` / `generate_queries.py`: each successful record needs
`queries`, `chat_content`, and the usual identifiers and expected case ID.
`--input` is also accepted for the query artifact in `retrieve` mode. Older
artifacts missing `chat_content` can supply `--dialogues-file data/dialogs.json`;
the script restores by `sample_index` and verifies `call_sno` and the expected
case ID against the source to avoid judging the wrong dialogue.

Generate queries using your own prompt, then immediately retrieve and filter:

```bash
python evaluate_laya.py all \
  --config config.json \
  --input data/dialogs.json \
  --prompt-file prompts/laya_multi_query.txt \
  --query-output results/custom_multi_queries.json \
  --output results/custom_multi_laya.json \
  --device cuda \
  --fusion-method score
```

The included prompt is an editable starting point. Like other `custom_multi`
templates, it must contain `{dialogue}` and ask for a JSON object whose `query`
field is an array, for example `{"query": ["first query", "second query"]}`.
`generate` runs only generation (`--output` specifies the query artifact);
`all --query-file FILE` bypasses generation and uses that file. The existing
`llm`, `query_generation`, and `retrieval` config sections still apply;
generation requires a prompt from `--prompt-file` or
`query_generation.prompt_file`. Retrieval concurrency comes from
`--concurrency` / `retrieval.concurrency`; generation concurrency comes from
`--query-concurrency` / `query_generation.concurrency`.

Optional Laya defaults in `config.json` (CLI values take precedence):

```json
{
  "laya_filter": {
    "model": "convaiinnovations/laya-multilingual",
    "device": "cuda",
    "threshold": 0.5,
    "batch_size": 16,
    "max_len": 1024,
    "head_max_len": 256,
    "fast": false
  }
}
```

The multilingual checkpoint is the default; `--laya-model` can name a local
checkpoint directory or another Hub model. The first load downloads weights.
Omit `--device` to let Laya choose, or use `--device cpu` / `mps` as appropriate.
The model loads once per run. Unique title/dialogue pairs are passed to
`predict_batch` with `--batch-size` (default 16) and length grouping; sample
retrieval can run concurrently, while shared-model inference is serialized.
For supported NVIDIA setups, install `"laya[fast]"` and add `--fast` to enable
the optional TileLang path. First-use compilation can affect timings.

Filtering uses a fixed two-option `choice` question with `related` and
`unrelated` descriptions. Only **case title + original dialogue** are passed
to Laya; neither the expected case ID nor the retrieval score is model input.
Keep a case when `probabilities.related >= threshold` (default 0.5).
This threshold is an experimental setting, not a calibrated accuracy guarantee.
`--max-len` and `--head-max-len` control token budgets. If Laya reports input
truncation, an empty title, or invalid output, the sample fails filtering with
an explicit error instead of silently using an incomplete decision. Raise the
token limit or shorten the source dialogue when needed.

After filtering, `round_robin` takes turns among the surviving per-query
lists, skipping duplicate IDs; `score` retains the highest retrieval score
per ID and sorts descending. Both apply `top_k` **after** filtering, so lower
ranked relevant cases can fill the places of removed cases. Laya probabilities
are used for filtering, not as replacement ranking scores. Empty surviving
lists are valid successful results with no recall hit. Partial query retrieval
failures still use the successful lists and report `partial_success`.

The output retains `artifact_type: retrieval_evaluation` and the existing
`retrieval_trace`, `matched_rank`, and `metrics` fields for downstream tools.
It additionally records:

- `prefilter_metrics`: Recall@K before filtering with the same fusion and top K.
- `per_query_traces` / `prefilter_trace`: original deduplicated lists and baseline fusion.
- `laya_judgments`: every unique case's title, probability, choice, and keep/drop decision.
- Candidate counts, `filter_status`, and `filter_error` for inspecting removals and failures.
- `model_load_seconds` and per-sample retrieval/filter/fusion/total timings.
- `union_top_k`: hits and recall at K=1/3/5/10 (plus the configured `top_k`),
  counted when **any query's original rank** for the expected case is at most K,
  before filtering or fusion. Each sample counts at most once. Each record also
  has `union_matched_rank` and per-K boolean `union_top_k` values.
- `threshold_search`: the best probability threshold and fusion strategy for
  Recall@10, the best setting for each strategy, and every tested setting.

New retrieval runs automatically compute these analyses. To analyse an already
saved Laya result file offline, with no model loading or retrieval calls:

```bash
python evaluate_laya.py analyze \
  --input results/custom_multi_laya.json \
  --output results/custom_multi_laya_analyzed.json
```

This mode only needs the result's `per_query_traces` and `laya_judgments` plus
evaluation identifiers/statuses; it does not read `config.json`. It preserves
the original configured `metrics` and final trace and adds analysis fields.
Query-generation artifacts alone do not contain the retrieval results or
Laya probabilities needed for this analysis.

The search tests `round_robin` and `score`, always fusing to **10 cases** even
if the original run used another `top_k`. It tests 0, 1, and every distinct
saved `related_probability`. With the `>=` keep rule, these cover every
attainable candidate set, so the search is exact rather than a coarse grid.
It reuses probabilities and does not rerun Laya. Contributions are accumulated
by per-sample probability intervals, avoiding replaying the entire dataset
for every global threshold. `threshold_search.best` records `threshold`,
`fusion_method`, `hits_at_10`, and `recall_at_10`; `best_by_fusion` records each
strategy's optimum. `results` records all tested combinations. For ties the
reported best uses the lowest threshold, then `round_robin` before `score`.
All input samples remain in the recall denominator. Samples with failed or
incomplete Laya judgments count as misses and are listed in `skipped_samples`;
if none are evaluable, `best` is null. This optimum is measured on the saved
dataset used to select it; validate the chosen setting on separate data.

`union_top_k.all_candidates_recall` also reports union recall over **all saved
candidates**, which is the upper bound for the full saved candidate pool.
`union_top_k.recall_at_10` is the upper bound when restricting each query to
its original Top 10. A case originally ranked 11 or lower can enter the final
Top 10 after filtering, so final Recall@10 can exceed the original-Top-10
union value when the saved lists contain more than 10 cases each.

Filtering times include batch preparation and, with concurrent samples, waiting
for the model lock. They are pipeline timings, not isolated GPU kernel timings.
Both Recall@K summaries use all input samples as the denominator; a filtering
failure counts as a failure in final metrics while its successful prefilter
retrieval remains represented in the baseline.

## Multi-query retrieval fusion

`--method multi_query` asks the model to propose up to three queries per
dialogue, each focused on a different angle (e.g. "电脑坏了怎么修理" and
"联系资产管理员" for the same conversation). During retrieval, every query for
a sample is sent to the retrieval service concurrently, and the resulting
traces are fused into one `retrieval_trace` before Recall@K is calculated.
Two fusion strategies are supported via `retrieval.fusion_method`
(or `--fusion-method`):

- `round_robin` (default): interleaves the per-query traces in their original
  rank order, taking the next not-yet-seen case from each query's list in
  turn, until `top_k` results are collected or every list is exhausted. This
  favors spreading results evenly across the different query angles.
- `score`: deduplicates by `case_id`, keeping each case's highest `score`
  across all queries, then sorts the merged list by score descending.

`retrieval.top_k` (default `10`, override with `--top-k`) caps how many fused
results are kept; ranks in the fused trace are renumbered starting at 1.

If one of a sample's queries fails to retrieve (e.g. a timeout) while at
least one other succeeds, the sample is still evaluated using the successful
queries' fused results, and `retrieval_status` is set to `"partial_success"`
(the failed query's error is kept in `retrieval_error`). Only when every
query for a sample fails is the sample marked `retrieval_status: "failed"`.

## Using a custom prompt file

To try a prompt without editing `prompt.py`, set `query_generation.method` to
`custom` (single query) or `custom_multi` (up to three queries, fused like
`multi_query`) and point `query_generation.prompt_file` (or `--prompt-file`)
at a text file:

```json
{
  "query_generation": {
    "method": "custom",
    "prompt_file": "prompts/my_experiment.txt",
    "input_path": "data/dialog_example.json",
    "output_path": "results/generated_queries.json",
    "concurrency": 4
  }
}
```

```bash
python generate_queries.py --method custom --prompt-file prompts/my_experiment.txt
python generate_queries.py --method custom_multi --prompt-file prompts/my_multi_experiment.txt
```

The file must contain the literal `{dialogue}` placeholder, which is replaced
with the dialogue text before the request is sent (the same substitution used
by `BASELINE_PROMPT`, `METHOD_V1_PROMPT`, and `MULTI_QUERY_PROMPT`).

For `custom`, the file must produce the single-query JSON format that
`baseline` and `method_v1` use, wrapped in a ```` ```json ```` fence or
returned directly:

```json
{"query": "your retrieval query"}
```

For `custom_multi`, the file must produce the `MULTI_QUERY_PROMPT` array
format instead (a bare string is also accepted and treated as one query):

```json
{"query": ["query angle 1", "query angle 2"]}
```

`custom` always goes through the single-query path, so retrieval and fusion
behave exactly as they do for `baseline`/`method_v1`. `custom_multi` behaves
exactly as `multi_query` does: its generated queries are retrieved
concurrently and fused per `retrieval.fusion_method`/`retrieval.top_k` (see
[Multi-query retrieval fusion](#multi-query-retrieval-fusion)).
`--prompt-file` takes precedence over `query_generation.prompt_file` when
both are set; missing it for `custom` or `custom_multi` is an error before
any model call is made. The resolved path is recorded (not its contents)
under `configuration.prompt_file` in the query artifact, so you can tell
which prompt file produced a given run.

## Compare baseline and METHOD_V1

Select the prompt method in `query_generation.method`, or override it for one
run with `--method`:

```bash
python evaluate.py all --method method_v1
```

For a clean comparison, write the two runs to different artifacts so that the
generated queries and retrieval traces remain available:

```bash
python evaluate.py all \
  --method baseline \
  --query-output results/baseline_queries.json \
  --retrieval-output results/baseline.json

python evaluate.py all \
  --method method_v1 \
  --query-output results/method_v1_queries.json \
  --retrieval-output results/method_v1.json
```

The query artifact records the canonical method under
`configuration.method` (and the backwards-compatible `configuration.generator`)
so each output can be identified later. Compare the final artifacts' values
under `metrics.recall_at_1`, `metrics.recall_at_3`, `metrics.recall_at_5`, and
`metrics.recall_at_10`.

The convenience entry point accepts the same option:

```bash
python generate_queries.py --method method_v1
```

## Compare two existing result files

To inspect cases that the first method recalls but the second method misses,
use `compare_results.py`. It aligns records by `sample_index` and compares the
ground-truth `matched_rank` at the requested cutoff (Recall@10 by default):

```bash
python compare_results.py \
  results/baseline.json \
  results/method_v1.json \
  --cutoff 10 \
  --output results/baseline_only.json
```

The command prints a summary and one line per difference. The optional JSON
report's `records` array contains only samples where the first file has a hit
within the cutoff and the second file does not. Each record includes both
methods' query, status, matched rank, errors, and retrieval trace; trace entries
are limited to `rank`, `case_id`, and `case_title`. Use `--cutoff 1`, `--cutoff 3`,
or `--cutoff 5` to compare the corresponding recall window.

`query_generation.concurrency` limits concurrent LLM calls, while
`retrieval.concurrency` limits concurrent retrieval calls.
`retrieval.timeout` applies to each retrieval request. For `multi_query`
records, a sample's own queries are always retrieved concurrently with each
other regardless of `retrieval.concurrency`, which only limits how many
samples are processed at once.

## Collect badcases

To review every sample that missed within top-K, use `collect_badcases.py`.
It reads a `retrieval_evaluation` artifact plus a case summary file (a JSON
object mapping `case_id` to `{"case_name": ..., "text": ...}`, as in
[data/case_example.json](data/case_example.json)) and fills in each miss's
ground-truth case title from the summary file — useful because the
evaluation artifact's own `gt_case_title` is only populated when the
ground-truth case happens to appear in the retrieval trace:

```bash
python collect_badcases.py \
  results/method_v1.json \
  data/case_example.json \
  --cutoff 10 \
  --output results/badcases.json
```

A badcase is a sample whose query generation and retrieval both succeeded but
whose `matched_rank` is `None` or falls outside `--cutoff` (default `10`).
Samples where `status` is `"failed"` (an LLM or retrieval error) are excluded,
since those are infrastructure failures rather than retrieval misses. Each
report record keeps `sample_index`, `call_sno`, `chat_content`,
`expected_case_id`, `query`/`queries`, `matched_rank`, and `retrieval_trace`,
plus the resolved `gt_case_title` and a `gt_case_found` flag (`false` when the
`expected_case_id` has no entry in the case summary file, so you can spot
missing case metadata instead of a silent `null`).

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

For `multi_query`, records also carry a `queries` array (the full list of up
to three generated queries; `query` is always `queries[0]`, kept for backward
compatibility). `baseline` and `method_v1` records leave `queries` as `null`.

An input or model error marks only that record as `failed`; all other samples
continue. The second stage reads this artifact. Failed query records are kept
in the final output with `retrieval_status: "skipped"` and count as a miss;
they do not cause a new model call.

## Retrieval artifact

The second-stage output is a JSON object containing a sanitized configuration,
aggregate metrics, and one record per query artifact entry. API keys and full
case content are never written. Each record stores:

- the generated `query`;
- ordered `retrieval_trace` entries with `rank`, `case_id`, `case_title`, and
  `score` (`null` when the retrieval response omitted it);
- `query_status`, `retrieval_status`, `matched_rank`, and any per-sample
  query/retrieval error.

`retrieval_status` is `"success"` for a normal single-query retrieval,
`"partial_success"` when a `multi_query` record fused results from queries
that partially failed, or `"failed"` when every query for a sample failed
(see [Multi-query retrieval fusion](#multi-query-retrieval-fusion)).

The retriever collects every numbered key (`top1`, `top2`, and so on) in
numeric order. It does not assume a fixed result count, so a response with 5,
7, 10, or another number of returned candidates evaluates correctly. Failed
model or retrieval requests are retained and count as a miss in recall, while
`successful_samples` lets you inspect service health separately.

## Add an improved strategy

`QueryGenerator` in `gen_query.py` is the extension point for single-query
strategies:

```python
from gen_query import QueryGenerator


class ImprovedQueryGenerator(QueryGenerator):
    def generate(self, dialogue: str) -> str:
        # Call your rewritten-query pipeline here.
        return "query for case retrieval"
```

For strategies that propose several queries per dialogue (like `multi_query`),
implement `MultiQueryGenerator.generate_queries` instead; `generate()` is
provided for you and returns the first query:

```python
from gen_query import MultiQueryGenerator


class ImprovedMultiQueryGenerator(MultiQueryGenerator):
    def generate_queries(self, dialogue: str) -> list[str]:
        # Call your rewritten-query pipeline here.
        return ["query angle 1", "query angle 2"]
```

The generation command uses the implementation automatically: `evaluate.py`
detects a `MultiQueryGenerator` instance and persists its full `queries` list,
and the retrieval stage fuses per-query results whenever a record has more
than one query. The dataset loading, query-file persistence, retrieval
tracing, and Recall@K calculation can all remain unchanged.

## Tests

```bash
python -m unittest discover -s tests -v
```
