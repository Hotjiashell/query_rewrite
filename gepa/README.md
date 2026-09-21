# DSPy GEPA query-rewrite optimization

This directory is independent of the existing `evaluate.py` pipeline. It uses
the same input contract: each dialogue needs `chat_content` and the target
`caseID`; `dataset.case_path` must map every such ID to its GT case title. GEPA
evolves the DSPy predictor instruction while the metric calls the existing
`search.py` endpoint and scores whether the target case appears within the
configured cutoff.

## Why this mapping is appropriate

`query` text is only an intermediate representation. The business signal is
whether retrieval returns the known target case, so GEPA optimizes
`Recall@K`, not similarity to a hand-written query. The metric is deterministic
apart from the task LLM and retrieval service. Every final evaluation artifact
keeps the generated query, every returned `top<number>` item in numerical order,
and only each case's `id` and `title`.

When `dataset.golden_query_path` is configured, successful records from
`get_goldenquery` add an offline-only reference to failed GEPA trajectories:
the target case title and the golden query. This follows the current golden
retry/analysis prompts: GEPA can distinguish missing target concepts from
noise terms that dominate returned titles. These fields are metadata for
reflection only; `dialogue` remains the predictor's sole runtime input.

The GT case title does not depend on golden queries. It is always included in
the feedback built from `dataset.case_path`, so every retrieval miss can be
diagnosed as missing target concepts and/or query noise. Training fails early
when any dialogue `caseID` lacks a title, rather than silently producing weak
feedback for part of the dataset.

The default training objective is `Recall@5`; evaluation always reports
`Recall@1`, `Recall@3`, `Recall@5`, and `Recall@10`. Keep a held-out validation
set: GEPA may see training traces to improve the instruction, but it must not
evolve against the final test set.

## Setup

```bash
cp gepa/config.example.json gepa/config.json
pip install -r gepa/requirements.txt
```

Set the API key environment variable named in `task_lm.api_key_env` and
`reflection_lm.api_key_env`. Both DSPy LMs are created with
`extra_body={"chat_template_kwargs": {"enable_thinking": false}}`, matching
the existing OpenAI-compatible request. The task and reflection models can be
different; use a stronger reflection model where budget permits.
The example uses task temperature `0.0` and reflection temperature `1.0`:
the query generator stays stable while GEPA can propose diverse revisions.

## DSPy development evaluation

```bash
python gepa/evaluate.py --config gepa/config.json --output gepa/runs/baseline.json
python gepa/optimize.py --config gepa/config.json --run-name recall5-v1
python gepa/evaluate.py --config gepa/config.json \
  --program gepa/runs/recall5-v1/optimized_program.json \
  --output gepa/runs/recall5-v1/evaluation.json
```

如果希望脱离 `optimized_program.json`，直接测试包含 DSPy 字段协议的最终提示词：

```bash
python gepa/export_dspy_prompt.py \
  --instruction gepa/runs/recall5-v1/optimized_instruction.txt \
  --output gepa/runs/recall5-v1/dspy_prompt.txt

python gepa/evaluate_prompt.py \
  --config gepa/test_config.json \
  --prompt gepa/runs/recall5-v1/dspy_prompt.txt \
  --output gepa/runs/recall5-v1/test_evaluation.json
```

该评估脚本会在每个样本完成后更新输出文件。若因 API 超时、限流或网络问题中断，使用相同参数并加上 `--resume` 续跑；已有成功样本会跳过，失败样本会重新请求：

```bash
python gepa/evaluate_prompt.py \
  --config gepa/test_config.json \
  --prompt gepa/runs/recall5-v1/dspy_prompt.txt \
  --output gepa/runs/recall5-v1/test_evaluation.json \
  --resume
```

导出的提示词包含 `[[ ## dialogue ## ]]`、`[[ ## query ## ]]` 和
`[[ ## completed ## ]]`。评估脚本要求模型按该协议返回，并直接使用
`--prompt` 指定的文件，不加载 DSPy program artifact。

`max_metric_calls` is a retrieval-evaluation budget, not a number of prompt
variants. Start around 100 only after the dataset contains at least 50-100
representative, correctly labelled dialogues. Use a small budget (for example
10) for connectivity checks. Each run exports the instruction and DSPy program
for DSPy-side inspection.

`optimization.reflection_minibatch_size` controls how many training trajectories
the reflection model sees in one prompt-update step. It defaults to `3`.

To keep the current `evaluate.py` production pipeline, export the selected
instruction to its existing `custom` prompt contract and then run its normal
two-stage evaluator:

```bash
python gepa/export_prompt.py \
  --instruction gepa/runs/recall5-v1/optimized_instruction.txt \
  --output gepa/runs/recall5-v1/custom_prompt.txt
python evaluate.py all --config config.json --method custom \
  --prompt-file gepa/runs/recall5-v1/custom_prompt.txt
```

The exporter restores the JSON code-block and `{dialogue}` placeholder required
by `gen_query.py`; those are output-transport details, not text that GEPA
should spend its search budget changing.

Use the existing `evaluate.py all` command for the final baseline-versus-GEPA
comparison: it exercises the exact production prompt renderer and JSON parser.
The DSPy evaluation command is useful during training, but DSPy's field adapter
does not serialize output in precisely the same way as `gen_query.py`.

## Data you need to provide

1. A larger labelled JSON dataset in the same shape as `data/dialog_example.json`.
   `caseID` must be the exact ID returned by the retrieval service. A practical
   minimum is 100 examples, with a fixed untouched test set; 300+ is preferable.
2. Working task-model and reflection-model endpoints, model names, and API keys.
   The reflection model needs to follow prompt-editing instructions reliably.
3. A stable retrieval endpoint. Its indexed case corpus and ranking configuration
   must remain frozen during a comparison; otherwise a score change cannot be
   attributed to the prompt.
4. A product decision on the production objective: use `metric_cutoff: 5` for
   coverage, or set it to `1` when first-result precision is the true requirement.

`dataset.golden_query_path` is optional. Generate it first with
`python -m get_goldenquery`; point it at the resulting
`golden_query_generation` artifact only when it was produced from the same
dialogue dataset and its `sample_index` values align. Add it under `dataset`
only after that artifact exists:

```json
"golden_query_path": "../results/golden_queries.json"
```

The case corpus/title file is optional for this first version. Supplying a
sanitized `caseID -> title` snapshot later would improve failure explanations,
but case content should not be sent to the reflection model unless approved.

## Output contract

`evaluation.json` contains one record per input sample. `retrieval_trace` is
the ordered list produced from all actual `top<number>` keys, even when the
service returns only `top1`, `top2`, `top5`, `top7`, or any other sparse range.
Failures remain in the denominator, matching the existing evaluator's Recall@K
semantics.
