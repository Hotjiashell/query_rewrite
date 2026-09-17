# DSPy GEPA query-rewrite optimization

This directory is independent of the existing `evaluate.py` pipeline. It uses
the same input contract: each dialogue needs `chat_content` and the target
`caseID`. GEPA evolves the DSPy predictor instruction while the metric calls
the existing `search.py` endpoint and scores whether the target case appears
within the configured cutoff.

## Why this mapping is appropriate

`query` text is only an intermediate representation. The business signal is
whether retrieval returns the known target case, so GEPA optimizes
`Recall@K`, not similarity to a hand-written query. The metric is deterministic
apart from the task LLM and retrieval service. Every final evaluation artifact
keeps the generated query, every returned `top<number>` item in numerical order,
and only each case's `id` and `title`.

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

`max_metric_calls` is a retrieval-evaluation budget, not a number of prompt
variants. Start around 100 only after the dataset contains at least 50-100
representative, correctly labelled dialogues. Use a small budget (for example
10) for connectivity checks. Each run exports the instruction and DSPy program
for DSPy-side inspection.

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

The case corpus/title file is optional for this first version. Supplying a
sanitized `caseID -> title` snapshot later would improve failure explanations,
but case content should not be sent to the reflection model unless approved.

## Output contract

`evaluation.json` contains one record per input sample. `retrieval_trace` is
the ordered list produced from all actual `top<number>` keys, even when the
service returns only `top1`, `top2`, `top5`, `top7`, or any other sparse range.
Failures remain in the denominator, matching the existing evaluator's Recall@K
semantics.
