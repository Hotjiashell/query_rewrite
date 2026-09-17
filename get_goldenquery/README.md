# Golden Query Generation

For every dialogue, this command gives the LLM the dialogue and its labelled
case's title only. It retrieves the generated query immediately. When
the labelled case is absent from Top-5 by default, the exact returned Top-5 cases are
fed back to the LLM for a revised query, up to `max_retries` revisions.

```bash
python -m get_goldenquery --config get_goldenquery/config.example.json
```

CLI arguments override the configuration file:

```bash
python -m get_goldenquery \
  --dialogues data/dialog_example.json \
  --cases data/case_example.json \
  --output results/golden_queries.json \
  --max-retries 5 --concurrency 4
```

The dialogue file follows `evaluate.py` (`chat_content`, `caseID`, and optional
`call_sno`). The case file may be a mapping such as
`{"KT000001": {"case_name": "...", "text": "..."}}`, or a list with
`caseID`/`case_id` and a title field. Case text/content is ignored. The result
file contains the final query and all attempts, including the Top-K titles
supplied to the retry prompt. A sample succeeds only when its labelled case ID
is in Top-K.
The terminal displays real-time completion, Top-K hits, and failures. The final
`summary` includes `hits_at_1/3/5/10` and `recall_at_1/3/5/10`, calculated
from the last query for every input sample.

## Analyse Misses

Compare a successful golden-query artifact with an `evaluate.py generate`
query artifact. The ordinary query is retrieved again, and only samples where
the golden query succeeded but the ordinary query misses the GT case in Top-10
are sent to the LLM for analysis.

```bash
python -m get_goldenquery.analyse \
  --golden results/golden_queries.json \
  --queries results/generated_queries.json \
  --output results/golden_query_miss_analysis.json
```

It also accepts `--config get_goldenquery/config.example.json`; its
`golden_query_analysis.top_k` defaults to 10.

For each analysed miss, the output has `analysis.reason`,
`analysis.missing_keywords`, and `analysis.noise_keywords`. Missing keywords
must occur in both the golden query and GT title but not the ordinary query.
Noise words must be absent from the golden query, present in the ordinary
query, and repeatedly appear in the ordinary query's Top-10 titles. Only case
titles are sent to the model; case content is never sent.
