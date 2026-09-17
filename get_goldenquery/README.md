# Golden Query Generation

For every dialogue, this command gives the LLM the dialogue and its labelled
case's title and content. It retrieves the generated query immediately. When
the labelled case is absent from Top-10, the exact returned Top-10 cases are
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
`caseID`/`case_id`, title, and content fields. The result file contains the
final query and all attempts, including the full Top-K content supplied to the
retry prompt. A sample succeeds only when its labelled case ID is in Top-K.
