"""Evaluate a standalone DSPy-protocol prompt against the retrieval service."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from common import load_case_titles, load_examples, load_settings, write_json
from dspy_prompt import parse_query, render_prompt
from query_program import make_lm
from retrieval import CaseRetriever, matched_rank


def _call_lm(lm, prompt: str) -> str:
    outputs = lm(prompt)
    if not outputs:
        raise ValueError("model returned no completion")
    first = outputs[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict) and isinstance(first.get("text"), str):
        return first["text"]
    raise ValueError("model returned an unsupported completion shape")


def _record(lm, prompt_template: str, retriever: CaseRetriever, sample) -> dict:
    try:
        prompt = prompt_template.replace("{dialogue}", sample.dialogue)
        query = parse_query(_call_lm(lm, prompt))
        cases = retriever.retrieve(query)
        rank = matched_rank(sample.expected_case_id, cases)
        return {
            "sample_index": sample.sample_index,
            "call_sno": sample.call_sno,
            "expected_case_id": sample.expected_case_id,
            "query": query,
            "matched_rank": rank,
            "retrieval_trace": [asdict(case) for case in cases],
            "status": "success",
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_index": sample.sample_index,
            "call_sno": sample.call_sno,
            "expected_case_id": sample.expected_case_id,
            "query": None,
            "matched_rank": None,
            "retrieval_trace": [],
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _metrics(records: list[dict]) -> dict:
    total = len(records)
    result = {"total_samples": total, "successful_samples": sum(row["status"] == "success" for row in records)}
    for cutoff in (1, 3, 5, 10):
        hits = sum(isinstance(row["matched_rank"], int) and row["matched_rank"] <= cutoff for row in records)
        result[f"hits_at_{cutoff}"] = hits
        result[f"recall_at_{cutoff}"] = hits / total if total else 0.0
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a standalone DSPy-protocol prompt.")
    parser.add_argument("--config", default="gepa/config.json")
    parser.add_argument("--prompt", required=True, help="Prompt exported by export_dspy_prompt.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    prompt_path = Path(args.prompt)
    try:
        prompt_template = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"prompt file does not exist: {prompt_path}") from exc
    if "{dialogue}" not in prompt_template:
        raise ValueError("prompt file must contain the {dialogue} placeholder")

    import dspy

    lm = make_lm(settings.task_lm)
    dspy.configure(lm=lm)
    retriever = CaseRetriever(settings.retrieval.url, settings.retrieval.timeout, settings.retrieval.top_k)
    samples = load_examples(settings.input_path, load_case_titles(settings.case_path))
    records: list[dict | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=settings.num_threads) as pool:
        futures = {pool.submit(_record, lm, prompt_template, retriever, sample): index for index, sample in enumerate(samples)}
        for completed, future in enumerate(as_completed(futures), start=1):
            records[futures[future]] = future.result()
            print(f"\r评估进度: {completed}/{len(samples)}", end="", flush=True)
    if samples:
        print()
    completed_records = [record for record in records if record is not None]
    payload = {
        "schema_version": 1,
        "artifact_type": "gepa_dspy_prompt_evaluation",
        "configuration": {"prompt": str(prompt_path), "retrieval_top_k": settings.retrieval.top_k},
        "metrics": _metrics(completed_records),
        "records": completed_records,
    }
    write_json(payload, args.output)
    print(payload["metrics"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
