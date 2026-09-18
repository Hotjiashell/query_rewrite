"""Evaluate baseline or a GEPA-optimized DSPy program and preserve retrieval traces."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from common import DialogueExample, load_case_titles, load_examples, load_settings, write_json
from query_program import make_lm, make_program
from retrieval import CaseRetriever, matched_rank


def _load_program(program_path: str | None):
    program = make_program()
    if program_path:
        program.load(program_path)
    return program


def _record(program, retriever: CaseRetriever, sample: DialogueExample) -> dict:
    try:
        prediction = program(dialogue=sample.dialogue)
        query = getattr(prediction, "query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("DSPy returned an empty query")
        cases = retriever.retrieve(query.strip())
        rank = matched_rank(sample.expected_case_id, cases)
        return {
            "sample_index": sample.sample_index,
            "call_sno": sample.call_sno,
            "expected_case_id": sample.expected_case_id,
            "query": query.strip(),
            "matched_rank": rank,
            "retrieval_trace": [asdict(case) for case in cases],
            "status": "success",
            "error": None,
        }
    except Exception as exc:
        return {"sample_index": sample.sample_index, "call_sno": sample.call_sno, "expected_case_id": sample.expected_case_id, "query": None, "matched_rank": None, "retrieval_trace": [], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _metrics(records: list[dict]) -> dict:
    total = len(records)
    result = {"total_samples": total, "successful_samples": sum(row["status"] == "success" for row in records)}
    for cutoff in (1, 3, 5, 10):
        hits = sum(isinstance(row["matched_rank"], int) and row["matched_rank"] <= cutoff for row in records)
        result[f"hits_at_{cutoff}"] = hits
        result[f"recall_at_{cutoff}"] = hits / total if total else 0.0
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate query rewriting with the existing retrieval service.")
    parser.add_argument("--config", default="gepa/config.json")
    parser.add_argument("--program", help="Path to optimized_program.json; omit for the BASELINE_PROMPT seed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    import dspy

    dspy.configure(lm=make_lm(settings.task_lm))
    program = _load_program(args.program)
    retriever = CaseRetriever(settings.retrieval.url, settings.retrieval.timeout, settings.retrieval.top_k)
    samples = load_examples(settings.input_path, load_case_titles(settings.case_path))
    records: list[dict | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=settings.num_threads) as pool:
        futures = {pool.submit(_record, program, retriever, sample): index for index, sample in enumerate(samples)}
        completed = 0
        total = len(samples)
        for future in as_completed(futures):
            records[futures[future]] = future.result()
            completed += 1
            print(f"\r评估进度: {completed}/{total}", end="", flush=True)
    if samples:
        print()
    completed_records = [record for record in records if record is not None]
    payload = {"schema_version": 1, "artifact_type": "gepa_retrieval_evaluation", "configuration": {"program": args.program or "baseline", "retrieval_top_k": settings.retrieval.top_k}, "metrics": _metrics(completed_records), "records": completed_records}
    write_json(payload, args.output)
    print(payload["metrics"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
