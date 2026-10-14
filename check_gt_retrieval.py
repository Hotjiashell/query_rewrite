"""Check whether test-set GT cases can be retrieved using their own titles."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from collect_badcases import load_case_summary
from evaluate import (
    GeneratedQueryRecord,
    RetrievalEvaluator,
    Retriever,
    SearchRetriever,
    _load_json,
    load_config_file,
    write_json_atomically,
)
from search import DEFAULT_RETRIEVAL_URL


def check_gt_retrieval(
    input_path: str | Path,
    case_summary_path: str | Path,
    retriever: Retriever,
    *,
    concurrency: int = 1,
    cutoff: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Retrieve once per GT ID; retain every associated test sample for review.

    A miss means a successful request did not return the expected ID. Input
    problems and request failures are reported separately, never as misses.
    """
    if cutoff is not None and cutoff < 1:
        raise ValueError("cutoff must be greater than zero")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    samples = _load_json(input_path, "test set")
    if not isinstance(samples, list):
        raise ValueError("test set must be a JSON array")
    cases = load_case_summary(case_summary_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    input_errors: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        raw_id = sample.get("caseID") if isinstance(sample, Mapping) else None
        case_id = str(raw_id).strip() if raw_id is not None else ""
        if not case_id:
            input_errors.append({"sample_index": index, "error": "missing caseID or invalid sample"})
            continue
        grouped.setdefault(case_id, []).append({
            "sample_index": index,
            "call_sno": sample.get("call_sno"),
            "chat_content": sample.get("chat_content"),
        })

    queries: list[GeneratedQueryRecord] = []
    titles: dict[str, str | None] = {}
    for index, case_id in enumerate(grouped):
        entry = cases.get(case_id)
        title = entry.get("case_name") if isinstance(entry, Mapping) else None
        title = title.strip() if isinstance(title, str) and title.strip() else None
        titles[case_id] = title
        queries.append(GeneratedQueryRecord(
            sample_index=index, call_sno=None, expected_case_id=case_id,
            query=title, status="success" if title else "failed",
            error=None if title else "missing GT case or empty case_name in case summary",
        ))

    evaluated = RetrievalEvaluator(retriever).evaluate(
        queries, concurrency=concurrency, progress=progress,
    )
    records: list[dict[str, Any]] = []
    for result in evaluated:
        case_id = result["expected_case_id"]
        rank = result["matched_rank"]
        if result["query_status"] != "success":
            outcome = "missing_title"
        elif result["retrieval_status"] != "success":
            outcome = "retrieval_failed"
        elif rank is not None and (cutoff is None or rank <= cutoff):
            outcome = "hit"
        else:
            outcome = "miss"
        records.append({
            "expected_case_id": case_id,
            "gt_case_title": titles[case_id],
            "query": result["query"],
            "outcome": outcome,
            "matched_rank": rank,
            "retrieval_status": result["retrieval_status"],
            "error": result["error"],
            "retrieval_trace": result["retrieval_trace"],
            "sample_count": len(grouped[case_id]),
            "samples": grouped[case_id],
        })
    counts = {outcome: sum(r["outcome"] == outcome for r in records)
              for outcome in ("hit", "miss", "missing_title", "retrieval_failed")}
    successful = counts["hit"] + counts["miss"]
    return {
        "schema_version": 1,
        "artifact_type": "gt_title_retrieval_check",
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "input_path": str(input_path), "case_summary_path": str(case_summary_path),
            "concurrency": concurrency, "cutoff": cutoff,
        },
        "summary": {
            "total_samples": len(samples), "unique_gt_cases": len(grouped),
            "invalid_samples": len(input_errors), "successful_cases": successful,
            **{f"{key}_cases": value for key, value in counts.items()},
            "hit_rate_among_successful": counts["hit"] / successful if successful else None,
            "miss_samples": sum(r["sample_count"] for r in records if r["outcome"] == "miss"),
        },
        "misses": [r for r in records if r["outcome"] == "miss"],
        "errors": [r for r in records if r["outcome"] in ("missing_title", "retrieval_failed")],
        "input_errors": input_errors,
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Test-set JSON array containing caseID")
    parser.add_argument("--case-summary", required=True, help="JSON mapping case ID to {case_name, text}")
    parser.add_argument("--output", default="results/gt_title_retrieval_check.json")
    parser.add_argument("--config", help="Optional evaluate.py config; reads only retrieval settings")
    parser.add_argument("--retrieval-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--cutoff", type=int, help="Check top-K only; default checks all returned candidates")
    args = parser.parse_args(argv)
    try:
        for source in (args.input, args.case_summary, args.config):
            if source and Path(source).resolve() == Path(args.output).resolve():
                raise ValueError("output must not overwrite an input or config file")
        config = load_config_file(args.config) if args.config else {}
        retrieval = config.get("retrieval", {})
        if not isinstance(retrieval, Mapping):
            raise ValueError("configuration section 'retrieval' must be an object")
        url = args.retrieval_url or retrieval.get("url", DEFAULT_RETRIEVAL_URL)
        timeout = args.timeout if args.timeout is not None else retrieval.get("timeout", 30.0)
        concurrency = args.concurrency if args.concurrency is not None else retrieval.get("concurrency", 1)
        if not isinstance(url, str) or not url.strip():
            raise ValueError("retrieval URL must be a non-empty string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be an integer greater than zero")
        report = check_gt_retrieval(
            args.input, args.case_summary, SearchRetriever(url, timeout),
            concurrency=concurrency, cutoff=args.cutoff, progress=True,
        )
        report["configuration"].update({"retrieval_url": url, "timeout": timeout})
        write_json_atomically(report, args.output)
        print("GT title retrieval check: " + " ".join(
            f"{key}={value}" for key, value in report["summary"].items()
        ))
        for record in report["misses"]:
            print(f"MISS {record['expected_case_id']} {record['gt_case_title']} (samples={record['sample_count']})")
        print(f"Report saved to {args.output}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"GT title retrieval check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
