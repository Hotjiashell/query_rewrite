"""Collect badcases (top-K misses) from a retrieval-evaluation artifact.

Cross-references a case summary file (``case_id -> {"case_name": ..., "text":
...}``) so every miss is annotated with the ground-truth case's title, since
the evaluation artifact's own ``gt_case_title`` is only populated when the
ground-truth case happens to appear in the retrieval trace:

    python collect_badcases.py \
        results/method_v1.json \
        data/case_example.json \
        --output results/badcases.json

Only the compact fields needed for case review are copied into the report;
arbitrary fields such as case content are never propagated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import QUERY_ARTIFACT_TYPE, RETRIEVAL_ARTIFACT_TYPE, write_json_atomically


DEFAULT_CUTOFF = 10


def _load_result_artifact(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"result file does not exist: {source}") from exc
    except OSError as exc:
        raise ValueError(f"could not read result file {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"result file is not valid JSON: {source}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"result file root must be a JSON object: {source}")
    if payload.get("artifact_type") != RETRIEVAL_ARTIFACT_TYPE:
        raise ValueError(f"input is not a '{RETRIEVAL_ARTIFACT_TYPE}' artifact: {source}")
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"result artifact 'records' must be an array: {source}")
    return payload


def _load_query_records(
    result_payload: Mapping[str, Any], results_path: str | Path
) -> dict[int, Mapping[str, Any]]:
    """Load generated-query records referenced by a retrieval artifact.

    Older retrieval artifacts did not copy ``queries`` into each result record,
    but retain the generated-query artifact path in their configuration.
    Missing or unusable references are treated as no fallback data so existing
    result files remain collectible.
    """

    configuration = result_payload.get("configuration")
    source_query_path = configuration.get("source_query_path") if isinstance(configuration, Mapping) else None
    if not isinstance(source_query_path, str) or not source_query_path.strip():
        return {}

    result_path = Path(results_path)
    source = Path(source_query_path)
    candidates = [source] if source.is_absolute() else [source, result_path.parent / source]
    query_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if query_path is None:
        return {}
    try:
        query_payload = json.loads(query_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(query_payload, Mapping) or query_payload.get("artifact_type") != QUERY_ARTIFACT_TYPE:
        return {}
    raw_records = query_payload.get("records")
    if not isinstance(raw_records, list):
        return {}

    records: dict[int, Mapping[str, Any]] = {}
    for position, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            continue
        sample_index = item.get("sample_index")
        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            sample_index = position
        records.setdefault(sample_index, item)
    return records


def _normalise_queries(raw_queries: Any) -> list[str] | None:
    if not isinstance(raw_queries, list):
        return None
    queries = [value.strip() for value in raw_queries if isinstance(value, str) and value.strip()]
    return queries or None


def load_case_summary(path: str | Path) -> Mapping[str, Any]:
    """Load a ``case_id -> {"case_name": ..., "text": ...}`` lookup table."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"case summary file does not exist: {source}") from exc
    except OSError as exc:
        raise ValueError(f"could not read case summary file {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"case summary file is not valid JSON: {source}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"case summary file root must be a JSON object: {source}")
    return payload


def _is_miss(record: Mapping[str, Any], cutoff: int) -> bool:
    """A badcase is a successfully-evaluated sample that missed within ``cutoff``.

    Samples whose query generation or retrieval outright failed are excluded:
    those are infrastructure problems, not misses attributable to query quality.
    """

    if record.get("status") != "success":
        return False
    rank = record.get("matched_rank")
    hit = isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= cutoff
    return not hit


def _normalise_trace(raw_trace: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_trace, list):
        return []
    trace: list[dict[str, Any]] = []
    for item in raw_trace:
        if not isinstance(item, Mapping):
            continue
        case_id = item.get("case_id")
        rank = item.get("rank")
        if case_id is None or isinstance(rank, bool) or not isinstance(rank, int):
            continue
        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool) else None
        trace.append(
            {
                "rank": rank,
                "case_id": str(case_id),
                "case_title": str(item.get("case_title") or ""),
                "score": score,
            }
        )
    return sorted(trace, key=lambda entry: entry["rank"])


def _lookup_case_name(case_summary: Mapping[str, Any], case_id: str | None) -> tuple[str | None, bool]:
    """Return ``(gt_case_title, gt_case_found)`` for one ground-truth case ID."""

    if case_id is None:
        return None, False
    entry = case_summary.get(str(case_id))
    if not isinstance(entry, Mapping):
        return None, False
    case_name = entry.get("case_name")
    return (str(case_name) if case_name is not None else None), True


def collect_badcases(
    results_path: str | Path,
    case_summary_path: str | Path,
    *,
    cutoff: int = DEFAULT_CUTOFF,
) -> dict[str, Any]:
    """Build a report of every top-``cutoff`` miss, with ground-truth titles filled in."""

    if cutoff < 1:
        raise ValueError("cutoff must be greater than zero")
    result_payload = _load_result_artifact(results_path)
    case_summary = load_case_summary(case_summary_path)
    query_records = _load_query_records(result_payload, results_path)

    badcases: list[dict[str, Any]] = []
    for position, record in enumerate(result_payload["records"]):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {position} in {results_path} must be a JSON object")
        if not _is_miss(record, cutoff):
            continue

        expected_case_id = record.get("expected_case_id")
        sample_index = record.get("sample_index")
        source_record = (
            query_records.get(sample_index)
            if isinstance(sample_index, int) and not isinstance(sample_index, bool)
            else None
        )
        query = record.get("query") if isinstance(record.get("query"), str) else None
        queries = _normalise_queries(record.get("queries"))
        if source_record is not None:
            if query is None and isinstance(source_record.get("query"), str):
                query = source_record["query"]
            if queries is None:
                queries = _normalise_queries(source_record.get("queries"))
        gt_case_title, gt_case_found = _lookup_case_name(case_summary, expected_case_id)
        badcases.append(
            {
                "sample_index": record.get("sample_index"),
                "call_sno": record.get("call_sno"),
                "chat_content": record.get("chat_content"),
                "expected_case_id": expected_case_id,
                "query": query,
                "queries": queries,
                "matched_rank": record.get("matched_rank"),
                "retrieval_trace": _normalise_trace(record.get("retrieval_trace")),
                "gt_case_title": gt_case_title,
                "gt_case_found": gt_case_found,
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "badcase_report",
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "results_path": str(results_path),
            "case_summary_path": str(case_summary_path),
            "cutoff": cutoff,
        },
        "summary": {
            "cutoff": cutoff,
            "total_records": len(result_payload["records"]),
            "badcase_count": len(badcases),
            "unresolved_gt_case_count": sum(not item["gt_case_found"] for item in badcases),
        },
        "records": badcases,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect samples that missed within top-K, with ground-truth case titles filled in."
    )
    parser.add_argument("results", help="retrieval_evaluation JSON produced by evaluate.py")
    parser.add_argument("case_summary", help="Case summary JSON mapping case_id -> {case_name, text}")
    parser.add_argument(
        "--cutoff",
        type=int,
        default=DEFAULT_CUTOFF,
        help="Miss cutoff, e.g. 10 for top-10 (default: 10)",
    )
    parser.add_argument("--output", required=True, help="Output JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = collect_badcases(args.results, args.case_summary, cutoff=args.cutoff)
        write_json_atomically(report, args.output)
        summary = report["summary"]
        print(
            "Badcase collection complete: "
            f"total={summary['total_records']} badcases={summary['badcase_count']} "
            f"unresolved_gt_case={summary['unresolved_gt_case_count']} output={args.output}"
        )
        return 0
    except ValueError as exc:
        print(f"Badcase collection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
