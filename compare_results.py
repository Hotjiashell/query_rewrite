"""Compare two retrieval-evaluation artifacts.

The main use case is finding samples recalled by the first method but missed
by the second method, for example when investigating a prompt change:

    python compare_results.py results/baseline.json results/method_v1.json

Only the compact, already-sanitized fields needed for analysis are copied to
the comparison report.  In particular, arbitrary fields such as case content
or API metadata are never propagated from the source files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import RETRIEVAL_ARTIFACT_TYPE, write_json_atomically


DEFAULT_CUTOFF = 10
SUPPORTED_CUTOFFS = (1, 3, 5, 10)


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
        raise ValueError(
            f"input is not a '{RETRIEVAL_ARTIFACT_TYPE}' artifact: {source}"
        )
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"result artifact 'records' must be an array: {source}")
    return payload


def _record_key(record: Mapping[str, Any], position: int) -> tuple[str, Any]:
    """Return a stable key, preferring the evaluator's sample index."""

    sample_index = record.get("sample_index")
    if isinstance(sample_index, int) and not isinstance(sample_index, bool):
        return ("sample_index", sample_index)
    call_sno = record.get("call_sno")
    if call_sno is not None and str(call_sno).strip():
        return ("call_sno", str(call_sno).strip())
    # Older or hand-written files may omit both identifiers.  Position is the
    # only deterministic fallback, and is still useful for side-by-side files.
    return ("position", position)


def _index_records(payload: Mapping[str, Any], source: str | Path) -> dict[tuple[str, Any], Mapping[str, Any]]:
    indexed: dict[tuple[str, Any], Mapping[str, Any]] = {}
    for position, raw_record in enumerate(payload["records"]):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"record {position} in {source} must be a JSON object")
        key = _record_key(raw_record, position)
        if key in indexed:
            raise ValueError(f"duplicate record key {key[1]!r} in {source}")
        indexed[key] = raw_record
    return indexed


def _normalise_trace(raw_trace: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_trace, list):
        return []
    trace: list[dict[str, Any]] = []
    for item in raw_trace:
        if not isinstance(item, Mapping):
            continue
        rank = item.get("rank")
        case_id = item.get("case_id")
        if isinstance(rank, bool) or not isinstance(rank, int) or case_id is None:
            continue
        trace.append(
            {
                "rank": rank,
                "case_id": str(case_id),
                "case_title": str(item.get("case_title") or ""),
            }
        )
    return sorted(trace, key=lambda item: item["rank"])


def _hit_at(record: Mapping[str, Any] | None, cutoff: int) -> bool:
    if record is None or record.get("status") != "success":
        return False
    rank = record.get("matched_rank")
    return isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= cutoff


def _safe_record(record: Mapping[str, Any] | None, cutoff: int) -> dict[str, Any]:
    """Copy only fields useful for comparing two methods."""

    if record is None:
        return {
            "present": False,
            "query": None,
            "status": "missing",
            "retrieval_status": "missing",
            "retrieval_error": "record is absent from this result file",
            "matched_rank": None,
            "hit": False,
            "retrieval_trace": [],
        }
    return {
        "present": True,
        "query": record.get("query") if isinstance(record.get("query"), str) else None,
        "status": str(record.get("status") or ""),
        "retrieval_status": str(record.get("retrieval_status") or ""),
        "retrieval_error": record.get("retrieval_error"),
        "query_status": str(record.get("query_status") or ""),
        "query_error": record.get("query_error"),
        "matched_rank": record.get("matched_rank")
        if isinstance(record.get("matched_rank"), int)
        and not isinstance(record.get("matched_rank"), bool)
        else None,
        "hit": _hit_at(record, cutoff),
        "retrieval_trace": _normalise_trace(record.get("retrieval_trace")),
    }


def compare_result_artifacts(
    first_path: str | Path,
    second_path: str | Path,
    *,
    cutoff: int = DEFAULT_CUTOFF,
) -> dict[str, Any]:
    """Find samples hit by ``first_path`` but missed by ``second_path``."""

    if cutoff < 1:
        raise ValueError("cutoff must be greater than zero")
    first_payload = _load_result_artifact(first_path)
    second_payload = _load_result_artifact(second_path)
    first_records = _index_records(first_payload, first_path)
    second_records = _index_records(second_payload, second_path)

    all_keys = list(first_records)
    all_keys.extend(key for key in second_records if key not in first_records)
    target_records: list[dict[str, Any]] = []
    first_hit_count = second_hit_count = both_hit_count = both_miss_count = 0
    for key in all_keys:
        first = first_records.get(key)
        second = second_records.get(key)
        first_hit = _hit_at(first, cutoff)
        second_hit = _hit_at(second, cutoff)
        first_hit_count += first_hit
        second_hit_count += second_hit
        if first_hit and second_hit:
            both_hit_count += 1
        elif not first_hit and not second_hit:
            both_miss_count += 1
        elif first_hit and not second_hit:
            target_records.append(
                {
                    "sample_index": first.get("sample_index") if first else second.get("sample_index"),
                    "call_sno": first.get("call_sno") if first else second.get("call_sno"),
                    "expected_case_id": first.get("expected_case_id")
                    if first
                    else second.get("expected_case_id"),
                    "first": _safe_record(first, cutoff),
                    "second": _safe_record(second, cutoff),
                }
            )

    return {
        "schema_version": 1,
        "artifact_type": "retrieval_comparison",
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "first_path": str(first_path),
            "second_path": str(second_path),
            "cutoff": cutoff,
        },
        "summary": {
            "cutoff": cutoff,
            "first_total": len(first_records),
            "second_total": len(second_records),
            "aligned_samples": len(all_keys),
            "first_only_samples": sum(key not in second_records for key in first_records),
            "second_only_samples": sum(key not in first_records for key in second_records),
            "first_hits": first_hit_count,
            "second_hits": second_hit_count,
            "both_hits": both_hit_count,
            "both_misses": both_miss_count,
            "first_hit_second_miss": len(target_records),
        },
        "records": target_records,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find samples recalled by the first result file but missed by the second."
    )
    parser.add_argument("first", help="First retrieval-evaluation JSON (the method expected to win)")
    parser.add_argument("second", help="Second retrieval-evaluation JSON")
    parser.add_argument(
        "--cutoff",
        type=int,
        choices=SUPPORTED_CUTOFFS,
        default=DEFAULT_CUTOFF,
        help="Recall cutoff to compare (default: 10)",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path; without it, only a summary is printed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = compare_result_artifacts(args.first, args.second, cutoff=args.cutoff)
        if args.output:
            write_json_atomically(report, args.output)
        summary = report["summary"]
        print(
            f"Recall@{args.cutoff} comparison: "
            f"first_hits={summary['first_hits']} second_hits={summary['second_hits']} "
            f"first_hit_second_miss={summary['first_hit_second_miss']} "
            f"aligned={summary['aligned_samples']}"
        )
        if args.output:
            print(f"Report written to {args.output}")
        for item in report["records"]:
            print(
                f"- sample_index={item['sample_index']} "
                f"expected_case_id={item['expected_case_id']} "
                f"first_rank={item['first']['matched_rank']} "
                f"second_rank={item['second']['matched_rank']} "
                f"first_query={item['first']['query']!r}"
            )
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
