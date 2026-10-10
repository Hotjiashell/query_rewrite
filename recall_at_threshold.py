"""Replay a saved StartLux/Laya result at one threshold, without network calls."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from evaluate import (
    RETRIEVAL_ARTIFACT_TYPE, RetrievedCase, calculate_metrics,
    load_config_file, write_json_atomically,
)
from evaluate_laya import FUSIONS


def evaluate_threshold(artifact: Mapping[str, Any], threshold: float,
                       fusion_method: str | None = None) -> dict[str, Any]:
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if artifact.get("artifact_type") != RETRIEVAL_ARTIFACT_TYPE:
        raise ValueError("input must be a retrieval_evaluation result artifact")
    fusion_method = fusion_method or artifact.get("configuration", {}).get("fusion_method", "round_robin")
    if fusion_method not in FUSIONS:
        raise ValueError(f"unsupported fusion_method: {fusion_method}")
    source_records = artifact.get("records")
    if not isinstance(source_records, list):
        raise ValueError("records must be an array")
    records = []
    for position, source in enumerate(source_records):
        record = {
            "sample_index": source.get("sample_index", position) if isinstance(source, Mapping) else position,
            "call_sno": source.get("call_sno") if isinstance(source, Mapping) else None,
            "expected_case_id": source.get("expected_case_id") if isinstance(source, Mapping) else None,
            "status": "failed", "error": None, "matched_rank": None,
            "retrieval_trace": [], "kept_candidate_count": 0,
        }
        try:
            if not isinstance(source, Mapping):
                raise ValueError("record must be an object")
            if source.get("filter_status") != "success":
                raise ValueError(source.get("filter_error") or "original relevance filtering did not succeed")
            if record["expected_case_id"] is None:
                raise ValueError("missing expected_case_id")
            raw_traces = source.get("per_query_traces")
            if not isinstance(raw_traces, list):
                raise ValueError("missing per_query_traces; original final trace alone cannot replay filtering")
            traces = []
            for item in raw_traces:
                trace = []
                for case in item["trace"]:
                    rank = case["rank"]
                    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                        raise ValueError("case rank must be a positive integer")
                    score = case.get("score")
                    trace.append(RetrievedCase(
                        rank, str(case["case_id"]), str(case.get("case_title") or ""),
                        score if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) else None,
                    ))
                traces.append(sorted(trace, key=lambda case: case.rank))
            judgments = source.get("startlux_judgments", source.get("laya_judgments"))
            if not isinstance(judgments, list):
                raise ValueError("missing startlux_judgments/laya_judgments")
            probabilities = {}
            for item in judgments:
                probability = item["related_probability"]
                if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("invalid related_probability")
                case_id = str(item["case_id"])
                if case_id in probabilities:
                    raise ValueError("duplicate relevance judgment")
                probabilities[case_id] = probability
            if set(probabilities) != {case.case_id for trace in traces for case in trace}:
                raise ValueError("relevance judgments do not cover all original candidates")
            kept = {case_id for case_id, probability in probabilities.items() if probability >= threshold}
            filtered = [[case for case in trace if case.case_id in kept] for trace in traces]
            fused = FUSIONS[fusion_method](filtered, 10)
            record.update(
                status="success", kept_candidate_count=len(kept),
                retrieval_trace=[asdict(case) for case in fused],
                matched_rank=next((case.rank for case in fused if case.case_id == str(record["expected_case_id"])), None),
            )
        except (ValueError, TypeError, KeyError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return {
        "artifact_type": RETRIEVAL_ARTIFACT_TYPE, "schema_version": 1,
        "configuration": {"pipeline": "offline_threshold_replay", "threshold": threshold,
                          "comparison": "related_probability >= threshold",
                          "fusion_method": fusion_method, "top_k": 10},
        "denominator": "all input samples; failed or incomplete judgments count as misses",
        "metrics": calculate_metrics(records), "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="saved evaluate_startlux.py result JSON")
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--fusion-method", choices=tuple(FUSIONS), help="default: source file fusion method")
    parser.add_argument("--output", help="optionally save replayed traces and recall metrics")
    args = parser.parse_args(argv)
    try:
        from pathlib import Path
        if args.output and Path(args.input).resolve() == Path(args.output).resolve():
            raise ValueError("output must not overwrite the original result")
        report = evaluate_threshold(load_config_file(args.input), args.threshold, args.fusion_method)
        report["configuration"]["source_result_path"] = args.input
        if args.output:
            write_json_atomically(report, args.output)
        print(f"threshold={args.threshold:g} fusion={report['configuration']['fusion_method']}")
        metrics = report["metrics"]
        print(f"samples={metrics['total_samples']} success={metrics['successful_samples']} failed={metrics['failed_samples']}")
        for k in (1, 3, 5, 10):
            print(f"Recall@{k}: {metrics[f'recall_at_{k}']:.6f} "
                  f"({metrics[f'hits_at_{k}']}/{metrics['total_samples']})")
        if args.output:
            print(f"output={args.output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
