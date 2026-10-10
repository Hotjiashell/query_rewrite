"""Custom multi-query evaluation: retrieve -> deduplicate -> Laya filter -> fuse.

Laya is optional and imported only when retrieval actually needs the model.
Existing generated_queries artifacts can be used without an LLM connection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import (
    METRIC_CUTOFFS, RETRIEVAL_ARTIFACT_TYPE,
    GeneratedQueryRecord, QueryGenerationRunner, RetrievalEvaluator, RetrievedCase,
    SearchRetriever, build_query_artifact, build_retrieval_artifact, calculate_metrics,
    extract_retrieval_trace, fuse_round_robin, load_config_file,
    load_dialogue_samples, load_generated_query_records, resolve_query_generation_config,
    resolve_retrieval_config, write_json_atomically,
)
from gen_query import create_query_generator


RELEVANCE_QUESTIONS = {
    "relevance": {
        "type": "choice",
        "instructions": "根据完整对话判断案例标题是否与用户当前需要解决的问题相关。"
                        "结合产品、业务场景、故障现象和限制条件，不要仅凭泛化词重合判断。",
        "criteria": {
            "related": "案例标题对应的问题与对话中的当前问题一致，或能提供直接有用的处理方法。",
            "unrelated": "案例属于不同产品、业务或问题，或只有泛化词重合，不能帮助解决当前问题。",
        },
    },
}


def fuse_score(traces: Sequence[Sequence[RetrievedCase]], top_k: int) -> list[RetrievedCase]:
    """Highest retrieval score per ID; zero is a real score, not missing."""
    best: dict[str, RetrievedCase] = {}
    def score(case: RetrievedCase) -> float:
        return case.score if case.score is not None and math.isfinite(case.score) else -math.inf
    for trace in traces:
        for case in trace:
            if case.case_id not in best or score(case) > score(best[case.case_id]):
                best[case.case_id] = case
    ordered = sorted(best.values(), key=score, reverse=True)
    return [replace(case, rank=i + 1) for i, case in enumerate(ordered[:top_k])]


FUSIONS = {"round_robin": fuse_round_robin, "score": fuse_score}


def analyse_saved_results(artifact: dict[str, Any]) -> dict[str, Any]:
    """Compute raw union recall and exact threshold/fusion search without model calls.

    A sample changes only at its own probabilities. Accumulate its contributions
    over intervals of the global threshold list, avoiding a full dataset replay
    at every global threshold.
    """
    if artifact.get("artifact_type") != RETRIEVAL_ARTIFACT_TYPE:
        raise ValueError("analysis requires a retrieval_evaluation artifact")
    records = artifact.get("records")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("result records must be an array of objects")
    cutoffs = set(METRIC_CUTOFFS)
    configured_k = artifact.get("configuration", {}).get("top_k", 10)
    if isinstance(configured_k, int) and not isinstance(configured_k, bool) and configured_k > 0:
        cutoffs.add(configured_k)
    union_hits = {k: 0 for k in sorted(cutoffs)}
    all_candidate_hits = 0
    prepared = []
    skipped = []
    thresholds = {0.0, 1.0}
    for position, record in enumerate(records):
        raw_traces = record.get("per_query_traces")
        if not isinstance(raw_traces, list):
            raise ValueError("missing per_query_traces; use a saved multi-query filter result file")
        traces = []
        for item in raw_traces:
            if not isinstance(item, Mapping) or not isinstance(item.get("trace"), list):
                raise ValueError(f"invalid per_query_traces in record {position}")
            trace = []
            for case in item["trace"]:
                if not isinstance(case, Mapping) or not isinstance(case.get("rank"), int) or case["rank"] < 1:
                    raise ValueError(f"invalid case rank in record {position}")
                score = case.get("score")
                trace.append(RetrievedCase(
                    case["rank"], str(case["case_id"]), str(case.get("case_title") or ""),
                    score if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) else None,
                ))
            traces.append(sorted(trace, key=lambda case: case.rank))
        expected = record.get("expected_case_id")
        expected = str(expected) if expected is not None else None
        best_rank = min((case.rank for trace in traces for case in trace if case.case_id == expected), default=None)
        record["union_matched_rank"] = best_rank
        record["union_hit"] = best_rank is not None
        all_candidate_hits += record["union_hit"]
        record["union_top_k"] = {str(k): best_rank is not None and best_rank <= k for k in union_hits}
        for k in union_hits:
            union_hits[k] += record["union_top_k"][str(k)]
        probabilities = {}
        reason = None
        if record.get("filter_status") != "success":
            reason = "filter did not succeed"
        else:
            judgments = record.get("startlux_judgments", record.get("laya_judgments", []))
            if not isinstance(judgments, list):
                judgments = []
                reason = "invalid laya_judgments"
            for judgment in judgments:
                probability = judgment.get("related_probability") if isinstance(judgment, Mapping) else None
                if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                    reason = "invalid related probability"
                    break
                case_id = str(judgment["case_id"])
                if case_id in probabilities:
                    reason = "duplicate Laya judgment"
                    break
                probabilities[case_id] = float(probability)
            if set(probabilities) != {case.case_id for trace in traces for case in trace}:
                reason = reason or "Laya judgments do not cover all retrieved candidates"
            if expected is None:
                reason = reason or "missing expected_case_id"
        if reason:
            skipped.append({"sample_index": record.get("sample_index", position), "reason": reason})
            continue
        thresholds.update(probabilities.values())
        prepared.append((expected, traces, probabilities))
    total = len(records)
    artifact["union_top_k"] = {
        "total_samples": total,
        "definition": "Any query retrieves the expected case at original rank <= k, before relevance filtering or fusion.",
        **{f"hits_at_{k}": hits for k, hits in union_hits.items()},
        **{f"recall_at_{k}": hits / total if total else 0.0 for k, hits in union_hits.items()},
        "all_candidates_hits": all_candidate_hits,
        "all_candidates_recall": all_candidate_hits / total if total else 0.0,
    }
    ordered_thresholds = sorted(thresholds)
    # The >= policy has one attainable keep-set per interval ending at a
    # distinct observed probability (plus 0 and 1). No grid approximation.
    deltas = {method: [0] * (len(ordered_thresholds) + 1) for method in FUSIONS}
    kept_deltas = [0] * (len(ordered_thresholds) + 1)
    for expected, traces, probabilities in prepared:
        previous = None
        for threshold in sorted({0.0, 1.0, *probabilities.values()}):
            start = 0 if previous is None else bisect_right(ordered_thresholds, previous)
            end = bisect_right(ordered_thresholds, threshold)
            kept = {case_id for case_id, probability in probabilities.items() if probability >= threshold}
            filtered = [[case for case in trace if case.case_id in kept] for trace in traces]
            kept_deltas[start] += len(kept)
            kept_deltas[end] -= len(kept)
            for method, fuse in FUSIONS.items():
                hit = int(any(case.case_id == expected for case in fuse(filtered, 10)))
                deltas[method][start] += hit
                deltas[method][end] -= hit
            previous = threshold
    rows = []
    running_hits = dict.fromkeys(FUSIONS, 0)
    kept_count = 0
    for i, threshold in enumerate(ordered_thresholds):
        kept_count += kept_deltas[i]
        for method in FUSIONS:
            running_hits[method] += deltas[method][i]
            rows.append({
                "threshold": threshold, "fusion_method": method,
                "hits_at_10": running_hits[method],
                "recall_at_10": running_hits[method] / total if total else 0.0,
                "average_kept_candidates": kept_count / total if total else 0.0,
            })
    best_hits = max((row["hits_at_10"] for row in rows), default=0)
    tied = [row for row in rows if row["hits_at_10"] == best_hits] if prepared else []
    artifact["threshold_search"] = {
        "objective": "recall_at_10", "fusion_top_k": 10,
        "comparison": "related_probability >= threshold",
        "search_method": "exact_observed_probabilities_and_endpoints",
        "total_samples": total, "evaluable_samples": len(prepared),
        "skipped_samples": skipped,
        "denominator": "all input samples; unavailable judgments count as misses",
        "tested_threshold_count": len(ordered_thresholds),
        "tested_configuration_count": len(rows),
        "tie_break": "lowest threshold, then round_robin before score",
        "best": tied[0] if tied else None,
        "best_by_fusion": {
            method: max((row for row in rows if row["fusion_method"] == method), key=lambda row: row["hits_at_10"])
            if prepared else None for method in FUSIONS
        },
        "best_configuration_count": len(tied),
        "results": rows,
        "evaluation_scope": "Optimum on this saved dataset, not a held-out estimate.",
    }
    return artifact


class LayaRelevanceFilter:
    """Batch independent title/dialogue pairs; serialize a shared model's forwards."""

    def __init__(self, agent: Any, *, threshold: float = 0.5, batch_size: int = 16,
                 max_len: int = 1024, head_max_len: int = 256) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (batch_size, max_len, head_max_len)):
            raise ValueError("batch_size, max_len and head_max_len must be integers")
        if min(batch_size, max_len, head_max_len) < 1 or head_max_len >= max_len:
            raise ValueError("positive batch/token budgets required; head_max_len must be below max_len")
        self.agent = agent
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_len = max_len
        self.head_max_len = head_max_len
        self._lock = threading.Lock()

    def judge(self, dialogue: str, cases: Sequence[RetrievedCase]) -> list[dict[str, Any]]:
        if not cases:
            return []
        if any(not case.case_title.strip() for case in cases):
            raise ValueError("cannot judge a case with an empty title")
        states = [{"dialogue": dialogue, "case_title": case.case_title} for case in cases]
        with self._lock:
            results = self.agent.predict_batch(
                states, RELEVANCE_QUESTIONS, batch_size=self.batch_size,
                max_len=self.max_len, head_max_len=self.head_max_len,
                sort_by_length=True,
            )
        if len(results) != len(cases):
            raise ValueError("Laya returned an unexpected number of results")
        judgments = []
        for case, result in zip(cases, results):
            # A decision that did not read the full pair must not silently discard a case.
            if result.get("usage", {}).get("truncated"):
                raise ValueError("Laya input truncated; increase --max-len or shorten the dialogue")
            answer = result["answers"]["relevance"]
            probability = answer["probabilities"]["related"]
            if isinstance(probability, bool) or not isinstance(probability, (float, int)):
                raise ValueError("invalid Laya relevance probability")
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid Laya relevance probability")
            judgments.append({
                "case_id": case.case_id, "case_title": case.case_title,
                "related_probability": probability,
                "keep": probability >= self.threshold,
                "choice": answer["choice"],
                "probabilities": answer["probabilities"],
            })
        return judgments


class LayaMultiQueryEvaluator(RetrievalEvaluator):
    """Retain per-query ordering while judging each case ID once per dialogue."""

    def __init__(self, retriever: Any, relevance_filter: LayaRelevanceFilter,
                 *, fusion_method: str = "round_robin", top_k: int = 10) -> None:
        super().__init__(retriever, fusion_method=fusion_method, top_k=top_k)
        self.relevance_filter = relevance_filter

    def evaluate_query(self, source: GeneratedQueryRecord) -> dict[str, Any]:
        record: dict[str, Any] = {
            "sample_index": source.sample_index, "call_sno": source.call_sno,
            "expected_case_id": source.expected_case_id, "chat_content": source.chat_content,
            "query": source.query, "queries": source.queries,
            "query_status": source.status, "query_error": source.error,
            "retrieval_status": "skipped", "retrieval_error": None,
            "filter_status": "skipped", "filter_error": None,
            "retrieval_trace": [], "prefilter_trace": [], "per_query_traces": [],
            "laya_judgments": [], "matched_rank": None, "prefilter_matched_rank": None,
            "gt_case_title": None, "unique_candidate_count": 0,
            "kept_candidate_count": 0, "filtered_candidate_count": 0,
            "status": "failed", "error": None, "timings": {},
        }
        started = time.perf_counter()
        try:
            if source.status != "success":
                raise ValueError(source.error or "query generation did not succeed")
            if not source.queries:
                raise ValueError("multi-query artifact requires a nonempty queries array")
            if not source.chat_content:
                raise ValueError("missing chat_content; use --dialogues-file to restore source dialogues")
            queries = list(dict.fromkeys(source.queries))
            record["queries"] = queries
            traces: list[list[RetrievedCase]] = [[] for _ in queries]
            errors: list[str] = []
            succeeded = 0
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                futures = {executor.submit(self._retriever.retrieve, query): i
                           for i, query in enumerate(queries)}
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        trace = extract_retrieval_trace(future.result())
                        # Remove duplicates within a query, retaining its first rank.
                        seen = set()
                        for case in trace:
                            if case.case_id not in seen:
                                traces[i].append(case)
                                seen.add(case.case_id)
                        succeeded += 1
                    except Exception as exc:
                        errors.append(f"{queries[i]!r}: {type(exc).__name__}: {exc}")
            record["timings"]["retrieval_seconds"] = time.perf_counter() - started
            record["retrieval_error"] = "; ".join(errors) or None
            record["retrieval_status"] = ("partial_success" if errors else "success") if succeeded else "failed"
            record["per_query_traces"] = [
                {"query": query, "trace": [asdict(case) for case in trace]}
                for query, trace in zip(queries, traces)
            ]
            if not succeeded:
                raise ValueError("all query retrievals failed: " + "; ".join(errors))
            unique: dict[str, RetrievedCase] = {}
            for trace in traces:
                for case in trace:
                    if case.case_id not in unique or (not unique[case.case_id].case_title and case.case_title):
                        unique[case.case_id] = case
            record["unique_candidate_count"] = len(unique)
            fuse = FUSIONS[self._fusion_method]
            baseline = fuse(traces, self._top_k)
            record["prefilter_trace"] = [asdict(case) for case in baseline]
            record["prefilter_matched_rank"] = next(
                (case.rank for case in baseline if case.case_id == source.expected_case_id), None)
            record["gt_case_title"] = unique[source.expected_case_id].case_title if source.expected_case_id in unique else None
            filtering_started = time.perf_counter()
            record["filter_status"] = "running"
            judgments = self.relevance_filter.judge(source.chat_content, list(unique.values()))
            record["timings"]["laya_filter_seconds"] = time.perf_counter() - filtering_started
            record["laya_judgments"] = judgments
            kept = {item["case_id"] for item in judgments if item["keep"]}
            record["kept_candidate_count"] = len(kept)
            record["filtered_candidate_count"] = len(unique) - len(kept)
            fusion_started = time.perf_counter()
            filtered = [[case for case in trace if case.case_id in kept] for trace in traces]
            fused = fuse(filtered, self._top_k)
            record["timings"]["fusion_seconds"] = time.perf_counter() - fusion_started
            record.update(
                filter_status="success", status="success",
                retrieval_trace=[asdict(case) for case in fused],
                matched_rank=next((case.rank for case in fused if case.case_id == source.expected_case_id), None),
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if record["filter_status"] == "running":
                record["filter_status"] = "failed"
                record["filter_error"] = record["error"]
        record["timings"]["total_seconds"] = time.perf_counter() - started
        return record


def restore_dialogues(records: Sequence[GeneratedQueryRecord], path: str) -> list[GeneratedQueryRecord]:
    """Restore older artifacts by index, verifying IDs to prevent mismatched dialogues."""
    samples = load_dialogue_samples(path)
    restored = []
    for record in records:
        if not record.chat_content:
            index = record.sample_index
            if not 0 <= index < len(samples):
                raise ValueError(f"sample_index {index} is outside the source dialogue file")
            sample = samples[index]
            if sample.expected_case_id != record.expected_case_id or sample.call_sno != record.call_sno:
                raise ValueError(f"source dialogue IDs do not match at sample_index {index}")
            record = replace(record, chat_content=sample.dialogue)
        restored.append(record)
    return restored


def parse_args(argv: Sequence[str] | None = None, *, backend: str = "laya",
               benchmark: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ if backend == "laya" else "Custom multi-query evaluation with StartLux-Decision-4B.")
    if benchmark:
        parser.description = "Serial StartLux multi-query latency benchmark; offline analysis is excluded."
        parser.set_defaults(stage="all", concurrency=1, query_concurrency=1, query_output=None)
        parser.add_argument("--test-num", type=int, help="samples to benchmark; 0 means all (default 0)")
    else:
        parser.add_argument("stage", nargs="?", default="retrieve", choices=("generate", "retrieve", "all", "analyze"))
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--input", help="source dialogues for generate/all; query artifact for retrieve")
    parser.add_argument("--query-file", help="existing generated_queries artifact; bypasses generation")
    if not benchmark:
        parser.add_argument("--query-output", help="generated query artifact path")
    parser.add_argument("--output", help="final retrieval artifact (query artifact in generate mode)")
    parser.add_argument("--dialogues-file", help="restore missing chat_content in an older query artifact")
    parser.add_argument("--prompt-file", help="custom multi-query prompt containing {dialogue}")
    if not benchmark:
        parser.add_argument("--concurrency", type=int, help="concurrent retrieval samples; shared-model forwards are serialized")
        parser.add_argument("--query-concurrency", type=int)
    for name in ("base-url", "model", "api-key"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--retrieval-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--fusion-method", choices=tuple(FUSIONS))
    parser.add_argument("--top-k", type=int)
    if backend == "laya":
        parser.add_argument("--laya-model", help="Hub checkpoint or local checkpoint directory")
    else:
        parser.add_argument("--startlux-model", help="local checkpoint directory (default: StartLux-Decision-4B)")
        parser.add_argument("--startlux-endpoint", help="StartLux HTTP endpoint (default: http://127.0.0.1:8090/v1/systemone)")
        parser.add_argument("--startlux-timeout", type=float, help="StartLux HTTP timeout in seconds (default: 120)")
        parser.add_argument("--max-batch-tokens", type=int, help="padded batch token budget (default: 65536)")
        parser.set_defaults(laya_model=None, head_max_len=None, fast=None)
    parser.add_argument("--device", help="cpu, cuda, mps; default auto" if backend == "laya" else "PyTorch device: cpu or cuda; default auto")
    parser.add_argument("--threshold", type=float, help="minimum P(related), default 0.5")
    parser.add_argument("--batch-size", type=int, help="case pairs per model batch, default 16")
    parser.add_argument("--max-len", type=int, help="total token budget, default " + ("1024" if backend == "laya" else "4096"))
    if backend == "laya":
        parser.add_argument("--head-max-len", type=int, help="question token budget, default 256")
        parser.add_argument("--fast", action="store_true", default=None, help="TileLang CUDA fast path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, args: argparse.Namespace | None = None,
         filter_factory=None) -> int:
    args = args if args is not None else parse_args(argv)
    try:
        if args.stage == "analyze":
            if not args.input or not args.output:
                raise ValueError("analyze requires --input RESULT_FILE and --output OUTPUT_FILE")
            if Path(args.input).resolve() == Path(args.output).resolve():
                raise ValueError("analysis output must not overwrite the source result file")
            artifact = analyse_saved_results(load_config_file(args.input))
            artifact["analysis_source_path"] = args.input
            write_json_atomically(artifact, args.output)
            print(json.dumps({"output": args.output, "union_top_k": artifact["union_top_k"],
                              "best": artifact["threshold_search"]["best"]}, ensure_ascii=False, indent=2))
            return 0
        if args.stage == "generate" and args.query_file:
            raise ValueError("--query-file is for retrieve/all, not generate")
        config = load_config_file(args.config)
        query_path = args.query_file
        if args.stage in ("generate", "all") and not query_path:
            generation_args = argparse.Namespace(**vars(args))
            generation_args.method = "custom_multi"
            generation_args.output = args.output if args.stage == "generate" else args.query_output
            generation_args.concurrency = args.query_concurrency
            generation = resolve_query_generation_config(generation_args)
            if Path(generation.input_path).resolve() == Path(generation.output_path).resolve():
                raise ValueError("query output must not overwrite the source dialogue file")
            records = QueryGenerationRunner(create_query_generator(
                generation.llm, "custom_multi", prompt_file=generation.prompt_file,
            )).generate(load_dialogue_samples(generation.input_path), generation.concurrency, progress=True)
            artifact = build_query_artifact(
                records, input_path=generation.input_path, model_name=generation.llm.model_name,
                concurrency=generation.concurrency, method="custom_multi", prompt_file=generation.prompt_file,
            )
            write_json_atomically(artifact, generation.output_path)
            query_path = generation.output_path
            if args.stage == "generate":
                print(f"Query generation complete: {generation.output_path}")
                return 0
        retrieval_args = argparse.Namespace(**vars(args))
        retrieval_args.input = query_path or args.input
        retrieval = resolve_retrieval_config(retrieval_args)
        if Path(retrieval.input_path).resolve() == Path(retrieval.output_path).resolve():
            raise ValueError("output must not overwrite the source query artifact")
        records = load_generated_query_records(retrieval.input_path)
        if args.dialogues_file:
            records = restore_dialogues(records, args.dialogues_file)
        filter_name = "startlux_filter" if filter_factory else "laya_filter"
        settings = config.get(filter_name, {})
        if not isinstance(settings, Mapping):
            raise ValueError(f"{filter_name} configuration must be an object")
        def setting(name: str, default: Any) -> Any:
            value = getattr(args, name)
            return value if value is not None else settings.get(name, default)
        options = {
            "threshold": setting("threshold", 0.5), "batch_size": setting("batch_size", 16),
            "max_len": setting("max_len", 1024), "head_max_len": setting("head_max_len", 256),
        }
        model = args.laya_model or settings.get("model", "convaiinnovations/laya-multilingual")
        device = setting("device", None)
        fast = setting("fast", False)
        load_started = time.perf_counter()
        if filter_factory:
            relevance_filter, filter_metadata = filter_factory(args, settings)
        else:
            # Validate budgets before downloading any checkpoint.
            relevance_filter = LayaRelevanceFilter(None, **options)
            try:
                import laya
            except ImportError as exc:
                raise RuntimeError("Install Laya with: python -m pip install -r requirements-laya.txt") from exc
            relevance_filter.agent = laya.load(model, device=device, fast=fast)
            filter_metadata = {**options, "model": model, "device": device, "fast": fast}
        load_seconds = time.perf_counter() - load_started
        evaluated = LayaMultiQueryEvaluator(
            SearchRetriever(retrieval.url, retrieval.timeout), relevance_filter,
            fusion_method=retrieval.fusion_method, top_k=retrieval.top_k,
        ).evaluate(records, retrieval.concurrency, progress=True)
        if filter_factory:
            for record in evaluated:
                record["startlux_judgments"] = record.pop("laya_judgments")
                if "laya_filter_seconds" in record["timings"]:
                    record["timings"]["startlux_filter_seconds"] = record["timings"].pop("laya_filter_seconds")
        artifact = build_retrieval_artifact(
            evaluated, input_path=retrieval.input_path, retrieval_url=retrieval.url,
            timeout=retrieval.timeout, concurrency=retrieval.concurrency,
            fusion_method=retrieval.fusion_method, top_k=retrieval.top_k,
        )
        artifact["configuration"]["pipeline"] = "multi_query_" + filter_name
        artifact["configuration"][filter_name] = filter_metadata
        artifact["configuration"]["relevance_questions"] = RELEVANCE_QUESTIONS
        artifact["model_load_seconds"] = load_seconds
        artifact["prefilter_metrics"] = calculate_metrics([
            {**record, "status": "success" if record["retrieval_status"] in ("success", "partial_success") else "failed",
             "matched_rank": record["prefilter_matched_rank"]} for record in evaluated
        ])
        analyse_saved_results(artifact)
        write_json_atomically(artifact, retrieval.output_path)
        print(json.dumps({"output": retrieval.output_path, "prefilter_metrics": artifact["prefilter_metrics"],
                          "metrics": artifact["metrics"], "union_top_k": artifact["union_top_k"],
                          "best": artifact["threshold_search"]["best"]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
