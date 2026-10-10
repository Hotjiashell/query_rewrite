"""Serial multi-query latency benchmark: generation -> retrieval -> filter -> fusion.

Hyperparameter search, metrics, model/client setup and disk writes are outside
the sample timers. Existing query artifacts can skip generation explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from evaluate import (
    QueryGenerationRunner, SearchRetriever, build_retrieval_artifact,
    calculate_metrics, load_config_file, load_dialogue_samples,
    load_generated_query_records, resolve_query_generation_config,
    resolve_retrieval_config, write_json_atomically,
)
from evaluate_laya import (
    LayaMultiQueryEvaluator, RELEVANCE_QUESTIONS, analyse_saved_results,
    parse_args as parse_evaluation_args, restore_dialogues,
)
from evaluate_startlux import create_filter
from gen_query import create_query_generator
from latency_benchmark import _latency_summary


class TimedFilter:
    """Observe the filter's actual boundaries, including unsuccessful calls."""

    def __init__(self, relevance_filter: Any) -> None:
        self.filter = relevance_filter
        self.started = None
        self.finished = None

    def judge(self, dialogue, cases):
        self.started = time.perf_counter()
        try:
            return self.filter.judge(dialogue, cases)
        finally:
            self.finished = time.perf_counter()


class SerialStartLuxBenchmark:
    def __init__(self, retriever, relevance_filter, samples, *, generator=None,
                 fusion_method="round_robin", top_k=10):
        self.samples = samples
        self.query_runner = QueryGenerationRunner(generator) if generator is not None else None
        self.timed_filter = TimedFilter(relevance_filter)
        self.evaluator = LayaMultiQueryEvaluator(
            retriever, self.timed_filter, fusion_method=fusion_method, top_k=top_k,
        )

    def run(self, *, progress=False):
        output = []
        for i, sample in enumerate(self.samples):
            self.timed_filter.started = self.timed_filter.finished = None
            started = time.perf_counter()
            source = self.query_runner.generate_sample(sample) if self.query_runner else sample
            generated = time.perf_counter()
            retrieval_started = time.perf_counter()
            record = self.evaluator.evaluate_query(source)
            finished = time.perf_counter()  # stop before metrics/search/report formatting
            timings = record["timings"]
            record["query_generation_time_sec"] = generated - started if self.query_runner else None
            record["retrieval_only_time_sec"] = timings.get("retrieval_seconds")
            record["filter_only_time_sec"] = (
                self.timed_filter.finished - self.timed_filter.started
                if self.timed_filter.finished is not None else None
            )
            record["fusion_only_time_sec"] = timings.get("fusion_seconds")
            record["time_to_retrieval_sec"] = (
                retrieval_started - started + timings["retrieval_seconds"]
                if record["retrieval_status"] in ("success", "partial_success") else None
            )
            record["time_to_filter_sec"] = (
                self.timed_filter.finished - started if record["filter_status"] == "success" else None
            )
            record["time_to_fusion_sec"] = finished - started if record["status"] == "success" else None
            record["total_attempt_time_sec"] = finished - started
            record["startlux_judgments"] = record.pop("laya_judgments")
            if "laya_filter_seconds" in timings:
                timings["startlux_filter_seconds"] = timings.pop("laya_filter_seconds")
            output.append(record)
            if progress:
                print(f"\r[startlux benchmark] {i + 1}/{len(self.samples)} "
                      f"success={sum(r['status'] == 'success' for r in output)}",
                      end="\n" if i + 1 == len(self.samples) else "", file=sys.stderr, flush=True)
        return output


def build_benchmark_artifact(records, *, input_path, retrieval, filter_metadata,
                             generation_included, test_num, setup_seconds):
    # This function is intentionally invoked only after all sample timers stop.
    artifact = build_retrieval_artifact(
        records, input_path=input_path, retrieval_url=retrieval.url,
        timeout=retrieval.timeout, concurrency=1,
        fusion_method=retrieval.fusion_method, top_k=retrieval.top_k,
    )
    artifact["configuration"].update(
        pipeline="serial_startlux_latency_benchmark", serial=True,
        startlux_filter=filter_metadata, relevance_questions=RELEVANCE_QUESTIONS,
        generation_included=generation_included,
        test_num_requested=test_num, test_num=len(records),
        timing_origin="before query generation" if generation_included else "before retrieval; query generation skipped",
        timing_excludes=["model/client setup", "hyperparameter search", "aggregate metrics", "file IO", "progress output"],
    )
    artifact["setup_seconds"] = setup_seconds
    artifact["prefilter_metrics"] = calculate_metrics([
        {**r, "status": "success" if r["retrieval_status"] in ("success", "partial_success") else "failed",
         "matched_rank": r["prefilter_matched_rank"]} for r in records
    ])
    # Keep top-level metrics compatible with the evaluation/offline replay tools.
    artifact["latency"] = {
        name: _latency_summary(records, field) for name, field in {
            "query_generation_only": "query_generation_time_sec",
            "retrieval_only": "retrieval_only_time_sec",
            "filter_only": "filter_only_time_sec",
            "fusion_only": "fusion_only_time_sec",
            "time_to_retrieval": "time_to_retrieval_sec",
            "time_to_filter": "time_to_filter_sec",
            "time_to_fusion": "time_to_fusion_sec",
            "total_attempt": "total_attempt_time_sec",
        }.items()
    }
    analysis_started = time.perf_counter()
    analyse_saved_results(artifact)
    artifact["offline_analysis_seconds"] = time.perf_counter() - analysis_started
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_evaluation_args(argv, backend="startlux", benchmark=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config_file(args.config)
        benchmark = config.get("startlux_benchmark", {})
        if not isinstance(benchmark, dict):
            raise ValueError("startlux_benchmark must be an object")
        test_num = args.test_num if args.test_num is not None else benchmark.get("test_num", 0)
        if isinstance(test_num, bool) or not isinstance(test_num, int) or test_num < 0:
            raise ValueError("test_num must be a nonnegative integer")
        args.output = args.output or benchmark.get("output_path", "results/latency_benchmark_startlux.json")
        args.input = args.input or benchmark.get("input_path")
        generator = None
        generation = None
        if args.query_file:
            input_path = args.query_file
            samples = load_generated_query_records(input_path)
            if args.dialogues_file:
                samples = restore_dialogues(samples, args.dialogues_file)
        else:
            generation_args = argparse.Namespace(**vars(args))
            generation_args.method = "custom_multi"
            generation = resolve_query_generation_config(generation_args)
            input_path = generation.input_path
            samples = load_dialogue_samples(input_path)
            generator = create_query_generator(generation.llm, "custom_multi", prompt_file=generation.prompt_file)
        retrieval_args = argparse.Namespace(**vars(args))
        retrieval_args.input = input_path
        retrieval = resolve_retrieval_config(retrieval_args)
        if Path(input_path).resolve() == Path(args.output).resolve():
            raise ValueError("output must not overwrite input")
        samples = samples[:test_num] if test_num else samples
        settings = config.get("startlux_filter", {})
        if not isinstance(settings, dict):
            raise ValueError("startlux_filter must be an object")
        setup_started = time.perf_counter()
        relevance_filter, metadata = create_filter(args, settings)
        setup_seconds = time.perf_counter() - setup_started
        records = SerialStartLuxBenchmark(
            SearchRetriever(retrieval.url, retrieval.timeout), relevance_filter, samples,
            generator=generator, fusion_method=retrieval.fusion_method, top_k=retrieval.top_k,
        ).run(progress=True)
        artifact = build_benchmark_artifact(
            records, input_path=input_path, retrieval=retrieval, filter_metadata=metadata,
            generation_included=generator is not None, test_num=test_num, setup_seconds=setup_seconds,
        )
        if generation:
            artifact["configuration"].update(method="custom_multi", model_name=generation.llm.model_name,
                                             prompt_file=generation.prompt_file)
        write_json_atomically(artifact, args.output)
        print(json.dumps({"output": args.output, "metrics": artifact["metrics"],
                          "prefilter_metrics": artifact["prefilter_metrics"],
                          "latency": artifact["latency"]}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"StartLux benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
