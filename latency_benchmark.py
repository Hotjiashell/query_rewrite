"""Serial end-to-end latency and recall benchmark.

Each selected sample is processed in input order:

    query generation -> retrieval -> reranking

The two latency measurements start immediately before query generation. This
makes ``time_to_retrieval_sec`` the end-to-end time until retrieval returns,
and ``time_to_rerank_sec`` the end-to-end time until reranking returns.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from evaluate import (
    DEFAULT_FUSION_METHOD,
    DEFAULT_TOP_K,
    DEFAULT_RETRIEVAL_URL,
    QueryGenerationRunner,
    RetrievalEvaluator,
    SearchRetriever,
    calculate_metrics,
    load_dialogue_samples,
    load_config_file,
    write_json_atomically,
)
from gen_query import (
    CUSTOM_METHODS,
    SUPPORTED_QUERY_METHODS,
    LLMConfig,
    build_openai_client,
    create_query_generator,
    normalize_query_method,
)
from rerank_cases import (
    DEFAULT_CANDIDATE_LIMIT,
    CaseReranker,
    RerankConfig,
    calculate_rerank_metrics,
)


BENCHMARK_ARTIFACT_TYPE = "serial_latency_recall_benchmark"
DEFAULT_OUTPUT_PATH = "results/latency_benchmark.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    input_path: str
    output_path: str
    test_num: int
    llm: LLMConfig
    method: str
    prompt_file: str | None
    retrieval_url: str
    retrieval_timeout: float
    fusion_method: str
    top_k: int
    candidate_limit: int


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration section '{name}' must be an object")
    return section


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _string_setting(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid configuration value: {name}")
    return value.strip()


def _integer_setting(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        if minimum == 0:
            requirement = "greater than or equal to zero"
        else:
            requirement = "greater than zero"
        raise ValueError(f"configuration value '{name}' must be an integer {requirement}")
    return value


def _positive_number_setting(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"configuration value '{name}' must be a number greater than zero")
    return float(value)


def _temperature_setting(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2:
        raise ValueError(f"configuration value '{name}' must be between 0 and 2")
    return float(value)


def _resolve_llm_config(args: argparse.Namespace, config: Mapping[str, Any]) -> LLMConfig:
    section = _config_section(config, "llm")
    configured_key_env = section.get("api_key_env")
    if configured_key_env is not None and (
        not isinstance(configured_key_env, str) or not configured_key_env.strip()
    ):
        raise ValueError("configuration value 'llm.api_key_env' must be a non-empty string")
    configured_key = os.getenv(configured_key_env.strip()) if configured_key_env else None
    return LLMConfig(
        base_url=_string_setting(
            "llm.base_url",
            _first_defined(args.base_url, section.get("base_url"), os.getenv("OPENAI_BASE_URL")),
        ),
        model_name=_string_setting(
            "llm.model_name",
            _first_defined(args.model, section.get("model_name"), os.getenv("OPENAI_MODEL")),
        ),
        api_key=_string_setting(
            "llm.api_key",
            _first_defined(args.api_key, section.get("api_key"), configured_key, os.getenv("OPENAI_API_KEY")),
        ),
        temperature=_temperature_setting(
            "llm.temperature",
            _first_defined(args.temperature, section.get("temperature"), 0.0),
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a serial query-generation, retrieval, and reranking benchmark."
    )
    parser.add_argument("--config", default="config.json", help="JSON config file (default: config.json)")
    parser.add_argument("--input", help="override the benchmark input, or query_generation.input_path")
    parser.add_argument("--output", help=f"benchmark artifact path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument(
        "--test-num",
        type=int,
        help="number of samples to run; 0 means all samples (default: config value or 0)",
    )
    parser.add_argument(
        "--method",
        type=normalize_query_method,
        choices=SUPPORTED_QUERY_METHODS,
        help="query-generation method",
    )
    parser.add_argument("--prompt-file", help="override query_generation.prompt_file")
    parser.add_argument("--base-url", help="override llm.base_url")
    parser.add_argument("--model", help="override llm.model_name")
    parser.add_argument("--api-key", help="override llm.api_key")
    parser.add_argument("--temperature", type=float, help="override llm.temperature (0 to 2)")
    parser.add_argument("--retrieval-url", help="override retrieval.url")
    parser.add_argument("--timeout", type=float, help="override retrieval.timeout")
    parser.add_argument(
        "--fusion-method",
        choices=("round_robin", "score"),
        help="override retrieval.fusion_method",
    )
    parser.add_argument("--top-k", type=int, help="override retrieval.top_k")
    parser.add_argument("--candidate-limit", type=int, help="override rerank.candidate_limit")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> BenchmarkConfig:
    config = load_config_file(args.config)
    benchmark = _config_section(config, "benchmark")
    generation = _config_section(config, "query_generation")
    retrieval = _config_section(config, "retrieval")
    rerank = _config_section(config, "rerank")

    method = normalize_query_method(
        _first_defined(args.method, benchmark.get("method"), generation.get("method"), "baseline")
    )
    prompt_file = _first_defined(
        args.prompt_file,
        benchmark.get("prompt_file"),
        generation.get("prompt_file"),
    )
    if method in CUSTOM_METHODS and not prompt_file:
        raise ValueError(f"query_generation.prompt_file is required when method is '{method}'")

    test_num = _integer_setting(
        "benchmark.test_num",
        _first_defined(args.test_num, benchmark.get("test_num"), 0),
        minimum=0,
    )
    candidate_limit = _integer_setting(
        "rerank.candidate_limit",
        _first_defined(args.candidate_limit, benchmark.get("candidate_limit"), rerank.get("candidate_limit"), DEFAULT_CANDIDATE_LIMIT),
        minimum=0,
    )
    top_k = _integer_setting(
        "retrieval.top_k",
        _first_defined(args.top_k, benchmark.get("top_k"), retrieval.get("top_k"), DEFAULT_TOP_K),
    )
    fusion_method = _string_setting(
        "retrieval.fusion_method",
        _first_defined(
            args.fusion_method,
            benchmark.get("fusion_method"),
            retrieval.get("fusion_method"),
            DEFAULT_FUSION_METHOD,
        ),
    ).lower()
    if fusion_method not in {"round_robin", "score"}:
        raise ValueError("configuration value 'retrieval.fusion_method' must be one of: round_robin, score")

    return BenchmarkConfig(
        input_path=_string_setting(
            "benchmark.input_path",
            _first_defined(args.input, benchmark.get("input_path"), generation.get("input_path")),
        ),
        output_path=_first_defined(args.output, benchmark.get("output_path"), DEFAULT_OUTPUT_PATH),
        test_num=test_num,
        llm=_resolve_llm_config(args, config),
        method=method,
        prompt_file=prompt_file,
        retrieval_url=_string_setting(
            "retrieval.url",
            _first_defined(
                args.retrieval_url,
                benchmark.get("retrieval_url"),
                retrieval.get("url"),
                DEFAULT_RETRIEVAL_URL,
            ),
        ),
        retrieval_timeout=_positive_number_setting(
            "retrieval.timeout",
            _first_defined(args.timeout, benchmark.get("timeout"), retrieval.get("timeout"), 30.0),
        ),
        fusion_method=fusion_method,
        top_k=top_k,
        candidate_limit=candidate_limit,
    )


def _print_progress(completed: int, total: int, retrieval_successes: int, rerank_successes: int) -> None:
    percentage = 100.0 if total == 0 else completed / total * 100
    print(
        f"\r[benchmark] {completed}/{total} ({percentage:5.1f}%) "
        f"retrieval_success={retrieval_successes} rerank_success={rerank_successes}",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if completed >= total:
        print(file=sys.stderr)


def _latency_summary(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    durations = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float))
        and not isinstance(record.get(field), bool)
    ]
    total = sum(durations)
    return {
        "completed_samples": len(durations),
        "total_seconds": total,
        "average_seconds": total / len(durations) if durations else 0.0,
    }


class SerialLatencyBenchmark:
    """Run every sample and every stage serially while retaining full traces."""

    def __init__(
        self,
        generator: Any,
        retrieval_evaluator: RetrievalEvaluator,
        reranker: CaseReranker,
        samples: Sequence[Any],
    ) -> None:
        self._query_runner = QueryGenerationRunner(generator)
        self._retrieval_evaluator = retrieval_evaluator
        self._reranker = reranker
        self._samples = samples

    def run(self, *, progress: bool = False) -> list[dict[str, Any]]:
        total = len(self._samples)
        retrieval_successes = 0
        rerank_successes = 0
        output: list[dict[str, Any]] = []

        for position, sample in enumerate(self._samples):
            started = time.perf_counter()
            query_record = self._query_runner.generate_sample(sample)
            retrieval_record = self._retrieval_evaluator.evaluate_query(query_record)
            retrieval_finished = time.perf_counter()

            if query_record.status == "success":
                retrieval_record["time_to_retrieval_sec"] = retrieval_finished - started
                retrieval_successes += retrieval_record.get("status") == "success"

            reranked_record = self._reranker.rerank_sample(retrieval_record, position)
            rerank_finished = time.perf_counter()
            if retrieval_record.get("retrieval_status") in {"success", "partial_success"}:
                reranked_record["time_to_rerank_sec"] = rerank_finished - started
                reranked_record["rerank_only_time_sec"] = rerank_finished - retrieval_finished
                rerank_successes += reranked_record.get("rerank_status") == "success"

            output.append(reranked_record)
            if progress:
                _print_progress(position + 1, total, retrieval_successes, rerank_successes)

        return output


def build_benchmark_artifact(
    records: Sequence[Mapping[str, Any]],
    *,
    config: BenchmarkConfig,
    requested_test_num: int,
) -> dict[str, Any]:
    actual_test_num = len(records)
    return {
        "schema_version": 1,
        "artifact_type": BENCHMARK_ARTIFACT_TYPE,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "input_path": config.input_path,
            "output_path": config.output_path,
            "test_num_requested": requested_test_num,
            "test_num": actual_test_num,
            "method": config.method,
            "model_name": config.llm.model_name,
            "prompt_file": config.prompt_file,
            "retrieval_url": config.retrieval_url,
            "retrieval_timeout": config.retrieval_timeout,
            "fusion_method": config.fusion_method,
            "top_k": config.top_k,
            "candidate_limit": config.candidate_limit,
            "serial": True,
            "timing_scope": "from before query generation to the corresponding stage return",
        },
        "metrics": {
            "retrieval": calculate_metrics(records),
            "rerank": calculate_rerank_metrics(records),
            "latency": {
                "time_to_retrieval": _latency_summary(records, "time_to_retrieval_sec"),
                "time_to_rerank": _latency_summary(records, "time_to_rerank_sec"),
                "rerank_only": _latency_summary(records, "rerank_only_time_sec"),
            },
        },
        "records": list(records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = resolve_config(args)
        config.llm.validate()
        samples = load_dialogue_samples(config.input_path)
        selected_samples = samples if config.test_num == 0 else samples[: config.test_num]

        generator = create_query_generator(
            config.llm,
            config.method,
            prompt_file=config.prompt_file,
        )
        retrieval_evaluator = RetrievalEvaluator(
            SearchRetriever(config.retrieval_url, config.retrieval_timeout),
            fusion_method=config.fusion_method,
            top_k=config.top_k,
            parallel_queries=False,
        )
        rerank_config = RerankConfig(
            input_path=config.input_path,
            output_path=config.output_path,
            llm=config.llm,
            concurrency=1,
            candidate_limit=config.candidate_limit,
        )
        reranker = CaseReranker(build_openai_client(config.llm), rerank_config)
        records = SerialLatencyBenchmark(
            generator,
            retrieval_evaluator,
            reranker,
            selected_samples,
        ).run(progress=True)
        artifact = build_benchmark_artifact(
            records,
            config=config,
            requested_test_num=config.test_num,
        )
        write_json_atomically(artifact, config.output_path)

        retrieval = artifact["metrics"]["retrieval"]
        rerank = artifact["metrics"]["rerank"]
        latency = artifact["metrics"]["latency"]
        print(
            "Serial benchmark complete: "
            f"total={len(records)} "
            f"retrieval_R@10={retrieval['recall_at_10']:.4f} "
            f"rerank_R@10={rerank['recall_at_10']:.4f} "
            f"avg_to_retrieval_ms={latency['time_to_retrieval']['average_seconds'] * 1000:.2f} "
            f"avg_to_rerank_ms={latency['time_to_rerank']['average_seconds'] * 1000:.2f} "
            f"output={config.output_path}"
        )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Serial benchmark failed to start or save results: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
