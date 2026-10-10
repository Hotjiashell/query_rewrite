"""Evaluate the fixed single-query API in huawei_baseline/api.py."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

# Support both `python -m huawei_baseline.evaluate_baseline` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huawei_baseline import api
from evaluate import (
    QueryGenerationRunner, RetrievalEvaluator, SearchRetriever,
    build_query_artifact, build_retrieval_artifact, load_config_file,
    load_dialogue_samples, load_generated_query_records,
    resolve_retrieval_config, write_json_atomically,
)
from gen_query import QueryGenerator


class HuaweiQueryGenerator(QueryGenerator):
    def generate(self, dialogue: str) -> str:
        response = api.get_query(dialogue)
        if not isinstance(response, Mapping):
            raise ValueError("get_query must return an object with baseline_user_query; "
                             "check the implementation in huawei_baseline/api.py")
        query = response.get("baseline_user_query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("get_query returned an empty or invalid baseline_user_query")
        return query.strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("generate", "retrieve", "all"), default="all")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--input", help="source dialogues; query artifact in retrieve mode")
    parser.add_argument("--output", help="final result; query artifact in generate mode")
    parser.add_argument("--query-output", help="query artifact written in all mode")
    parser.add_argument("--concurrency", type=int, help="sample concurrency (default 1)")
    parser.add_argument("--query-concurrency", type=int, help="generation concurrency override")
    parser.add_argument("--retrieval-concurrency", type=int, help="retrieval concurrency override")
    parser.add_argument("--retrieval-url")
    parser.add_argument("--timeout", type=float, help="retrieval HTTP timeout in seconds")
    return parser.parse_args(argv)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def main(argv=None):
    args = parse_args(argv)
    try:
        config = load_config_file(args.config)
        settings = config.get("huawei_baseline", {})
        query_settings = config.get("query_generation", {})
        if not isinstance(settings, dict) or not isinstance(query_settings, dict):
            raise ValueError("huawei_baseline and query_generation must be objects")
        concurrency = _positive_int(
            args.concurrency if args.concurrency is not None else settings.get("concurrency", 1), "concurrency")
        query_concurrency = _positive_int(
            args.query_concurrency if args.query_concurrency is not None else concurrency, "query concurrency")
        retrieval_concurrency = _positive_int(
            args.retrieval_concurrency if args.retrieval_concurrency is not None else concurrency, "retrieval concurrency")
        query_path = args.query_output or settings.get("query_output_path", "huawei_baseline/results/queries.json")
        result_path = args.output or settings.get("output_path", "huawei_baseline/results/evaluation.json")
        if args.stage == "generate":
            query_path = args.output or query_path
        if args.stage == "retrieve":
            input_path = args.input or query_path
            outputs = [result_path]
        else:
            input_path = args.input or settings.get("input_path") or query_settings.get("input_path")
            outputs = [query_path] + ([result_path] if args.stage == "all" else [])
        if not isinstance(input_path, str) or not input_path.strip():
            raise ValueError("specify --input or huawei_baseline.input_path")
        paths = [Path(p).resolve() for p in [input_path, *outputs]]
        if len(set(paths)) != len(paths):
            raise ValueError("input, query output and result output must use distinct paths")

        retrieval = None
        if args.stage != "generate":
            retrieval_args = argparse.Namespace(
                config=args.config, input=input_path if args.stage == "retrieve" else query_path,
                output=result_path, retrieval_url=args.retrieval_url, timeout=args.timeout,
                concurrency=retrieval_concurrency, fusion_method="round_robin", top_k=10,
            )
            retrieval = resolve_retrieval_config(retrieval_args)

        if args.stage != "retrieve":
            generated = QueryGenerationRunner(HuaweiQueryGenerator()).generate(
                load_dialogue_samples(input_path), concurrency=query_concurrency, progress=True,
            )
            artifact = build_query_artifact(
                generated, input_path=input_path, model_name="huawei_baseline.api.get_query",
                concurrency=query_concurrency, method="baseline",
            )
            artifact["configuration"]["query_backend"] = "huawei_baseline.api.get_query"
            write_json_atomically(artifact, query_path)
            print(f"Query generation complete: {query_path}")
            if args.stage == "generate":
                print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2))
                return 0
        else:
            generated = load_generated_query_records(input_path)
            if any(record.queries for record in generated):
                raise ValueError("Huawei baseline accepts single-query artifacts only")

        records = RetrievalEvaluator(SearchRetriever(retrieval.url, retrieval.timeout)).evaluate(
            generated, concurrency=retrieval_concurrency, progress=True,
        )
        artifact = build_retrieval_artifact(
            records, input_path=retrieval.input_path, retrieval_url=retrieval.url,
            timeout=retrieval.timeout, concurrency=retrieval_concurrency,
        )
        artifact["configuration"].update(baseline="huawei_baseline", query_backend="huawei_baseline.api.get_query")
        write_json_atomically(artifact, result_path)
        print(json.dumps({"output": result_path, "metrics": artifact["metrics"]}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Huawei baseline evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
