"""Run and persist query-rewrite retrieval evaluations.

The default ``config.json`` carries normal run settings. Command-line options
can override individual settings for one-off experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from gen_query import BaselineQueryGenerator, LLMConfig, QueryGenerator
from search import DEFAULT_RETRIEVAL_URL, test_retrieval


METRIC_CUTOFFS = (1, 3, 5, 10)
TOP_KEY_PATTERN = re.compile(r"^top(\d+)$", re.IGNORECASE)


class Retriever(Protocol):
    """Retrieval dependency for the evaluator."""

    def retrieve(self, query: str) -> Mapping[str, Any]:
        """Return the raw retrieval response for a query."""


@dataclass(frozen=True)
class DialogueSample:
    """Minimal data needed for a labeled retrieval evaluation sample."""

    index: int
    call_sno: str | None
    dialogue: str
    expected_case_id: str


@dataclass(frozen=True)
class RetrievedCase:
    """Safe, compact retrieval trace retained in result artifacts."""

    rank: int
    case_id: str
    case_title: str


@dataclass(frozen=True)
class RunConfig:
    """All resolved settings needed to execute one evaluation run."""

    input_path: str
    output_path: str
    llm: LLMConfig
    retrieval_url: str
    retrieval_timeout: float
    concurrency: int


class SearchRetriever:
    """Adapter that makes the existing ``search.py`` endpoint injectable."""

    def __init__(self, url: str = DEFAULT_RETRIEVAL_URL, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout

    def retrieve(self, query: str) -> Mapping[str, Any]:
        return test_retrieval(query, url=self._url, timeout=self._timeout)


def load_dialogue_samples(path: str | Path) -> list[DialogueSample]:
    """Load the documented list-shaped dialogue dataset and validate labels."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"input file does not exist: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"input file is not valid JSON: {input_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("input JSON must be a list of dialogue objects")

    samples: list[DialogueSample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"sample {index} must be a JSON object")
        dialogue = item.get("chat_content")
        case_id = item.get("caseID")
        if not isinstance(dialogue, str) or not dialogue.strip():
            raise ValueError(f"sample {index} has empty or missing 'chat_content'")
        if case_id is None or not str(case_id).strip():
            raise ValueError(f"sample {index} has empty or missing 'caseID'")
        call_sno = item.get("call_sno")
        samples.append(
            DialogueSample(
                index=index,
                call_sno=str(call_sno) if call_sno is not None else None,
                dialogue=dialogue,
                expected_case_id=str(case_id),
            )
        )
    return samples


def extract_retrieval_trace(response: Mapping[str, Any]) -> list[RetrievedCase]:
    """Extract every numbered ``top*`` entry in numeric rank order.

    The service may return 5, 7, 10, or another number of candidates.  This
    walks all keys in ``retrieval_result`` rather than trusting a declared N or
    assuming a fixed response length.  Full case content and scores are not
    written to the artifact by design.
    """

    retrieval_result = response.get("retrieval_result")
    if not isinstance(retrieval_result, Mapping):
        raise ValueError("retrieval response has no object-valued 'retrieval_result'")

    numbered_cases: list[tuple[int, Mapping[str, Any]]] = []
    for key, value in retrieval_result.items():
        match = TOP_KEY_PATTERN.fullmatch(str(key))
        if match and isinstance(value, Mapping):
            numbered_cases.append((int(match.group(1)), value))

    numbered_cases.sort(key=lambda item: item[0])
    trace: list[RetrievedCase] = []
    for rank, case in numbered_cases:
        case_id = case.get("case_id")
        case_title = case.get("case_title")
        if case_id is None:
            continue
        trace.append(
            RetrievedCase(
                rank=rank,
                case_id=str(case_id),
                case_title=str(case_title) if case_title is not None else "",
            )
        )
    return trace


def _matched_rank(expected_case_id: str, trace: Sequence[RetrievedCase]) -> int | None:
    for case in trace:
        if case.case_id == expected_case_id:
            return case.rank
    return None


class Evaluator:
    """Evaluate any query generator against a retrieval service."""

    def __init__(self, generator: QueryGenerator, retriever: Retriever) -> None:
        self._generator = generator
        self._retriever = retriever

    def evaluate_sample(self, sample: DialogueSample) -> dict[str, Any]:
        """Return a persistable record; per-sample failures do not abort a run."""

        record: dict[str, Any] = {
            "sample_index": sample.index,
            "call_sno": sample.call_sno,
            "expected_case_id": sample.expected_case_id,
            "query": None,
            "retrieval_trace": [],
            "matched_rank": None,
            "status": "failed",
            "error": None,
        }
        try:
            query = self._generator.generate(sample.dialogue)
            trace = extract_retrieval_trace(self._retriever.retrieve(query))
            matched_rank = _matched_rank(sample.expected_case_id, trace)
            record.update(
                {
                    "query": query,
                    "retrieval_trace": [asdict(case) for case in trace],
                    "matched_rank": matched_rank,
                    "status": "success",
                }
            )
        except Exception as exc:  # Record API and parsing errors per example.
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    def evaluate(self, samples: Sequence[DialogueSample], concurrency: int = 1) -> list[dict[str, Any]]:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if concurrency == 1:
            return [self.evaluate_sample(sample) for sample in samples]

        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(self.evaluate_sample, sample): sample.index for sample in samples}
            for future in as_completed(futures):
                records.append(future.result())
        return sorted(records, key=lambda record: int(record["sample_index"]))


def calculate_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate Recall@K with every input record as the denominator.

    Failed generation/retrieval attempts count as misses, preventing request
    failures from inflating recall.  ``successful_samples`` exposes the health
    of the run alongside the retrieval metrics.
    """

    materialized = list(records)
    total = len(materialized)
    successful = sum(record.get("status") == "success" for record in materialized)
    metrics: dict[str, Any] = {
        "total_samples": total,
        "successful_samples": successful,
        "failed_samples": total - successful,
    }
    for cutoff in METRIC_CUTOFFS:
        hits = sum(
            record.get("status") == "success"
            and isinstance(record.get("matched_rank"), int)
            and record["matched_rank"] <= cutoff
            for record in materialized
        )
        metrics[f"recall_at_{cutoff}"] = hits / total if total else 0.0
        metrics[f"hits_at_{cutoff}"] = hits
    return metrics


def build_artifact(
    records: Sequence[Mapping[str, Any]],
    *,
    input_path: str | Path,
    model_name: str,
    retrieval_url: str,
    concurrency: int,
) -> dict[str, Any]:
    """Create the complete JSON evaluation artifact without exposing API keys."""

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "generator": "baseline",
            "model_name": model_name,
            "retrieval_url": retrieval_url,
            "concurrency": concurrency,
            "input_path": str(input_path),
        },
        "metrics": calculate_metrics(records),
        "records": list(records),
    }


def write_json_atomically(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a complete artifact, leaving no partial JSON after interruption."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(target)


def load_config_file(path: str | Path) -> Mapping[str, Any]:
    """Read a JSON config file and ensure its root is an object."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"configuration file is not valid JSON: {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("configuration file root must be a JSON object")
    return payload


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration section '{name}' must be an object")
    return section


def _first_defined(*values: Any) -> Any:
    """Return the first value that was explicitly provided and is not empty."""

    for value in values:
        if value is not None and value != "":
            return value
    return None


def _string_setting(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid configuration value: {name}")
    return value.strip()


def _integer_setting(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"configuration value '{name}' must be an integer greater than zero")
    return value


def _positive_number_setting(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"configuration value '{name}' must be a number greater than zero")
    return float(value)


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    """Merge CLI arguments, config JSON, and environment variables.

    Precedence is CLI argument, config-file value, then environment variable.
    The retrieval URL, timeout, and concurrency have safe defaults.
    """

    config = load_config_file(args.config)
    llm_section = _config_section(config, "llm")
    retrieval_section = _config_section(config, "retrieval")
    evaluation_section = _config_section(config, "evaluation")

    configured_key_env = llm_section.get("api_key_env")
    if configured_key_env is not None and (
        not isinstance(configured_key_env, str) or not configured_key_env.strip()
    ):
        raise ValueError("configuration value 'llm.api_key_env' must be a non-empty string")
    configured_key = os.getenv(configured_key_env.strip()) if configured_key_env else None

    llm = LLMConfig(
        base_url=_string_setting(
            "llm.base_url",
            _first_defined(args.base_url, llm_section.get("base_url"), os.getenv("OPENAI_BASE_URL")),
        ),
        model_name=_string_setting(
            "llm.model_name",
            _first_defined(args.model, llm_section.get("model_name"), os.getenv("OPENAI_MODEL")),
        ),
        api_key=_string_setting(
            "llm.api_key",
            _first_defined(args.api_key, llm_section.get("api_key"), configured_key, os.getenv("OPENAI_API_KEY")),
        ),
    )
    return RunConfig(
        input_path=_string_setting(
            "evaluation.input_path",
            _first_defined(args.input, evaluation_section.get("input_path")),
        ),
        output_path=_string_setting(
            "evaluation.output_path",
            _first_defined(args.output, evaluation_section.get("output_path")),
        ),
        llm=llm,
        retrieval_url=_string_setting(
            "retrieval.url",
            _first_defined(args.retrieval_url, retrieval_section.get("url"), DEFAULT_RETRIEVAL_URL),
        ),
        retrieval_timeout=_positive_number_setting(
            "retrieval.timeout",
            _first_defined(args.timeout, retrieval_section.get("timeout"), 30.0),
        ),
        concurrency=_integer_setting(
            "evaluation.concurrency",
            _first_defined(args.concurrency, evaluation_section.get("concurrency"), 1),
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the one-shot query-rewrite baseline.")
    parser.add_argument("--config", default="config.json", help="Run configuration JSON (default: config.json)")
    parser.add_argument("--input", help="Override evaluation.input_path")
    parser.add_argument("--output", help="Override evaluation.output_path")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL (or OPENAI_BASE_URL)")
    parser.add_argument("--model", help="Model name (or OPENAI_MODEL)")
    parser.add_argument("--api-key", help="API key (or OPENAI_API_KEY)")
    parser.add_argument(
        "--retrieval-url",
        help=f"Override retrieval.url (default: {DEFAULT_RETRIEVAL_URL})",
    )
    parser.add_argument("--concurrency", type=int, help="Override evaluation.concurrency")
    parser.add_argument("--timeout", type=float, help="Override retrieval.timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_config = resolve_run_config(args)
        run_config.llm.validate()
        samples = load_dialogue_samples(run_config.input_path)
        evaluator = Evaluator(
            generator=BaselineQueryGenerator.from_config(run_config.llm),
            retriever=SearchRetriever(
                url=run_config.retrieval_url,
                timeout=run_config.retrieval_timeout,
            ),
        )
        records = evaluator.evaluate(samples, concurrency=run_config.concurrency)
        artifact = build_artifact(
            records,
            input_path=run_config.input_path,
            model_name=run_config.llm.model_name,
            retrieval_url=run_config.retrieval_url,
            concurrency=run_config.concurrency,
        )
        write_json_atomically(artifact, run_config.output_path)
    except (ValueError, RuntimeError) as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 2

    metrics = artifact["metrics"]
    print(
        "Evaluation complete: "
        f"total={metrics['total_samples']} success={metrics['successful_samples']} "
        f"R@1={metrics['recall_at_1']:.4f} "
        f"R@3={metrics['recall_at_3']:.4f} "
        f"R@5={metrics['recall_at_5']:.4f} "
        f"R@10={metrics['recall_at_10']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
