"""Two-stage query rewrite evaluation.

Run the stages independently:

    python evaluate.py generate
    python evaluate.py retrieve

The first command calls the LLM and writes a generated-query artifact. The
second command reads that artifact, calls only the retrieval service, and
writes retrieval traces and Recall@K metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from gen_query import (
    CUSTOM_METHODS,
    SUPPORTED_QUERY_METHODS,
    LLMConfig,
    MultiQueryGenerator,
    QueryGenerator,
    create_query_generator,
    normalize_query_method,
)
from search import DEFAULT_RETRIEVAL_URL, RetrievalRequestError, test_retrieval


METRIC_CUTOFFS = (1, 3, 5, 10)
TOP_KEY_PATTERN = re.compile(r"^top(\d+)$", re.IGNORECASE)
QUERY_ARTIFACT_TYPE = "generated_queries"
RETRIEVAL_ARTIFACT_TYPE = "retrieval_evaluation"


class Retriever(Protocol):
    """Retrieval dependency used by the second stage."""

    def retrieve(self, query: str) -> Mapping[str, Any]:
        """Return the raw retrieval response for one query."""


@dataclass(frozen=True)
class DialogueSample:
    """A source dialogue, including any error isolated during input parsing."""

    index: int
    call_sno: str | None
    dialogue: str | None
    expected_case_id: str | None
    input_error: str | None = None


@dataclass(frozen=True)
class GeneratedQueryRecord:
    """The portable result of query generation for one input sample."""

    sample_index: int
    call_sno: str | None
    expected_case_id: str | None
    query: str | None
    status: str
    error: str | None
    chat_content: str | None = None
    queries: list[str] | None = None


@dataclass(frozen=True)
class RetrievedCase:
    """Safe, compact retrieval trace retained in result artifacts."""

    rank: int
    case_id: str
    case_title: str
    score: float | None = None


@dataclass(frozen=True)
class QueryGenerationConfig:
    """Resolved settings for the LLM-backed query-generation stage."""

    input_path: str
    output_path: str
    llm: LLMConfig
    method: str
    concurrency: int
    prompt_file: str | None = None


@dataclass(frozen=True)
class RetrievalConfig:
    """Resolved settings for the retrieval and metric-calculation stage."""

    input_path: str
    output_path: str
    url: str
    timeout: float
    concurrency: int
    fusion_method: str
    top_k: int


class SearchRetriever:
    """Adapter that makes the existing ``search.py`` endpoint injectable."""

    def __init__(self, url: str = DEFAULT_RETRIEVAL_URL, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout

    def retrieve(self, query: str) -> Mapping[str, Any]:
        return test_retrieval(query, url=self._url, timeout=self._timeout)


def _print_progress(
    stage: str,
    completed: int,
    total: int,
    successful: int,
    failed: int,
) -> None:
    """Render one in-place progress update without adding a dependency on tqdm."""

    percentage = 100.0 if total == 0 else completed / total * 100
    print(
        f"\r[{stage}] {completed}/{total} ({percentage:5.1f}%) "
        f"success={successful} failed={failed}",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if completed >= total:
        print(file=sys.stderr)


def _load_json(path: str | Path, description: str) -> Any:
    source_path = Path(path)
    try:
        return json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} does not exist: {source_path}") from exc
    except OSError as exc:
        raise ValueError(f"could not read {description}: {source_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {source_path}: {exc}") from exc


def _normalise_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalised = str(value).strip()
    return normalised or None


def load_dialogue_samples(path: str | Path) -> list[DialogueSample]:
    """Load source data and isolate malformed entries rather than aborting a run."""

    payload = _load_json(path, "input file")
    if not isinstance(payload, list):
        raise ValueError("input JSON must be a list of dialogue objects")

    samples: list[DialogueSample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            samples.append(
                DialogueSample(
                    index=index,
                    call_sno=None,
                    dialogue=None,
                    expected_case_id=None,
                    input_error="InvalidInput: sample must be a JSON object",
                )
            )
            continue

        dialogue = item.get("chat_content")
        case_id = _normalise_optional_string(item.get("caseID"))
        errors: list[str] = []
        if not isinstance(dialogue, str) or not dialogue.strip():
            dialogue = None
            errors.append("empty or missing 'chat_content'")
        if case_id is None:
            errors.append("empty or missing 'caseID'")
        samples.append(
            DialogueSample(
                index=index,
                call_sno=_normalise_optional_string(item.get("call_sno")),
                dialogue=dialogue,
                expected_case_id=case_id,
                input_error=f"InvalidInput: {'; '.join(errors)}" if errors else None,
            )
        )
    return samples


class QueryGenerationRunner:
    """First-stage runner that produces a query artifact without retrieval."""

    def __init__(self, generator: QueryGenerator) -> None:
        self._generator = generator

    def generate_sample(self, sample: DialogueSample) -> GeneratedQueryRecord:
        if sample.input_error:
            return GeneratedQueryRecord(
                sample_index=sample.index,
                call_sno=sample.call_sno,
                expected_case_id=sample.expected_case_id,
                query=None,
                status="failed",
                error=sample.input_error,
                chat_content=sample.dialogue,
            )

        try:
            queries: list[str] | None = None
            if isinstance(self._generator, MultiQueryGenerator):
                generated = self._generator.generate_queries(sample.dialogue or "")
                if not generated or not isinstance(generated[0], str) or not generated[0].strip():
                    raise ValueError("query generator returned an empty query")
                queries = [item.strip() for item in generated if isinstance(item, str) and item.strip()]
                query = queries[0]
            else:
                query = self._generator.generate(sample.dialogue or "")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query generator returned an empty query")
            return GeneratedQueryRecord(
                sample_index=sample.index,
                call_sno=sample.call_sno,
                expected_case_id=sample.expected_case_id,
                query=query.strip(),
                status="success",
                error=None,
                chat_content=sample.dialogue,
                queries=queries,
            )
        except Exception as exc:  # A model failure must not stop other samples.
            return GeneratedQueryRecord(
                sample_index=sample.index,
                call_sno=sample.call_sno,
                expected_case_id=sample.expected_case_id,
                query=None,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                chat_content=sample.dialogue,
            )

    def generate(
        self,
        samples: Sequence[DialogueSample],
        concurrency: int = 1,
        *,
        progress: bool = False,
    ) -> list[GeneratedQueryRecord]:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        total = len(samples)
        completed = 0
        successful = 0
        failed = 0
        if concurrency == 1:
            records: list[GeneratedQueryRecord] = []
            for sample in samples:
                record = self.generate_sample(sample)
                records.append(record)
                completed += 1
                successful += record.status == "success"
                failed += record.status != "success"
                if progress:
                    _print_progress("generate", completed, total, successful, failed)
            return records

        records: list[GeneratedQueryRecord] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.generate_sample, sample) for sample in samples]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                successful += record.status == "success"
                failed += record.status != "success"
                if progress:
                    _print_progress("generate", completed, total, successful, failed)
        return sorted(records, key=lambda record: record.sample_index)


def calculate_stage_summary(records: Sequence[GeneratedQueryRecord]) -> dict[str, int]:
    """Summarise first-stage records without treating failures as successes."""

    successful = sum(record.status == "success" for record in records)
    return {
        "total_samples": len(records),
        "successful_samples": successful,
        "failed_samples": len(records) - successful,
    }


def build_query_artifact(
    records: Sequence[GeneratedQueryRecord],
    *,
    input_path: str | Path,
    model_name: str,
    concurrency: int,
    method: str = "baseline",
    prompt_file: str | None = None,
) -> dict[str, Any]:
    """Build the first-stage artifact, deliberately excluding dialogue and keys."""

    canonical_method = normalize_query_method(method)
    return {
        "schema_version": 1,
        "artifact_type": QUERY_ARTIFACT_TYPE,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            # ``generator`` is retained for compatibility with earlier query
            # artifacts; ``method`` makes the configured prompt explicit.
            "generator": canonical_method,
            "method": canonical_method,
            "model_name": model_name,
            "input_path": str(input_path),
            "concurrency": concurrency,
            "prompt_file": prompt_file,
        },
        "summary": calculate_stage_summary(records),
        "records": [asdict(record) for record in records],
    }


def load_generated_query_records(path: str | Path) -> list[GeneratedQueryRecord]:
    """Load a query artifact and isolate malformed records for the next stage."""

    payload = _load_json(path, "query artifact")
    if not isinstance(payload, Mapping):
        raise ValueError("query artifact root must be a JSON object")
    if payload.get("artifact_type") != QUERY_ARTIFACT_TYPE:
        raise ValueError(f"input is not a '{QUERY_ARTIFACT_TYPE}' artifact")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("query artifact 'records' must be a JSON array")

    records: list[GeneratedQueryRecord] = []
    for position, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            records.append(
                GeneratedQueryRecord(
                    sample_index=position,
                    call_sno=None,
                    expected_case_id=None,
                    query=None,
                    status="failed",
                    error="InvalidQueryArtifact: record must be a JSON object",
                    chat_content=None,
                )
            )
            continue

        raw_index = item.get("sample_index")
        sample_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else position
        expected_case_id = _normalise_optional_string(item.get("expected_case_id"))
        query = item.get("query")
        query = query.strip() if isinstance(query, str) and query.strip() else None
        raw_queries = item.get("queries")
        queries: list[str] | None = None
        if isinstance(raw_queries, list):
            cleaned = [value.strip() for value in raw_queries if isinstance(value, str) and value.strip()]
            queries = cleaned or None
        source_status = item.get("status")
        source_error = _normalise_optional_string(item.get("error"))
        errors: list[str] = []
        if source_status != "success":
            errors.append(source_error or "query generation did not succeed")
        if query is None:
            errors.append("missing or empty query")
        if expected_case_id is None:
            errors.append("missing expected_case_id")
        records.append(
            GeneratedQueryRecord(
                sample_index=sample_index,
                call_sno=_normalise_optional_string(item.get("call_sno")),
                expected_case_id=expected_case_id,
                query=query,
                status="failed" if errors else "success",
                error="; ".join(dict.fromkeys(errors)) or None,
                chat_content=_normalise_optional_string(item.get("chat_content")),
                queries=queries,
            )
        )
    return records


def extract_retrieval_trace(response: Mapping[str, Any]) -> list[RetrievedCase]:
    """Extract every numbered ``top*`` entry in numeric rank order."""

    retrieval_result = response.get("retrieval_result")
    if not isinstance(retrieval_result, Mapping):
        raise ValueError("retrieval response has no object-valued 'retrieval_result'")

    numbered_cases: list[tuple[int, Mapping[str, Any]]] = []
    for key, value in retrieval_result.items():
        match = TOP_KEY_PATTERN.fullmatch(str(key))
        if match and isinstance(value, Mapping):
            numbered_cases.append((int(match.group(1)), value))

    trace: list[RetrievedCase] = []
    for rank, case in sorted(numbered_cases, key=lambda item: item[0]):
        case_id = case.get("case_id")
        if case_id is None:
            continue
        raw_score = case.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool) else None
        trace.append(
            RetrievedCase(
                rank=rank,
                case_id=str(case_id),
                case_title=str(case.get("case_title") or ""),
                score=score,
            )
        )
    return trace


def _matched_rank(expected_case_id: str, trace: Sequence[RetrievedCase]) -> int | None:
    for case in trace:
        if case.case_id == expected_case_id:
            return case.rank
    return None


def fuse_round_robin(traces: Sequence[Sequence[RetrievedCase]], top_k: int) -> list[RetrievedCase]:
    """Interleave per-query traces in their original rank order, deduplicated by case_id."""

    seen: set[str] = set()
    fused: list[RetrievedCase] = []
    cursors = [0] * len(traces)
    remaining = True
    while remaining and len(fused) < top_k:
        remaining = False
        for trace_index, trace in enumerate(traces):
            cursor = cursors[trace_index]
            while cursor < len(trace):
                case = trace[cursor]
                cursor += 1
                if case.case_id in seen:
                    continue
                seen.add(case.case_id)
                fused.append(case)
                remaining = True
                break
            cursors[trace_index] = cursor
            if len(fused) >= top_k:
                break
    return [replace(case, rank=index + 1) for index, case in enumerate(fused[:top_k])]


def fuse_by_score(traces: Sequence[Sequence[RetrievedCase]], top_k: int) -> list[RetrievedCase]:
    """Deduplicate by case_id keeping the highest score, then sort by score descending."""

    best: dict[str, RetrievedCase] = {}
    for trace in traces:
        for case in trace:
            current = best.get(case.case_id)
            if current is None or (case.score or float("-inf")) > (current.score or float("-inf")):
                best[case.case_id] = case
    ordered = sorted(best.values(), key=lambda case: case.score if case.score is not None else float("-inf"), reverse=True)
    return [replace(case, rank=index + 1) for index, case in enumerate(ordered[:top_k])]


FUSION_METHODS = {
    "round_robin": fuse_round_robin,
    "score": fuse_by_score,
}
DEFAULT_FUSION_METHOD = "round_robin"
DEFAULT_TOP_K = 10


class RetrievalEvaluator:
    """Second-stage runner that consumes generated queries without using an LLM."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        fusion_method: str = DEFAULT_FUSION_METHOD,
        top_k: int = DEFAULT_TOP_K,
        parallel_queries: bool = True,
    ) -> None:
        if fusion_method not in FUSION_METHODS:
            supported = ", ".join(sorted(FUSION_METHODS))
            raise ValueError(f"unsupported fusion method '{fusion_method}'; supported methods: {supported}")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._retriever = retriever
        self._fusion_method = fusion_method
        self._top_k = top_k
        self._parallel_queries = parallel_queries

    def _retrieve_many(self, queries: Sequence[str]) -> tuple[list[list[RetrievedCase]], list[str]]:
        """Retrieve every query for one sample, tolerating partial failure."""

        if not self._parallel_queries:
            traces: list[list[RetrievedCase]] = []
            errors: list[str] = []
            for query in queries:
                try:
                    traces.append(extract_retrieval_trace(self._retriever.retrieve(query)))
                except Exception as exc:
                    errors.append(f"{query!r}: {type(exc).__name__}: {exc}")
            return traces, errors

        results: list[list[RetrievedCase] | None] = [None] * len(queries)
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            future_to_index = {
                executor.submit(self._retriever.retrieve, query): index for index, query in enumerate(queries)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = extract_retrieval_trace(future.result())
                except Exception as exc:
                    errors.append(f"{queries[index]!r}: {type(exc).__name__}: {exc}")
        traces = [trace for trace in results if trace is not None]
        return traces, errors

    def evaluate_query(self, query_record: GeneratedQueryRecord) -> dict[str, Any]:
        record: dict[str, Any] = {
            "sample_index": query_record.sample_index,
            "call_sno": query_record.call_sno,
            "expected_case_id": query_record.expected_case_id,
            "chat_content": query_record.chat_content,
            "gt_case_title": None,
            "query": query_record.query,
            "query_status": query_record.status,
            "query_error": query_record.error,
            "retrieval_status": "skipped",
            "retrieval_error": None,
            "retrieval_trace": [],
            "matched_rank": None,
            "status": "failed",
            "error": query_record.error,
        }
        if query_record.status != "success" or not query_record.query or not query_record.expected_case_id:
            record["retrieval_error"] = "not attempted because query generation did not succeed"
            return record

        queries = query_record.queries
        try:
            if queries and len(queries) > 1:
                traces, retrieval_errors = self._retrieve_many(queries)
                if not traces:
                    raise RetrievalRequestError(f"all queries failed: {'; '.join(retrieval_errors)}")
                trace = FUSION_METHODS[self._fusion_method](traces, self._top_k)
                retrieval_status = "partial_success" if retrieval_errors else "success"
                retrieval_error = "; ".join(retrieval_errors) or None
            else:
                trace = extract_retrieval_trace(self._retriever.retrieve(query_record.query))
                retrieval_status = "success"
                retrieval_error = None
            gt_case_title = next(
                (case.case_title for case in trace if case.case_id == query_record.expected_case_id),
                None,
            )
            record.update(
                {
                    "retrieval_status": retrieval_status,
                    "retrieval_error": retrieval_error,
                    "retrieval_trace": [asdict(case) for case in trace],
                    "matched_rank": _matched_rank(query_record.expected_case_id, trace),
                    "gt_case_title": gt_case_title,
                    "status": "success",
                    "error": None,
                }
            )
        except Exception as exc:  # One retrieval failure must not stop other queries.
            error = f"{type(exc).__name__}: {exc}"
            record.update(
                {
                    "retrieval_status": "failed",
                    "retrieval_error": error,
                    "error": error,
                }
            )
        return record

    def evaluate(
        self,
        query_records: Sequence[GeneratedQueryRecord],
        concurrency: int = 1,
        *,
        progress: bool = False,
    ) -> list[dict[str, Any]]:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        total = len(query_records)
        completed = 0
        successful = 0
        failed = 0
        if concurrency == 1:
            records: list[dict[str, Any]] = []
            for query_record in query_records:
                record = self.evaluate_query(query_record)
                records.append(record)
                completed += 1
                successful += record.get("status") == "success"
                failed += record.get("status") != "success"
                if progress:
                    _print_progress("retrieve", completed, total, successful, failed)
            return records

        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.evaluate_query, record) for record in query_records]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                successful += record.get("status") == "success"
                failed += record.get("status") != "success"
                if progress:
                    _print_progress("retrieve", completed, total, successful, failed)
        return sorted(records, key=lambda record: int(record["sample_index"]))


class Evaluator:
    """Compatibility helper for in-process callers that want both stages.

    CLI use should prefer the separate ``generate`` and ``retrieve`` commands,
    which persist the query artifact between the two API-bound operations.
    """

    def __init__(
        self,
        generator: QueryGenerator,
        retriever: Retriever,
        *,
        fusion_method: str = DEFAULT_FUSION_METHOD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._query_runner = QueryGenerationRunner(generator)
        self._retrieval_evaluator = RetrievalEvaluator(retriever, fusion_method=fusion_method, top_k=top_k)

    def evaluate_sample(self, sample: DialogueSample) -> dict[str, Any]:
        return self._retrieval_evaluator.evaluate_query(self._query_runner.generate_sample(sample))

    def evaluate(self, samples: Sequence[DialogueSample], concurrency: int = 1) -> list[dict[str, Any]]:
        generated_queries = self._query_runner.generate(samples, concurrency=concurrency)
        return self._retrieval_evaluator.evaluate(generated_queries, concurrency=concurrency)


def calculate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate Recall@K using every query-artifact record as the denominator."""

    total = len(records)
    successful = sum(record.get("status") == "success" for record in records)
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
            for record in records
        )
        metrics[f"recall_at_{cutoff}"] = hits / total if total else 0.0
        metrics[f"hits_at_{cutoff}"] = hits
    return metrics


def build_retrieval_artifact(
    records: Sequence[Mapping[str, Any]],
    *,
    input_path: str | Path,
    retrieval_url: str,
    timeout: float,
    concurrency: int,
    fusion_method: str = DEFAULT_FUSION_METHOD,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Build the final artifact containing traces and recall metrics."""

    return {
        "schema_version": 1,
        "artifact_type": RETRIEVAL_ARTIFACT_TYPE,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "source_query_path": str(input_path),
            "retrieval_url": retrieval_url,
            "timeout": timeout,
            "concurrency": concurrency,
            "fusion_method": fusion_method,
            "top_k": top_k,
        },
        "metrics": calculate_metrics(records),
        "records": list(records),
    }


def write_json_atomically(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a complete artifact, leaving no partial JSON after interruption."""

    target = Path(output_path)
    try:
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
    except OSError as exc:
        raise RuntimeError(f"could not write result file: {target}: {exc}") from exc


def load_config_file(path: str | Path) -> Mapping[str, Any]:
    """Read a JSON configuration file and ensure its root is an object."""

    payload = _load_json(path, "configuration file")
    if not isinstance(payload, Mapping):
        raise ValueError("configuration file root must be a JSON object")
    return payload


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


def _integer_setting(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"configuration value '{name}' must be an integer greater than zero")
    return value


def _positive_number_setting(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"configuration value '{name}' must be a number greater than zero")
    return float(value)


def _temperature_setting(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2:
        raise ValueError(f"configuration value '{name}' must be between 0 and 2")
    return float(value)


def _fusion_method_setting(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration value '{name}' must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in FUSION_METHODS:
        supported = ", ".join(sorted(FUSION_METHODS))
        raise ValueError(f"configuration value '{name}' must be one of: {supported}")
    return normalized


def _resolve_llm_config(args: argparse.Namespace, config: Mapping[str, Any]) -> LLMConfig:
    llm_section = _config_section(config, "llm")
    configured_key_env = llm_section.get("api_key_env")
    if configured_key_env is not None and (
        not isinstance(configured_key_env, str) or not configured_key_env.strip()
    ):
        raise ValueError("configuration value 'llm.api_key_env' must be a non-empty string")
    configured_key = os.getenv(configured_key_env.strip()) if configured_key_env else None
    return LLMConfig(
        base_url=_string_setting(
            "llm.base_url",
            _first_defined(args.base_url, llm_section.get("base_url"), os.getenv("OPENAI_BASE_URL")),
        ),
        model_name=_string_setting(
            "llm.model_name",
            _first_defined(args.model, llm_section.get("model_name"), os.getenv("OPENAI_MODEL")),
        ),
        temperature=_temperature_setting(
            "llm.temperature",
            _first_defined(args.temperature, llm_section.get("temperature"), 0.0),
        ),
        api_key=_string_setting(
            "llm.api_key",
            _first_defined(args.api_key, llm_section.get("api_key"), configured_key, os.getenv("OPENAI_API_KEY")),
        ),
    )


def resolve_query_generation_config(args: argparse.Namespace) -> QueryGenerationConfig:
    """Resolve settings for ``generate`` with CLI taking precedence over JSON."""

    config = load_config_file(args.config)
    generation_section = _config_section(config, "query_generation")
    method = normalize_query_method(
        _first_defined(args.method, generation_section.get("method"), "baseline")
    )
    prompt_file = _first_defined(args.prompt_file, generation_section.get("prompt_file"))
    if method in CUSTOM_METHODS and not prompt_file:
        raise ValueError(f"query_generation.prompt_file is required when method is '{method}'")
    return QueryGenerationConfig(
        input_path=_string_setting(
            "query_generation.input_path",
            _first_defined(args.input, generation_section.get("input_path")),
        ),
        output_path=_string_setting(
            "query_generation.output_path",
            _first_defined(args.output, generation_section.get("output_path")),
        ),
        llm=_resolve_llm_config(args, config),
        method=method,
        concurrency=_integer_setting(
            "query_generation.concurrency",
            _first_defined(args.concurrency, generation_section.get("concurrency"), 1),
        ),
        prompt_file=prompt_file,
    )


def resolve_retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    """Resolve settings for ``retrieve`` with CLI taking precedence over JSON."""

    config = load_config_file(args.config)
    retrieval_section = _config_section(config, "retrieval")
    return RetrievalConfig(
        input_path=_string_setting(
            "retrieval.input_path",
            _first_defined(args.input, retrieval_section.get("input_path")),
        ),
        output_path=_string_setting(
            "retrieval.output_path",
            _first_defined(args.output, retrieval_section.get("output_path")),
        ),
        url=_string_setting(
            "retrieval.url",
            _first_defined(args.retrieval_url, retrieval_section.get("url"), DEFAULT_RETRIEVAL_URL),
        ),
        timeout=_positive_number_setting(
            "retrieval.timeout",
            _first_defined(args.timeout, retrieval_section.get("timeout"), 30.0),
        ),
        concurrency=_integer_setting(
            "retrieval.concurrency",
            _first_defined(args.concurrency, retrieval_section.get("concurrency"), 1),
        ),
        fusion_method=_fusion_method_setting(
            "retrieval.fusion_method",
            _first_defined(args.fusion_method, retrieval_section.get("fusion_method"), DEFAULT_FUSION_METHOD),
        ),
        top_k=_integer_setting(
            "retrieval.top_k",
            _first_defined(args.top_k, retrieval_section.get("top_k"), DEFAULT_TOP_K),
        ),
    )


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    forced_stage: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one stage of query-rewrite evaluation.")
    if forced_stage is None:
        parser.add_argument("stage", choices=("generate", "retrieve", "all"), help="Stage to run")
    parser.add_argument("--config", default="config.json", help="Run configuration JSON (default: config.json)")
    parser.add_argument("--input", help="Override the current stage input path")
    parser.add_argument("--output", help="Override the current stage output path")
    parser.add_argument("--concurrency", type=int, help="Override the current stage concurrency")
    parser.add_argument("--query-input", help="Override query_generation.input_path in all mode")
    parser.add_argument("--query-output", help="Override query_generation.output_path in all mode")
    parser.add_argument(
        "--query-concurrency",
        type=int,
        help="Override query_generation.concurrency in all mode",
    )
    parser.add_argument("--retrieval-input", help="Override retrieval.input_path in all mode")
    parser.add_argument("--retrieval-output", help="Override retrieval.output_path in all mode")
    parser.add_argument(
        "--retrieval-concurrency",
        type=int,
        help="Override retrieval.concurrency in all mode",
    )
    parser.add_argument("--base-url", help="Override llm.base_url for the generate stage")
    parser.add_argument("--model", help="Override llm.model_name for the generate stage")
    parser.add_argument("--api-key", help="Override llm.api_key for the generate stage")
    parser.add_argument("--temperature", type=float, help="Override llm.temperature (0 to 2) for the generate stage")
    parser.add_argument(
        "--method",
        type=normalize_query_method,
        choices=SUPPORTED_QUERY_METHODS,
        help=(
            "Override query_generation.method for the generate stage "
            "(baseline, method_v1, multi_query, custom, or custom_multi)"
        ),
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a custom prompt template (required when --method custom or custom_multi is used)",
    )
    parser.add_argument("--retrieval-url", help="Override retrieval.url for the retrieve stage")
    parser.add_argument("--timeout", type=float, help="Override retrieval.timeout for the retrieve stage")
    parser.add_argument(
        "--fusion-method",
        choices=sorted(FUSION_METHODS),
        help="Override retrieval.fusion_method for the retrieve stage (round_robin or score)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override retrieval.top_k for the retrieve stage",
    )
    args = parser.parse_args(argv)
    if forced_stage is not None:
        args.stage = forced_stage
    return args


def _run_generate(args: argparse.Namespace) -> int:
    config = resolve_query_generation_config(args)
    config.llm.validate()
    generator = create_query_generator(config.llm, config.method, prompt_file=config.prompt_file)
    records = QueryGenerationRunner(generator).generate(
        load_dialogue_samples(config.input_path),
        concurrency=config.concurrency,
        progress=True,
    )
    artifact = build_query_artifact(
        records,
        input_path=config.input_path,
        model_name=config.llm.model_name,
        concurrency=config.concurrency,
        method=config.method,
        prompt_file=config.prompt_file,
    )
    write_json_atomically(artifact, config.output_path)
    summary = artifact["summary"]
    print(
        "Query generation complete: "
        f"method={config.method} "
        f"total={summary['total_samples']} success={summary['successful_samples']} "
        f"failed={summary['failed_samples']} output={config.output_path}"
    )
    return 0


def _run_retrieve(args: argparse.Namespace) -> int:
    config = resolve_retrieval_config(args)
    records = RetrievalEvaluator(
        SearchRetriever(config.url, config.timeout),
        fusion_method=config.fusion_method,
        top_k=config.top_k,
    ).evaluate(
        load_generated_query_records(config.input_path),
        concurrency=config.concurrency,
        progress=True,
    )
    artifact = build_retrieval_artifact(
        records,
        input_path=config.input_path,
        retrieval_url=config.url,
        timeout=config.timeout,
        concurrency=config.concurrency,
        fusion_method=config.fusion_method,
        top_k=config.top_k,
    )
    write_json_atomically(artifact, config.output_path)
    metrics = artifact["metrics"]
    print(
        "Retrieval evaluation complete: "
        f"total={metrics['total_samples']} success={metrics['successful_samples']} "
        f"R@1={metrics['recall_at_1']:.4f} "
        f"R@3={metrics['recall_at_3']:.4f} "
        f"R@5={metrics['recall_at_5']:.4f} "
        f"R@10={metrics['recall_at_10']:.4f} output={config.output_path}"
    )
    return 0


def _run_all(args: argparse.Namespace) -> int:
    """Run generation followed immediately by retrieval using both configs."""

    generation_args = argparse.Namespace(**vars(args))
    generation_args.input = args.query_input if args.query_input is not None else args.input
    generation_args.output = args.query_output
    generation_args.concurrency = (
        args.query_concurrency if args.query_concurrency is not None else args.concurrency
    )
    generation_config = resolve_query_generation_config(generation_args)
    generation_config.llm.validate()
    generated_records = QueryGenerationRunner(
        create_query_generator(
            generation_config.llm, generation_config.method, prompt_file=generation_config.prompt_file
        )
    ).generate(
        load_dialogue_samples(generation_config.input_path),
        concurrency=generation_config.concurrency,
        progress=True,
    )
    query_artifact = build_query_artifact(
        generated_records,
        input_path=generation_config.input_path,
        model_name=generation_config.llm.model_name,
        concurrency=generation_config.concurrency,
        method=generation_config.method,
        prompt_file=generation_config.prompt_file,
    )
    write_json_atomically(query_artifact, generation_config.output_path)

    # In all-in-one mode the freshly generated artifact is the source of truth
    # for retrieval, even if retrieval.input_path still points to an older file.
    retrieval_args = argparse.Namespace(**vars(args))
    retrieval_args.input = args.retrieval_input
    retrieval_args.output = args.retrieval_output if args.retrieval_output is not None else args.output
    retrieval_args.concurrency = (
        args.retrieval_concurrency if args.retrieval_concurrency is not None else args.concurrency
    )
    retrieval_config = replace(
        resolve_retrieval_config(retrieval_args),
        input_path=generation_config.output_path,
    )
    retrieval_records = RetrievalEvaluator(
        SearchRetriever(retrieval_config.url, retrieval_config.timeout),
        fusion_method=retrieval_config.fusion_method,
        top_k=retrieval_config.top_k,
    ).evaluate(
        load_generated_query_records(retrieval_config.input_path),
        concurrency=retrieval_config.concurrency,
        progress=True,
    )
    retrieval_artifact = build_retrieval_artifact(
        retrieval_records,
        input_path=retrieval_config.input_path,
        retrieval_url=retrieval_config.url,
        timeout=retrieval_config.timeout,
        concurrency=retrieval_config.concurrency,
        fusion_method=retrieval_config.fusion_method,
        top_k=retrieval_config.top_k,
    )
    write_json_atomically(retrieval_artifact, retrieval_config.output_path)

    generation_summary = query_artifact["summary"]
    retrieval_metrics = retrieval_artifact["metrics"]
    print(
        "All stages complete: "
        f"method={generation_config.method} "
        f"generated={generation_summary['successful_samples']}/{generation_summary['total_samples']} "
        f"retrieved={retrieval_metrics['successful_samples']}/{retrieval_metrics['total_samples']} "
        f"R@1={retrieval_metrics['recall_at_1']:.4f} "
        f"R@3={retrieval_metrics['recall_at_3']:.4f} "
        f"R@5={retrieval_metrics['recall_at_5']:.4f} "
        f"R@10={retrieval_metrics['recall_at_10']:.4f} "
        f"query_output={generation_config.output_path} "
        f"result_output={retrieval_config.output_path}"
    )
    return 0


def main(argv: Sequence[str] | None = None, *, forced_stage: str | None = None) -> int:
    args = parse_args(argv, forced_stage=forced_stage)
    try:
        if args.stage == "generate":
            return _run_generate(args)
        if args.stage == "retrieve":
            return _run_retrieve(args)
        return _run_all(args)
    except (ValueError, RuntimeError) as exc:
        print(f"{args.stage.capitalize()} stage failed to start or save results: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
