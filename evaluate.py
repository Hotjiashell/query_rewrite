"""Run and persist query-rewrite retrieval evaluations.

Example:
    python evaluate.py --input data/dialog_example.json --output results/baseline.json \
      --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" --api-key "$OPENAI_API_KEY"
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


def _env_or_argument(argument: str | None, env_name: str) -> str | None:
    return argument if argument is not None else os.getenv(env_name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the one-shot query-rewrite baseline.")
    parser.add_argument("--input", required=True, help="JSON input following data/dialog_example.json")
    parser.add_argument("--output", required=True, help="Path for the JSON result artifact")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL (or OPENAI_BASE_URL)")
    parser.add_argument("--model", help="Model name (or OPENAI_MODEL)")
    parser.add_argument("--api-key", help="API key (or OPENAI_API_KEY)")
    parser.add_argument(
        "--retrieval-url",
        default=DEFAULT_RETRIEVAL_URL,
        help=f"Case retrieval endpoint (default: {DEFAULT_RETRIEVAL_URL})",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent samples (default: 1)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Retrieval request timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = LLMConfig(
        base_url=_env_or_argument(args.base_url, "OPENAI_BASE_URL") or "",
        model_name=_env_or_argument(args.model, "OPENAI_MODEL") or "",
        api_key=_env_or_argument(args.api_key, "OPENAI_API_KEY") or "",
    )
    try:
        config.validate()
        samples = load_dialogue_samples(args.input)
        evaluator = Evaluator(
            generator=BaselineQueryGenerator.from_config(config),
            retriever=SearchRetriever(url=args.retrieval_url, timeout=args.timeout),
        )
        records = evaluator.evaluate(samples, concurrency=args.concurrency)
        artifact = build_artifact(
            records,
            input_path=args.input,
            model_name=config.model_name,
            retrieval_url=args.retrieval_url,
            concurrency=args.concurrency,
        )
        write_json_atomically(artifact, args.output)
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
