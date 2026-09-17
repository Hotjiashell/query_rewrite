"""Generate a query that retrieves each dialogue's labelled (golden) case.

Example:
    python -m get_goldenquery \
      --dialogues data/dialog_example.json \
      --cases data/case_example.json \
      --output results/golden_queries.json \
      --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" \
      --api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from evaluate import SearchRetriever, extract_retrieval_trace, load_dialogue_samples, write_json_atomically
from gen_query import LLMConfig, build_openai_client, extract_query


ARTIFACT_TYPE = "golden_query_generation"
METRIC_CUTOFFS = (1, 3, 5, 10)
DEFAULT_GOLDEN_TOP_K = 5

INITIAL_PROMPT = """你是企业知识库检索 query 设计专家。
请根据【用户对话】和它应命中的【目标案例】，生成一条最可能让检索系统召回该目标案例的中文检索 query。
query不要提及“案例”“caseID”“目标案例”，不要照抄整段对话，也不要输出解释。

【用户对话】
{dialogue}

【目标案例标题】
{case_title}

只输出 JSON：{{"query": "..."}}"""

RETRY_PROMPT = """你正在为企业知识库检索系统优化 query。当前 query 没有在 Top-{top_k} 中召回目标案例。
请结合用户对话、目标案例和本次真实检索结果，判断遗漏或混淆的关键检索词，并生成一条不同的、更能检索到目标案例的中文 query。
检索不到目的案例一般包括两个原因：
1. query 中缺少了目标案例的关键检索词；
2. query 中包含了与目标案例不相关的干扰词，导致检索结果被干扰。
你可以从上一轮真实检索结果中分析那些词是干扰词，从目标案例标题里分析那些词是关键词。
不要提及“案例”“caseID”“目标案例”。请先分析，再输出结果

【用户对话】
{dialogue}

【目标案例标题】
{case_title}

【上一轮 query】
{previous_query}

【上一轮真实 Top-{top_k} 检索结果】
{retrieval_results}

先分析，再输出 JSON：
...(分析)
```json
{{"query": "..."}}
```
"""


class Retriever(Protocol):
    def retrieve(self, query: str) -> Mapping[str, Any]:
        """Return one raw retrieval response."""


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    title: str


@dataclass(frozen=True)
class GoldenQueryConfig:
    dialogue_path: str
    case_path: str
    output_path: str
    llm: LLMConfig
    retrieval_url: str
    timeout: float
    max_retries: int
    top_k: int
    concurrency: int


def _normalise_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def load_cases(path: str | Path) -> dict[str, GoldenCase]:
    """Load case IDs and titles, accepting common title-field aliases."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"case file does not exist: {source}") from exc
    except OSError as exc:
        raise ValueError(f"could not read case file: {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"case file is not valid JSON: {source}: {exc}") from exc

    records: list[tuple[Any, Any]]
    if isinstance(payload, Mapping):
        records = list(payload.items())
    elif isinstance(payload, list):
        records = [(item.get("caseID") or item.get("case_id"), item) for item in payload if isinstance(item, Mapping)]
    else:
        raise ValueError("case JSON must be an object keyed by case ID or a list of case objects")

    cases: dict[str, GoldenCase] = {}
    for raw_id, item in records:
        if not isinstance(item, Mapping):
            continue
        case_id = _normalise_string(raw_id or item.get("caseID") or item.get("case_id"))
        title = _normalise_string(item.get("case_name") or item.get("case_title") or item.get("title"))
        if case_id and title:
            cases[case_id] = GoldenCase(case_id, title)
    return cases


def _call_model(client: Any, config: LLMConfig, prompt: str) -> str:
    """Extract the fenced JSON query after the retry prompt's free-form analysis."""

    response = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    try:
        return extract_query(response.choices[0].message.content)
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("model response has no first message content") from exc


def _format_trace(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Preserve all returned ranks so final Recall@K is not capped by feedback Top-K."""

    trace = extract_retrieval_trace(response)
    records: list[dict[str, Any]] = []
    for item in trace:
        records.append({
            "rank": item.rank,
            "case_id": item.case_id,
            "case_title": item.case_title,
        })
    return records


def _trace_for_prompt(trace: Sequence[Mapping[str, Any]]) -> str:
    if not trace:
        return "（检索服务未返回有效结果）"
    return "\n\n".join(
        f"Top {item['rank']}\n标题：{item['case_title']}"
        for item in trace
    )


class GoldenQueryRunner:
    def __init__(self, client: Any, config: GoldenQueryConfig, retriever: Retriever) -> None:
        self._client = client
        self._config = config
        self._retriever = retriever

    def run_sample(self, sample: Any, case: GoldenCase | None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "sample_index": sample.index,
            "call_sno": sample.call_sno,
            "expected_case_id": sample.expected_case_id,
            "golden_case": asdict(case) if case else None,
            "status": "failed",
            "error": sample.input_error,
            "final_query": None,
            "matched_rank": None,
            "attempts": [],
        }
        if sample.input_error:
            return record
        if case is None:
            record["error"] = f"MissingGoldenCase: no case title for {sample.expected_case_id!r}"
            return record

        query: str | None = None
        previous_trace: list[dict[str, Any]] = []
        for attempt_number in range(self._config.max_retries + 1):
            try:
                if attempt_number == 0:
                    prompt = INITIAL_PROMPT.format(
                        dialogue=sample.dialogue, case_title=case.title
                    )
                else:
                    prompt = RETRY_PROMPT.format(
                        dialogue=sample.dialogue,
                        case_title=case.title,
                        previous_query=query,
                        top_k=self._config.top_k,
                        retrieval_results=_trace_for_prompt(previous_trace),
                    )
                query = _call_model(self._client, self._config.llm, prompt)
                response = self._retriever.retrieve(query)
                full_trace = _format_trace(response)
                feedback_trace = full_trace[: self._config.top_k]
                matched_rank = next((item["rank"] for item in full_trace if item["case_id"] == case.case_id), None)
                passed = matched_rank is not None and matched_rank <= self._config.top_k
                record["attempts"].append({
                    "attempt": attempt_number + 1,
                    "query": query,
                    "retrieval_trace": feedback_trace,
                    "retrieval_matched_rank": matched_rank,
                    "matched_rank": matched_rank,
                    "error": None,
                })
                record["final_query"] = query
                record["matched_rank"] = matched_rank
                if passed:
                    record["status"] = "success"
                    record["error"] = None
                    return record
                previous_trace = feedback_trace
            except Exception as exc:
                record["attempts"].append({
                    "attempt": attempt_number + 1,
                    "query": query,
                    "retrieval_trace": [],
                    "matched_rank": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                record["error"] = record["attempts"][-1]["error"]
                return record
        record["error"] = f"NotRetrieved: target case was absent from Top-{self._config.top_k} after {self._config.max_retries + 1} attempts"
        return record

    def run(
        self,
        samples: Sequence[Any],
        cases: Mapping[str, GoldenCase],
        *,
        progress: bool = False,
    ) -> list[dict[str, Any]]:
        if self._config.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        total = len(samples)
        completed = 0
        successful = 0
        failed = 0
        if self._config.concurrency == 1:
            records: list[dict[str, Any]] = []
            for sample in samples:
                record = self.run_sample(sample, cases.get(sample.expected_case_id or ""))
                records.append(record)
                completed += 1
                successful += record["status"] == "success"
                failed += record["status"] != "success"
                if progress:
                    _print_progress(completed, total, successful, failed)
            return records
        with ThreadPoolExecutor(max_workers=self._config.concurrency) as executor:
            futures = [executor.submit(self.run_sample, sample, cases.get(sample.expected_case_id or "")) for sample in samples]
            records: list[dict[str, Any]] = []
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                successful += record["status"] == "success"
                failed += record["status"] != "success"
                if progress:
                    _print_progress(completed, total, successful, failed)
            return sorted(records, key=lambda item: item["sample_index"])


def _print_progress(completed: int, total: int, successful: int, failed: int) -> None:
    """Show batch progress on stderr without adding a progress-bar dependency."""

    percentage = 100.0 if total == 0 else completed / total * 100
    print(
        f"\r[golden-query] {completed}/{total} ({percentage:5.1f}%) "
        f"hit_top_k={successful} failed={failed}",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if completed >= total:
        print(file=sys.stderr)


def calculate_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Calculate final-query Recall@1/3/5/10 over every input sample."""

    total = len(records)
    successful = sum(record.get("status") == "success" for record in records)
    summary: dict[str, int | float] = {
        "total_samples": total,
        "successful_samples": successful,
        "failed_samples": total - successful,
    }
    for cutoff in METRIC_CUTOFFS:
        hits = sum(
            isinstance(record.get("matched_rank"), int) and record["matched_rank"] <= cutoff
            for record in records
        )
        summary[f"hits_at_{cutoff}"] = hits
        summary[f"recall_at_{cutoff}"] = hits / total if total else 0.0
    return summary


def _read_config(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("configuration file root must be an object")
    return payload


def _setting(value: Any, fallback: Any, name: str) -> Any:
    result = value if value is not None else fallback
    if result is None or (isinstance(result, str) and not result.strip()):
        raise ValueError(f"missing configuration value: {name}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate queries that retrieve labelled golden cases.")
    parser.add_argument("--config", help="JSON config with llm and golden_query sections")
    parser.add_argument("--dialogues", help="dialogue JSON, e.g. data/dialog_example.json")
    parser.add_argument("--cases", help="case JSON, e.g. data/case_example.json")
    parser.add_argument("--output", help="result JSON path")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--retrieval-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-retries", type=int, help="number of revisions after the initial query")
    parser.add_argument(
        "--top-k",
        type=int,
        help=f"Top-K used for pass/fail and feedback (default: {DEFAULT_GOLDEN_TOP_K})",
    )
    parser.add_argument("--concurrency", type=int)
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> GoldenQueryConfig:
    config = _read_config(args.config)
    llm_section = config.get("llm", {})
    section = config.get("golden_query", {})
    if not isinstance(llm_section, Mapping) or not isinstance(section, Mapping):
        raise ValueError("configuration sections 'llm' and 'golden_query' must be objects")
    key_env = llm_section.get("api_key_env")
    env_key = os.getenv(key_env) if isinstance(key_env, str) and key_env else os.getenv("OPENAI_API_KEY")
    llm = LLMConfig(
        base_url=str(_setting(args.base_url, llm_section.get("base_url") or os.getenv("OPENAI_BASE_URL"), "llm.base_url")),
        model_name=str(_setting(args.model, llm_section.get("model_name") or os.getenv("OPENAI_MODEL"), "llm.model_name")),
        api_key=str(_setting(args.api_key, llm_section.get("api_key") or env_key, "llm.api_key")),
        temperature=float(args.temperature if args.temperature is not None else llm_section.get("temperature", 0.0)),
    )
    cfg = GoldenQueryConfig(
        dialogue_path=str(_setting(args.dialogues, section.get("dialogue_path"), "golden_query.dialogue_path")),
        case_path=str(_setting(args.cases, section.get("case_path"), "golden_query.case_path")),
        output_path=str(_setting(args.output, section.get("output_path"), "golden_query.output_path")),
        llm=llm,
        retrieval_url=str(args.retrieval_url or section.get("retrieval_url") or "http://10.67.43.14:8276/run_case_retrieval"),
        timeout=float(args.timeout if args.timeout is not None else section.get("timeout", 30.0)),
        max_retries=int(args.max_retries if args.max_retries is not None else section.get("max_retries", 3)),
        top_k=int(args.top_k if args.top_k is not None else section.get("top_k", DEFAULT_GOLDEN_TOP_K)),
        concurrency=int(args.concurrency if args.concurrency is not None else section.get("concurrency", 1)),
    )
    if cfg.max_retries < 0 or cfg.top_k < 1 or cfg.concurrency < 1 or cfg.timeout <= 0:
        raise ValueError("max_retries must be >= 0; top_k, concurrency and timeout must be > 0")
    cfg.llm.validate()
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = resolve_config(parse_args(argv))
        records = GoldenQueryRunner(
            build_openai_client(config.llm), config, SearchRetriever(config.retrieval_url, config.timeout)
        ).run(load_dialogue_samples(config.dialogue_path), load_cases(config.case_path), progress=True)
        summary = calculate_summary(records)
        write_json_atomically({
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": {
                "dialogue_path": config.dialogue_path, "case_path": config.case_path,
                "retrieval_url": config.retrieval_url, "timeout": config.timeout,
                "max_retries": config.max_retries, "top_k": config.top_k,
                "concurrency": config.concurrency, "model_name": config.llm.model_name,
            },
            "summary": summary,
            "records": records,
        }, config.output_path)
        print(
            "Golden query generation complete: "
            f"Top-{config.top_k} hit={summary['successful_samples']}/{summary['total_samples']} "
            f"R@1={summary['recall_at_1']:.4f} "
            f"R@3={summary['recall_at_3']:.4f} "
            f"R@5={summary['recall_at_5']:.4f} "
            f"R@10={summary['recall_at_10']:.4f} "
            f"output={config.output_path}"
        )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Golden query generation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
