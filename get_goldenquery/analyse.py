"""Analyse why generated queries miss cases that golden queries retrieve.

The input query file is the ``generated_queries`` artifact produced by
``evaluate.py generate``. Each ordinary query is retrieved again so its actual
Top-10 titles can be compared against the golden-query artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from evaluate import SearchRetriever, extract_retrieval_trace, load_generated_query_records, write_json_atomically
from gen_query import LLMConfig, build_openai_client


ANALYSIS_ARTIFACT_TYPE = "golden_query_miss_analysis"
GOLDEN_ARTIFACT_TYPE = "golden_query_generation"
DEFAULT_TOP_K = 10

ANALYSIS_PROMPT = """你是企业知识库检索质量分析专家。请分析为什么【普通 query】没有在 Top-{top_k} 召回【GT 案例】，但【golden query】可以召回。

你需要从以下三个方面进行分析
1. 缺少的关键词：同时出现在 golden query 和 GT 案例标题中，但没有出现在普通 query 中，不一定是字面完全匹配，也可以是语义上相近。不要猜测输入中不存在的词，不要列举通用词。
2. 多出的噪声词：一个有业务意义的词或短语，不在 golden query 中，却在普通 query 中出现，并且在普通 query 的 Top-{top_k} 候选标题中至少重复出现 2 次或明显主导这些候选。不要把通用词误判为噪声。
3. 检索不到的理由：用一句简洁中文说明为什么检索不到GT案例。

【GT 案例标题】
{gt_case_title}

【golden query】
{golden_query}

【普通 query】
{ordinary_query}

【普通 query 的真实 Top-{top_k} 候选标题】
{top_titles}

请先分析，然后输出 JSON 文件：
...(分析)
```json
{{
  "reason": "...",
  "missing_keywords": ["..."],
  "noise_keywords": ["..."]
}}
```
"""


class Retriever(Protocol):
    def retrieve(self, query: str) -> Mapping[str, Any]:
        """Return one raw retrieval response."""


@dataclass(frozen=True)
class AnalysisConfig:
    golden_path: str
    query_path: str
    output_path: str
    llm: LLMConfig
    retrieval_url: str
    timeout: float
    top_k: int
    concurrency: int


def _normalise_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_json(path: str | Path, description: str) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} does not exist: {source}") from exc
    except OSError as exc:
        raise ValueError(f"could not read {description}: {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {source}: {exc}") from exc


def load_golden_records(path: str | Path) -> dict[int, Mapping[str, Any]]:
    """Load successful golden-query records keyed by source sample index."""

    payload = _load_json(path, "golden query artifact")
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != GOLDEN_ARTIFACT_TYPE:
        raise ValueError(f"input is not a '{GOLDEN_ARTIFACT_TYPE}' artifact")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("golden query artifact 'records' must be a JSON array")

    records: dict[int, Mapping[str, Any]] = {}
    for item in raw_records:
        if not isinstance(item, Mapping):
            continue
        index = item.get("sample_index")
        if isinstance(index, int) and not isinstance(index, bool):
            records[index] = item
    return records


def _format_top_titles(response: Mapping[str, Any], top_k: int) -> tuple[list[dict[str, Any]], int | None]:
    trace = extract_retrieval_trace(response)
    titles = [
        {"rank": item.rank, "case_id": item.case_id, "case_title": item.case_title}
        for item in trace[:top_k]
    ]
    return titles, None


def _top_titles_for_prompt(top_titles: Sequence[Mapping[str, Any]]) -> str:
    if not top_titles:
        return "（检索服务未返回有效候选）"
    return "\n".join(f"Top {item['rank']}：{item['case_title']}" for item in top_titles)


def _parse_analysis(content: str) -> dict[str, Any]:
    """Parse fenced JSON after the prompt asks the model to analyse first."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned empty content")
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.IGNORECASE | re.DOTALL)
    candidates.append(content.strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        reason = _normalise_string(payload.get("reason"))
        if reason is None:
            continue

        def clean_terms(value: Any) -> list[str] | None:
            if not isinstance(value, list):
                return None
            terms: list[str] = []
            for item in value:
                term = _normalise_string(item)
                if term and term not in terms:
                    terms.append(term)
            return terms

        missing = clean_terms(payload.get("missing_keywords"))
        noise = clean_terms(payload.get("noise_keywords"))
        if missing is not None and noise is not None:
            return {"reason": reason, "missing_keywords": missing, "noise_keywords": noise}
    raise ValueError("model response must contain reason, missing_keywords, and noise_keywords JSON fields")


def _call_model(client: Any, config: LLMConfig, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    try:
        return _parse_analysis(response.choices[0].message.content)
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("model response has no first message content") from exc


class GoldenMissAnalyser:
    def __init__(self, client: Any, config: AnalysisConfig, retriever: Retriever) -> None:
        self._client = client
        self._config = config
        self._retriever = retriever

    def analyse_sample(self, golden: Mapping[str, Any], ordinary: Any) -> dict[str, Any]:
        expected_case_id = _normalise_string(golden.get("expected_case_id"))
        golden_query = _normalise_string(golden.get("final_query"))
        golden_case = golden.get("golden_case")
        gt_case_title = _normalise_string(golden_case.get("title")) if isinstance(golden_case, Mapping) else None
        ordinary_query = _normalise_string(getattr(ordinary, "query", None))
        record: dict[str, Any] = {
            "sample_index": golden.get("sample_index"),
            "call_sno": golden.get("call_sno"),
            "expected_case_id": expected_case_id,
            "gt_case_title": gt_case_title,
            "golden_query": golden_query,
            "golden_matched_rank": golden.get("matched_rank"),
            "ordinary_query": ordinary_query,
            "ordinary_matched_rank": None,
            "ordinary_top_10": [],
            "status": "failed",
            "error": None,
            "analysis": None,
        }
        if not expected_case_id or not golden_query or not gt_case_title:
            record["error"] = "InvalidGoldenRecord: missing expected_case_id, final_query, or GT case title"
            return record
        if getattr(ordinary, "status", None) != "success" or not ordinary_query:
            record["error"] = f"InvalidOrdinaryQuery: {getattr(ordinary, 'error', None) or 'missing or failed query'}"
            return record

        try:
            top_titles, _ = _format_top_titles(self._retriever.retrieve(ordinary_query), self._config.top_k)
            ordinary_rank = next(
                (item["rank"] for item in top_titles if item["case_id"] == expected_case_id), None
            )
            record["ordinary_top_10"] = top_titles
            record["ordinary_matched_rank"] = ordinary_rank
            if ordinary_rank is not None:
                record["status"] = "skipped"
                record["error"] = f"OrdinaryQueryRetrieved: target was retrieved at rank {ordinary_rank}"
                return record

            prompt = ANALYSIS_PROMPT.format(
                top_k=self._config.top_k,
                gt_case_title=gt_case_title,
                golden_query=golden_query,
                ordinary_query=ordinary_query,
                top_titles=_top_titles_for_prompt(top_titles),
            )
            record["analysis"] = _call_model(self._client, self._config.llm, prompt)
            record["status"] = "success"
            return record
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record

    def analyse(
        self,
        golden_records: Mapping[int, Mapping[str, Any]],
        ordinary_records: Sequence[Any],
        *,
        progress: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        ordinary_by_index = {record.sample_index: record for record in ordinary_records}
        candidates: list[tuple[Mapping[str, Any], Any]] = []
        skipped_without_query = 0
        for index, golden in sorted(golden_records.items()):
            if golden.get("status") != "success":
                continue
            ordinary = ordinary_by_index.get(index)
            if ordinary is None:
                skipped_without_query += 1
                continue
            candidates.append((golden, ordinary))

        total = len(candidates)
        completed = successful = failed = 0
        if self._config.concurrency == 1:
            records: list[dict[str, Any]] = []
            for golden, ordinary in candidates:
                record = self.analyse_sample(golden, ordinary)
                records.append(record)
                completed += 1
                successful += record["status"] == "success"
                failed += record["status"] == "failed"
                if progress:
                    _print_progress(completed, total, successful, failed)
        else:
            records = []
            with ThreadPoolExecutor(max_workers=self._config.concurrency) as executor:
                futures = [executor.submit(self.analyse_sample, golden, ordinary) for golden, ordinary in candidates]
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    completed += 1
                    successful += record["status"] == "success"
                    failed += record["status"] == "failed"
                    if progress:
                        _print_progress(completed, total, successful, failed)
            records.sort(key=lambda item: int(item["sample_index"]))

        summary = {
            "golden_successful_samples": sum(record.get("status") == "success" for record in golden_records.values()),
            "candidate_samples": total,
            "analysed_misses": sum(record["status"] == "success" for record in records),
            "ordinary_query_retrieved": sum(record["status"] == "skipped" for record in records),
            "failed_samples": sum(record["status"] == "failed" for record in records),
            "golden_samples_without_ordinary_query": skipped_without_query,
        }
        return records, summary


def _print_progress(completed: int, total: int, successful: int, failed: int) -> None:
    percentage = 100.0 if total == 0 else completed / total * 100
    print(
        f"\r[golden-analysis] {completed}/{total} ({percentage:5.1f}%) "
        f"analysed={successful} failed={failed}",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if completed >= total:
        print(file=sys.stderr)


def _read_config(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    payload = _load_json(path, "configuration file")
    if not isinstance(payload, Mapping):
        raise ValueError("configuration file root must be an object")
    return payload


def _setting(value: Any, fallback: Any, name: str) -> Any:
    result = value if value is not None else fallback
    if result is None or (isinstance(result, str) and not result.strip()):
        raise ValueError(f"missing configuration value: {name}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse ordinary query misses against successful golden queries.")
    parser.add_argument("--config", help="JSON config with llm and golden_query_analysis sections")
    parser.add_argument("--golden", help="golden_query_generation artifact path")
    parser.add_argument("--queries", help="generated_queries artifact path from evaluate.py generate")
    parser.add_argument("--output", help="analysis artifact output path")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--retrieval-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--top-k", type=int, help=f"ordinary-query retrieval cutoff (default: {DEFAULT_TOP_K})")
    parser.add_argument("--concurrency", type=int)
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> AnalysisConfig:
    config = _read_config(args.config)
    llm_section = config.get("llm", {})
    section = config.get("golden_query_analysis", {})
    if not isinstance(llm_section, Mapping) or not isinstance(section, Mapping):
        raise ValueError("configuration sections 'llm' and 'golden_query_analysis' must be objects")
    key_env = llm_section.get("api_key_env")
    env_key = os.getenv(key_env) if isinstance(key_env, str) and key_env else os.getenv("OPENAI_API_KEY")
    llm = LLMConfig(
        base_url=str(_setting(args.base_url, llm_section.get("base_url") or os.getenv("OPENAI_BASE_URL"), "llm.base_url")),
        model_name=str(_setting(args.model, llm_section.get("model_name") or os.getenv("OPENAI_MODEL"), "llm.model_name")),
        api_key=str(_setting(args.api_key, llm_section.get("api_key") or env_key, "llm.api_key")),
        temperature=float(args.temperature if args.temperature is not None else llm_section.get("temperature", 0.0)),
    )
    resolved = AnalysisConfig(
        golden_path=str(_setting(args.golden, section.get("golden_path"), "golden_query_analysis.golden_path")),
        query_path=str(_setting(args.queries, section.get("query_path"), "golden_query_analysis.query_path")),
        output_path=str(_setting(args.output, section.get("output_path"), "golden_query_analysis.output_path")),
        llm=llm,
        retrieval_url=str(args.retrieval_url or section.get("retrieval_url") or "http://10.67.43.14:8276/run_case_retrieval"),
        timeout=float(args.timeout if args.timeout is not None else section.get("timeout", 30.0)),
        top_k=int(args.top_k if args.top_k is not None else section.get("top_k", DEFAULT_TOP_K)),
        concurrency=int(args.concurrency if args.concurrency is not None else section.get("concurrency", 1)),
    )
    if resolved.top_k < 1 or resolved.timeout <= 0 or resolved.concurrency < 1:
        raise ValueError("top_k, timeout, and concurrency must be greater than zero")
    resolved.llm.validate()
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = resolve_config(parse_args(argv))
        analyser = GoldenMissAnalyser(
            build_openai_client(config.llm), config, SearchRetriever(config.retrieval_url, config.timeout)
        )
        records, summary = analyser.analyse(
            load_golden_records(config.golden_path),
            load_generated_query_records(config.query_path),
            progress=True,
        )
        write_json_atomically(
            {
                "schema_version": 1,
                "artifact_type": ANALYSIS_ARTIFACT_TYPE,
                "created_at": datetime.now(UTC).isoformat(),
                "configuration": {
                    "golden_path": config.golden_path,
                    "query_path": config.query_path,
                    "retrieval_url": config.retrieval_url,
                    "timeout": config.timeout,
                    "top_k": config.top_k,
                    "concurrency": config.concurrency,
                    "model_name": config.llm.model_name,
                },
                "summary": summary,
                "records": records,
            },
            config.output_path,
        )
        print(
            "Golden query miss analysis complete: "
            f"analysed={summary['analysed_misses']} "
            f"ordinary_retrieved={summary['ordinary_query_retrieved']} "
            f"failed={summary['failed_samples']} output={config.output_path}"
        )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Golden query miss analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
