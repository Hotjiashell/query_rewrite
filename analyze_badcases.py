"""Use an LLM to find top-K cases that describe the same solution as each GT case.

The first model call sees the GT title/content and retrieved titles and returns
candidate ranks. The second call sees the full contents of those candidates and
returns a decision and reason for each one. Results are checkpointed after every
record so an interrupted run can be resumed safely.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from gen_query import LLMConfig, _call_model, build_openai_client


def _json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _extract(text: str) -> Any:
    candidates = re.findall(r"```(?:json)?\s*(.*?)\s*```", text or "", re.I | re.S)
    candidates.append((text or "").strip())
    for item in candidates:
        try:
            return json.loads(item)
        except json.JSONDecodeError:
            continue
    raise ValueError("model response is not valid JSON")


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _stage1_prompt(chat_content: str, gt_title: str, gt_text: str, candidates: list[Mapping[str, Any]]) -> str:
    titles = "\n".join(f"{x['rank']}. {x['title']}" for x in candidates)
    return f"""你是客服案例检索检查员。任务背景是：用户与客服进行了一段对话，客服要从历史案例库中找到能够直接或间接解决用户问题的案例。
GT案例是标注的正确案例，候选案例中也可能存在能够解决该用户问题的案例。

用户与客服对话：
{chat_content}

GT案例标题和内容如下：
标题：{gt_title}\n内容：{gt_text}

召回的候选案例标题（数字是rank）：
{titles}

请先分析GT和各候选标题的关系，说明哪些候选可能像GT案例一样，也能解决用户问题、并说明筛选依据。
分析完成后，在最后用一个```json```代码块给出最终结果，格式必须是：{{\"ranks\":[1,2],\"reason\":\"简要说明筛选依据\"}}。ranks必须是上面出现的数字。
"""


def _stage2_prompt(chat_content: str, gt_title: str, gt_text: str, candidates: list[Mapping[str, Any]]) -> str:
    cases = "\n\n".join(
        f"候选{index}（rank={candidate['rank']}，case_id={candidate['case_id']}）：\n标题：{candidate['title']}\n内容：{candidate['text']}"
        for index, candidate in enumerate(candidates, 1)
    )
    return f"""你是客服案例检索检查员。用户与客服进行了下面的对话，客服需要从历史案例库中找到能够直接或间接解决用户问题的案例。GT案例是标注的正确案例。请判断每个候选案例是否也能够解决这位用户的具体问题。
same=true表示候选案例和GT案例类似，也能直接或间接解决用户问题。same=false表示无法解决这段对话中的具体诉求。

用户与客服对话：
{chat_content}

GT案例：\n标题：{gt_title}\n内容：{gt_text}

{cases}

请逐个分析所有候选，说明问题、关键处理步骤和结论是否一致。全部分析完成后，在最后用一个```json```代码块给出最终结果，格式必须是：{{\"decisions\":[{{\"rank\":1,\"same\":true,\"reason\":\"具体中文理由\"}}]}}。每个候选必须有且只有一条decision，rank必须对应候选的rank。
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyze badcases with two-stage parallel LLM calls")
    p.add_argument("badcases", help="collect_badcases.py output JSON")
    p.add_argument("case_summary", help="case_id -> {case_name,text} JSON")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="config.json", help="LLM configuration JSON (default: config.json)")
    p.add_argument("--base-url", default=None, help="Override llm.base_url from config")
    p.add_argument("--model", default=None, help="Override llm.model_name from config")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-key-env", default=None, help="Override llm.api_key_env from config")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args(argv)
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    report = _json(args.badcases)
    summary = _json(args.case_summary)
    if not isinstance(report, Mapping) or not isinstance(report.get("records"), list):
        raise ValueError("badcases must contain a records array")
    if not isinstance(summary, Mapping):
        raise ValueError("case_summary root must be an object")
    llm_config = _json(args.config)
    if not isinstance(llm_config, Mapping) or not isinstance(llm_config.get("llm"), Mapping):
        raise ValueError(f"{args.config} must contain an llm object")
    llm_values = llm_config["llm"]
    base_url = args.base_url or llm_values.get("base_url")
    model_name = args.model or llm_values.get("model_name")
    api_key_env = args.api_key_env or llm_values.get("api_key_env") or "OPENAI_API_KEY"
    api_key = args.api_key if args.api_key is not None else llm_values.get("api_key")
    out = Path(args.output)
    existing = _json(out) if out.is_file() else {}
    done = {str(x.get("record_key")): x for x in existing.get("records", []) if isinstance(x, Mapping)}
    lock = threading.Lock()
    config = LLMConfig(str(base_url or ""), str(model_name or ""), str(api_key or __import__("os").environ.get(api_key_env, "")), args.temperature)
    config.validate()

    def process(index: int, raw: Mapping[str, Any]) -> dict[str, Any]:
        key = str(raw.get("sample_index", index))
        gt = summary.get(str(raw.get("expected_case_id")), {})
        if not isinstance(gt, Mapping) or not gt.get("text"):
            return {"record_key": key, "status": "error", "error": "GT case content not found"}
        traces = raw.get("retrieval_trace") if isinstance(raw.get("retrieval_trace"), list) else []
        candidates = []
        for t in traces[:10]:
            if not isinstance(t, Mapping): continue
            candidates.append({"rank": t.get("rank"), "case_id": str(t.get("case_id")), "title": str(t.get("case_title") or "")})
        try:
            client = build_openai_client(config)
            first = _extract(_call_model(client, config.model_name, _stage1_prompt(str(raw.get("chat_content") or ""), str(gt.get("case_name") or ""), str(gt.get("text")), candidates), args.temperature))
            ranks = first.get("ranks", []) if isinstance(first, Mapping) else []
            selected = [c for c in candidates if c["rank"] in ranks]
            decisions = []
            if selected:
                enriched = []
                missing = []
                for c in selected:
                    entry = summary.get(c["case_id"])
                    if not isinstance(entry, Mapping) or not entry.get("text"):
                        missing.append({**c, "same": None, "reason": "候选案例内容未找到"})
                    else:
                        enriched.append({**c, "text": str(entry.get("text"))})
                if enriched:
                    answer = _extract(_call_model(client, config.model_name, _stage2_prompt(str(raw.get("chat_content") or ""), str(gt.get("case_name") or ""), str(gt.get("text")), enriched), args.temperature))
                    raw_decisions = answer.get("decisions", []) if isinstance(answer, Mapping) else []
                    by_rank = {item.get("rank"): item for item in raw_decisions if isinstance(item, Mapping)}
                    for c in enriched:
                        item = by_rank.get(c["rank"], {})
                        decisions.append({k: c[k] for k in ("rank", "case_id", "title")} | {"same": item.get("same"), "reason": item.get("reason", "模型未返回该候选的判断")})
                decisions.extend(missing)
            return {"record_key": key, "sample_index": raw.get("sample_index"), "status": "success", "stage1": first, "decisions": decisions}
        except Exception as exc:
            return {"record_key": key, "sample_index": raw.get("sample_index"), "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    records = dict(done)
    pending = [(i, x) for i, x in enumerate(report["records"]) if isinstance(x, Mapping) and str(x.get("sample_index", i)) not in records]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process, i, x) for i, x in pending]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                records[result["record_key"]] = result
                payload = {"schema_version": 1, "artifact_type": "badcase_analysis", "created_at": datetime.now(UTC).isoformat(), "records": list(records.values())}
                _write(out, payload)
            print(f"progress {n}/{len(futures)}", flush=True)
    print(f"done: records={len(records)} output={out}")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr); raise SystemExit(2)
