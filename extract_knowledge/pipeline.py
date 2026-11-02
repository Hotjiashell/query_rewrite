"""Parallel, checkpointed badcase -> query augmentation knowledge pipeline."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import glob
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Atomic replacement, including checkpoints written by independent workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".writing-", delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temporary.replace(path)


def load_prompts(path):
    """Read editable Markdown sections without formatting literal JSON braces."""
    content = Path(path).read_text(encoding="utf-8")
    sections = re.split(r"^## (keywords|filter|extract)\s*$", content, flags=re.M)
    templates = {}
    for stage, body in zip(sections[1::2], sections[2::2]):
        if stage in templates:
            raise ValueError(f"提示词阶段重复：{stage}")
        if body.count("{input}") != 1:
            raise ValueError(f"提示词 {stage} 必须包含且仅包含一个 {{input}} 占位符")
        templates[stage] = body.strip()
    missing = {"keywords", "filter", "extract"} - templates.keys()
    if missing:
        raise ValueError(f"提示词文件缺少阶段：{', '.join(sorted(missing))}")
    return templates


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def canonical(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def text(value):
    return value.strip() if isinstance(value, str) else ""


def diverse(records, limit):
    """Round robin over GT cases; deterministic and never more than limit."""
    buckets = {}
    for row in records:
        buckets.setdefault(row["expected_case_id"], []).append(row)
    result = []
    for i in range(max((len(v) for v in buckets.values()), default=0)):
        for bucket in buckets.values():
            if i < len(bucket):
                result.append(bucket[i])
                if len(result) == limit:
                    return result
    return result


def load_records(paths, cases=None, dialogues=None):
    records, issues, seen = [], [], {}
    for path in paths:
        try:
            payload = read_json(path)
            if not isinstance(payload, dict) or payload.get("artifact_type") != "badcase_report" or not isinstance(payload.get("records"), list):
                raise ValueError("需要 collect_badcases.py 生成的 badcase_report / records")
        except (OSError, ValueError) as exc:
            issues.append({"source": str(path), "error": str(exc)})
            continue
        for index, raw in enumerate(payload["records"]):
            source = {"file": str(path), "record_index": index}
            try:
                if not isinstance(raw, dict):
                    raise ValueError("记录必须是对象")
                case_id = str(raw.get("expected_case_id") or "").strip()
                dialogue = text(raw.get("chat_content"))
                if not dialogue and dialogues:
                    dialogue = dialogues.get(str(raw.get("call_sno")), "")
                title = text(raw.get("gt_case_title"))
                if not title and cases:
                    entry = cases.get(case_id, {})
                    title = text(entry.get("case_name")) if isinstance(entry, dict) else ""
                if not case_id or not dialogue or not title:
                    raise ValueError("缺少 GT ID、对话或 GT 标题；可用 --cases / --dialogues 补齐")
                queries = list(dict.fromkeys([text(raw.get("query"))] + [text(q) for q in (raw.get("queries") or []) if isinstance(q, str)]))
                queries = [q for q in queries if q]
                # Same dialogue from multiple evaluation runs is counted once;
                # all queries and provenance remain available.
                key = digest([case_id, dialogue])
                if key in seen:
                    old = seen[key]
                    old["sources"].append(source)
                    old["queries"] = list(dict.fromkeys(old["queries"] + queries))
                    continue
                row = {"id": key, "call_sno": raw.get("call_sno"), "expected_case_id": case_id,
                       "gt_case_title": title, "chat_content": dialogue, "queries": queries, "sources": [source]}
                seen[key] = row
                records.append(row)
            except (ValueError, TypeError) as exc:
                issues.append({**source, "error": str(exc)})
    return records, issues


def validate(stage, obj):
    if not isinstance(obj, dict):
        raise ValueError("模型必须输出 JSON 对象")
    if stage == "keywords":
        items = obj.get("keywords")
        if not isinstance(items, list):
            raise ValueError("keywords 必须为数组")
        for item in items:
            if not isinstance(item, dict) or not text(item.get("keyword")) or not text(item.get("reason")):
                raise ValueError("每个关键词必须包含非空 keyword / reason")
    elif stage == "filter":
        if type(obj.get("covered")) is not bool or not text(obj.get("reason")):
            raise ValueError("过滤结果需要 covered 布尔值及 reason")
        if obj["covered"] and not text(obj.get("matched_knowledge")):
            raise ValueError("已覆盖必须引用 matched_knowledge")
    else:
        for name in ("knowledge", "reason"):
            if not text(obj.get(name)):
                raise ValueError(f"抽取缺少非空字符串 {name}")
    return obj


def model_input(stage, data):
    """Whitelist semantic fields at the model boundary; retain metadata locally."""
    def example(row):
        return {name: row[name] for name in ("chat_content", "gt_case_title", "queries")}

    if stage == "keywords":
        return {"records": [example(row) for row in data["records"]]}
    if stage == "filter":
        return {name: data[name] for name in ("keyword", "existing_knowledge")}
    if stage == "extract":
        return {"keyword": data["keyword"], "examples": [example(row) for row in data["examples"]]}
    raise ValueError(f"未知模型阶段：{stage}")


class Model:
    def __init__(self, args, llm):
        from openai import OpenAI
        self.args = args
        self.templates = load_prompts(args.prompts_file)
        self.model = llm["model_name"]
        self.client = OpenAI(base_url=llm["base_url"], api_key=llm["api_key"],
                             timeout=args.timeout, max_retries=0)
        self.identity = {k: llm[k] for k in ("base_url", "model_name", "temperature")}

    def task(self, stage, data):
        data = model_input(stage, data)
        templates = self.templates
        prompt = templates[stage].replace("{input}", json.dumps(data, ensure_ascii=False))
        key = digest([stage, prompt, self.identity, self.args.max_tokens])
        path = Path(self.args.output_dir) / "calls" / stage / f"{key}.json"
        if self.args.resume and path.exists():
            try:
                cached = read_json(path)
                if cached["status"] == "success":
                    validate(stage, cached["result"])
                    return {**cached, "cached": True}
            except (OSError, ValueError, KeyError, TypeError):
                pass
        trace = {"stage": stage, "key": key, "prompt": prompt, "input": data,
                 "configuration": self.identity, "attempts": [], "status": "failed"}
        for attempt in range(self.args.retries + 1):
            raw = None
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=self.identity["temperature"], max_tokens=self.args.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                raw = response.choices[0].message.content
                if response.choices[0].finish_reason == "length":
                    raise ValueError("模型输出被截断，调大 --max-tokens")
                if not isinstance(raw, str):
                    raise ValueError("模型返回空内容")
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
                result = validate(stage, json.loads(content))
                trace["result"] = result
                trace["status"] = "success"
                trace["attempts"].append({"response": raw})
                save(path, trace)
                return trace
            except Exception as exc:
                # Never retain exception messages that may include credentials/HTTP bodies.
                trace["attempts"].append({"response": raw, "error_type": type(exc).__name__,
                                          "error": str(exc) if isinstance(exc, ValueError) else "模型调用失败"})
                save(path, trace)
                if attempt < self.args.retries:
                    time.sleep(min(20, 2 ** attempt) + random.random())
        return trace


def parallel(stage, tasks, model, concurrency, output):
    results = [None] * len(tasks)
    failed = cached = completed = 0
    print(f"[{stage}] 0/{len(tasks)}", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(model.task, stage, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "failed", "error_type": type(exc).__name__, "input": tasks[i]}
            results[i] = result
            completed += 1
            failed += result["status"] != "success"
            cached += bool(result.get("cached"))
            print(f"\r[{stage}] {completed}/{len(tasks)} 成功={completed-failed} 失败={failed} 缓存={cached}",
                  end="\n" if completed == len(tasks) else "", file=sys.stderr, flush=True)
    save(output, results)
    return results


def run(args, llm):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    knowledge = Path(args.knowledge).read_text(encoding="utf-8")
    cases = read_json(args.cases) if args.cases else None
    dialogues = None
    if args.dialogues:
        dialogues = {str(r.get("call_sno")): text(r.get("chat_content")) for r in read_json(args.dialogues) if isinstance(r, dict)}
    paths = sorted({str(Path(p).resolve()) for p in args.badcases})
    records, issues = load_records(paths, cases, dialogues)
    save(out / "01_records.json", records)
    save(out / "input_issues.json", issues)
    if not records:
        raise ValueError("没有有效 badcase，详见 input_issues.json")
    save(out / "existing_knowledge.json", {"source": args.knowledge, "content": knowledge})
    groups = {}
    for row in records:
        groups.setdefault(row["expected_case_id"], []).append(row)
    gt_groups = [{"case_id": k, "count": len(v), "records": v} for k, v in sorted(groups.items())]
    save(out / "02_gt_clusters.json", gt_groups)
    model = Model(args, llm)
    save(out / "prompt_templates.json", model.templates)
    # Split large GT groups instead of silently dropping their evidence.
    tasks = []
    for group in gt_groups:
        for offset in range(0, group["count"], args.keyword_batch_size):
            tasks.append({"case_id": group["case_id"], "records": group["records"][offset:offset+args.keyword_batch_size]})
    keyword_results = parallel("keywords", tasks, model, args.concurrency, out / "03_keyword_results.json")
    clusters = {}
    rejected = []
    for task, response in zip(tasks, keyword_results):
        if response["status"] != "success":
            continue
        for item in response["result"]["keywords"]:
            keyword = text(item["keyword"])
            # Title grounding is enforced outside the model as well.
            matching = [r for r in groups[task["case_id"]] if canonical(keyword) in canonical(r["gt_case_title"])]
            if not matching:
                rejected.append({"keyword": keyword, "case_id": task["case_id"], "reason": "关键词未出现在 GT 标题中"})
                continue
            key = canonical(keyword)
            cluster = clusters.setdefault(key, {"keyword": keyword, "records": {}, "keyword_evidence": []})
            cluster["keyword_evidence"].append({"case_id": task["case_id"], "reason": item["reason"]})
            for row in matching:
                cluster["records"][row["id"]] = row
    ordered = []
    for key, cluster in clusters.items():
        cluster["records"] = list(cluster["records"].values())
        cluster["count"] = len(cluster["records"])
        cluster["gt_case_count"] = len({r["expected_case_id"] for r in cluster["records"]})
        cluster["id"] = digest(key)[:16]
        ordered.append(cluster)
    ordered.sort(key=lambda c: (-c["count"], canonical(c["keyword"])))
    save(out / "04_keyword_clusters.json", ordered)
    save(out / "rejected_keywords.json", rejected)
    filters = parallel("filter", [{"keyword": c["keyword"], "existing_knowledge": knowledge} for c in ordered],
                       model, args.concurrency, out / "05_filter_results.json")
    kept, excluded, unresolved = [], [], []
    for cluster, response in zip(ordered, filters):
        if response["status"] != "success":
            unresolved.append(cluster)
        elif response["result"]["covered"]:
            excluded.append({**cluster, "filter": response["result"]})
        else:
            kept.append(cluster)
    save(out / "06_retained_clusters.json", kept)
    save(out / "covered_clusters.json", excluded)
    save(out / "filter_failed_clusters.json", unresolved)
    extract_tasks = [{"cluster_id": c["id"], "keyword": c["keyword"],
                      "examples": diverse(c["records"], args.max_examples)} for c in kept]
    extracted = parallel("extract", extract_tasks, model, args.concurrency, out / "07_extraction_results.json")
    entries = []
    for cluster, task, response in zip(kept, extract_tasks, extracted):
        if response["status"] == "success":
            result = response["result"]
            entries.append({**result, "keyword": cluster["keyword"], "cluster_id": cluster["id"],
                            "cluster_count": cluster["count"], "gt_case_count": cluster["gt_case_count"],
                            "example_ids": [r["id"] for r in task["examples"]]})
    save(out / "knowledge.json", entries)
    (out / "knowledge.txt").write_text("\n".join(f"{i}. {r['knowledge'].strip()}" for i, r in enumerate(entries, 1)) + "\n", encoding="utf-8")
    failures = sum(r["status"] != "success" for r in keyword_results + filters + extracted)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "input_files": paths,
               "record_count": len(records), "input_issue_count": len(issues), "gt_cluster_count": len(groups),
               "keyword_cluster_count": len(ordered), "covered_cluster_count": len(excluded),
               "retained_cluster_count": len(kept), "knowledge_count": len(entries),
               "failed_model_tasks": failures, "partial": bool(failures or issues),
               "concurrency": args.concurrency, "max_examples": args.max_examples,
               "llm": model.identity}
    save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"知识条目：{out / 'knowledge.txt'}")
    return 1 if summary["partial"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("badcases", nargs="*", help="一个或多个 collect_badcases.py 输出文件（支持 shell 通配符）")
    parser.add_argument("--knowledge", help="只包含已有知识条目的独立文本文件，可为空")
    parser.add_argument("--prompts-file", help="包含三个阶段提示词的 Markdown 文件")
    parser.add_argument("--cases", help="可选的案例标题映射 JSON")
    parser.add_argument("--dialogues", help="可选的原始对话 JSON，按 call_sno 补齐对话")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default=str(HERE / "config.json"), help="配置文件，默认 extract_knowledge/config.json")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--retries", type=int, default=None, help="首次失败后的重试次数")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--keyword-batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, choices=range(1, 11), default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None, help="复用相同输入/提示词/模型配置的成功调用，重试失败调用")
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config)
        if not isinstance(config, dict):
            raise ValueError("配置文件必须是 JSON 对象")
        settings = config.get("extract_knowledge", {})
        if not isinstance(settings, dict):
            raise ValueError("extract_knowledge 配置节必须是对象")
        defaults = {"prompts_file": str(HERE / "prompts.md"), "knowledge": str(HERE / "existing_knowledge.txt"),
                    "cases": None, "dialogues": None, "output_dir": None,
                    "concurrency": 16, "timeout": 120, "retries": 2,
                    "max_tokens": 4096, "keyword_batch_size": 10,
                    "max_examples": 10, "resume": False}
        allowed = set(defaults) | {"badcases"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"未知 extract_knowledge 配置项：{', '.join(sorted(unknown))}")
        for name, default in defaults.items():
            if getattr(args, name) is None:
                value = settings.get(name, default)
                if name in {"prompts_file", "knowledge", "cases", "dialogues", "output_dir"} and value is not None:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{name} 必须是非空路径字符串或 null")
                    value = str(ROOT / value) if not Path(value).is_absolute() else value
                setattr(args, name, value)
        if not args.badcases:
            patterns = settings.get("badcases", [])
            if not isinstance(patterns, list) or any(not isinstance(v, str) or not v.strip() for v in patterns):
                raise ValueError("badcases 必须是非空路径字符串的数组")
            args.badcases = []
            for pattern in patterns:
                absolute = str(ROOT / pattern) if not Path(pattern).is_absolute() else pattern
                matches = sorted(glob.glob(absolute))
                if not matches:
                    raise ValueError(f"badcases 路径或通配符未匹配到文件：{pattern}")
                args.badcases.extend(matches)
        if not args.badcases or not args.output_dir:
            raise ValueError("请在配置文件 extract_knowledge 中填写 badcases 和 output_dir")
        for name in ("concurrency", "retries", "max_tokens", "keyword_batch_size", "max_examples"):
            if type(getattr(args, name)) is not int:
                raise ValueError(f"{name} 必须是整数")
        if not 1 <= args.max_examples <= 10:
            raise ValueError("max_examples 必须在 1～10 之间")
        if type(args.resume) is not bool:
            raise ValueError("resume 必须为布尔值")
        if type(args.timeout) not in (int, float):
            raise ValueError("timeout 必须为数值")
        if min(args.concurrency, args.keyword_batch_size, args.max_tokens) < 1 or args.retries < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("并发/批大小/token/超时必须为正数，重试次数不能为负")
        section = config.get("llm", {})
        llm = {"base_url": args.base_url or section.get("base_url") or os.getenv("OPENAI_BASE_URL"),
               "model_name": args.model or section.get("model_name") or os.getenv("OPENAI_MODEL"),
               "api_key": args.api_key or section.get("api_key") or os.getenv(section.get("api_key_env") or "OPENAI_API_KEY"),
               "temperature": section.get("temperature", 0.0)}
        for key in ("base_url", "model_name", "api_key"):
            if not isinstance(llm[key], str) or not llm[key].strip():
                raise ValueError(f"缺少模型配置：{key}")
        return run(args, llm)
    except KeyboardInterrupt:
        print("\n运行中断。已落盘的调用可通过 --resume 复用。", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
        print(f"启动或保存失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
