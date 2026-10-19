"""Local badcase annotation app; Python standard library only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_TYPES = ["其他"]


def retrieve_cases(query, *, url=None, timeout=30):
    # Keep the app's existing startup independent of optional requests dependency.
    project_root = str(ROOT.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from search import DEFAULT_RETRIEVAL_URL, test_retrieval
    return test_retrieval(query, url=url or DEFAULT_RETRIEVAL_URL, timeout=timeout)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def build_dataset(badcases, analysis, cases):
    if not isinstance(badcases, dict) or not isinstance(badcases.get("records"), list):
        raise ValueError("badcase文件需要包含records数组")
    if not isinstance(analysis, dict) or not isinstance(analysis.get("records"), list):
        raise ValueError("模型分析文件需要包含records数组")
    if not isinstance(cases, dict):
        raise ValueError("案例库需要是case_id到案例信息的映射")
    models = {}
    for row in analysis["records"]:
        if isinstance(row, dict):
            key = row.get("record_key")
            if key is None:
                key = row.get("sample_index")
            models[str(key)] = row
    result = []
    seen = set()
    for position, row in enumerate(badcases["records"]):
        if not isinstance(row, dict):
            raise ValueError(f"第{position + 1}条badcase不是对象")
        index = row.get("sample_index")
        key = str(index if index is not None else position)
        if key in seen:
            raise ValueError(f"重复sample_index: {key}，无法可靠关联标注")
        seen.add(key)
        model = models.get(key, {})
        gt_id = str(row.get("expected_case_id") or "")
        gt = cases.get(gt_id) or {}
        retrieved = []
        used = set()
        traces = [x for x in row.get("retrieval_trace") or [] if isinstance(x, dict) and isinstance(x.get("rank"), int)]
        for trace in sorted(traces, key=lambda x: x["rank"]):
            if not isinstance(trace, dict) or not isinstance(trace.get("rank"), int) or not 1 <= trace["rank"] <= 10:
                continue
            cid = str(trace.get("case_id") or "")
            if cid in used:
                continue
            used.add(cid)
            entry = cases.get(cid) or {}
            retrieved.append({"case_id": cid, "rank": trace["rank"],
                              "title": trace.get("case_title") or entry.get("case_name") or cid,
                              "text": entry.get("text") or "", "score": trace.get("score")})
        decisions = {str(d.get("case_id")): d for d in model.get("decisions", []) if isinstance(d, dict)}
        for item in retrieved:
            d = decisions.get(item["case_id"], {})
            item.update(model_same=d.get("same"), model_reason=d.get("reason", ""),
                        is_gt=item["case_id"] == gt_id)
        result.append({"key": key, "sample_index": index, "call_sno": row.get("call_sno"),
                       "chat_content": row.get("chat_content") or "", "query": row.get("query"),
                       "queries": row.get("queries") or [], "gt": {"case_id": gt_id,
                       "title": gt.get("case_name") or row.get("gt_case_title") or gt_id,
                       "text": gt.get("text") or ""}, "retrieved": retrieved,
                       "model_status": model.get("status", "missing"), "model_error": model.get("error"),
                       "stage1_reason": (model.get("stage1") or {}).get("reason", "")})
    return result


def load_gt_title_results(report, cases):
    """Read all saved title checks as display data, independent of labels."""
    if not isinstance(report, dict) or report.get("artifact_type") != "gt_title_retrieval_check":
        raise ValueError("GT标题检索报告必须是 gt_title_retrieval_check 格式")
    rows = report.get("records", report.get("misses"))
    if not isinstance(rows, list):
        raise ValueError("GT标题检索报告需要包含 records 数组")
    results = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("outcome") not in ("hit", "miss", "missing_title", "retrieval_failed") or not row.get("expected_case_id"):
            raise ValueError("GT标题检索报告包含无效记录")
        cid = str(row["expected_case_id"])
        if cid in results:
            raise ValueError(f"GT标题检索报告包含重复案例ID：{cid}")
        trace = row.get("retrieval_trace", [])
        if not isinstance(trace, list):
            raise ValueError(f"GT标题检索记录的 retrieval_trace 必须是数组：{cid}")
        candidates = []
        for item in trace:
            if not isinstance(item, dict) or not isinstance(item.get("rank"), int) or isinstance(item["rank"], bool) or item["rank"] < 1 or item.get("case_id") is None:
                raise ValueError(f"GT标题检索记录包含无效候选：{cid}")
            candidate_id = str(item["case_id"])
            entry = cases.get(candidate_id) or {}
            candidates.append({"rank": item["rank"], "case_id": candidate_id,
                               "title": item.get("case_title") or entry.get("case_name") or candidate_id,
                               "text": entry.get("text") or "", "score": item.get("score"), "is_gt": candidate_id == cid})
        candidates.sort(key=lambda c: c["rank"])
        matched_rank = next((c["rank"] for c in candidates if c["is_gt"]), None)
        if matched_rank != row.get("matched_rank"):
            raise ValueError(f"GT标题检索报告命中排名与候选矛盾：{cid}")
        results[cid] = {"query": row.get("query") or row.get("gt_case_title") or "",
                        "gt_case_id": cid, "gt_available": True, "gt_hit": matched_rank is not None,
                        "matched_rank": matched_rank, "created_at": report.get("created_at"),
                        "outcome": row["outcome"], "error": row.get("error"),
                        "cutoff": (report.get("configuration") or {}).get("cutoff"),
                        "cases": candidates}
    return results


class Store:
    def __init__(self, records, output, sources, *, retriever=None, gt_title_results=None):
        self.records = records
        self.by_key = {r["key"]: r for r in records}
        self.output = output
        self.lock = threading.Lock()
        self.retriever = retriever or retrieve_cases
        self.gt_title_results = gt_title_results or {}
        fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.state = {"schema_version": 1, "artifact_type": "badcase_annotations",
                      "dataset_fingerprint": fingerprint, "sources": sources, "revision": 0,
                      "updated_at": now(), "error_types": DEFAULT_TYPES[:], "annotations": {}}
        if output.exists():
            previous = load(output)
            if previous.get("artifact_type") != "badcase_annotations" or previous.get("dataset_fingerprint") != fingerprint:
                raise ValueError("输出文件属于另一份数据或模型结果已变化，请指定新的--output，避免混用标注")
            self.state = previous
        self.state.setdefault("types_revision", 0)
        self.state.setdefault("owners", [])
        for annotation in self.state["annotations"].values():
            annotation.setdefault("revision", 0)
            self.collect_owners(annotation)

    def collect_owners(self, annotation):
        for item in [annotation, *annotation.get("reviews", {}).values()]:
            name = item.get("owner", "").strip()
            if name and name not in self.state["owners"]:
                self.state["owners"].append(name)

    def try_retrieval(self, body):
        if not isinstance(body, dict):
            raise ValueError("检索请求必须是对象")
        key = str(body.get("key"))
        if key not in self.by_key:
            raise ValueError("无效badcase")
        query = body.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("请输入检索 query")
        query = query.strip()
        if len(query) > 10000:
            raise ValueError("检索 query 不能超过10000字")
        try:
            response = self.retriever(query)
            result = response.get("retrieval_result") if isinstance(response, dict) else None
            if not isinstance(result, dict):
                raise ValueError("检索响应缺少 retrieval_result 对象")
            cases = []
            gt_id = self.by_key[key]["gt"]["case_id"]
            for name, item in result.items():
                match = re.fullmatch(r"top([1-9]\d*)", str(name), re.IGNORECASE)
                if not match or not isinstance(item, dict) or item.get("case_id") is None:
                    continue
                cid = str(item["case_id"])
                cases.append({"rank": int(match[1]), "case_id": cid,
                              "title": str(item.get("case_title") or ""),
                              "text": str(item.get("content") or ""),
                              "score": item.get("score") if isinstance(item.get("score"), (int, float)) else None,
                              "is_gt": bool(gt_id) and cid == gt_id})
            cases.sort(key=lambda c: c["rank"])
        except Exception as exc:
            raise RuntimeError(f"检索失败：{exc}") from exc
        matched_rank = next((c["rank"] for c in cases if c["is_gt"]), None)
        return {"record_key": key, "query": query, "gt_case_id": gt_id,
                "gt_available": bool(gt_id), "gt_hit": matched_rank is not None,
                "matched_rank": matched_rank, "cases": cases}

    def summary(self):
        annotations = self.state["annotations"]
        values = list(annotations.values())
        completed = [a for a in values if a.get("completed")]
        alternative = [a for a in values if any(r.get("verdict") == "yes" for r in a.get("reviews", {}).values())]
        genuine = [a for a in completed if not any(r.get("verdict") in ("yes", "uncertain") for r in a.get("reviews", {}).values())]
        uncertain = [a for a in completed if not any(r.get("verdict") == "yes" for r in a.get("reviews", {}).values())
                     and any(r.get("verdict") == "uncertain" for r in a.get("reviews", {}).values())]
        counts = {name: 0 for name in self.state["error_types"]}
        for a in genuine:
            counts[a["error_type"]] = counts.get(a["error_type"], 0) + 1
        human_uncertain = sum(any(r.get("verdict") == "uncertain" for r in a.get("reviews", {}).values()) for a in values)
        model_counts = {scope: {**{name: 0 for name in self.state["error_types"]}, "": 0}
                        for scope in ("model_none", "model_positive")}
        for record in self.records:
            positive = any(c["model_same"] is True and not c["is_gt"] for c in record["retrieved"])
            if not positive and record["model_status"] != "success":
                continue
            scope = "model_positive" if positive else "model_none"
            error_type = annotations.get(record["key"], {}).get("error_type", "")
            model_counts[scope][error_type] = model_counts[scope].get(error_type, 0) + 1
        return {"total": len(self.records), "completed": len(completed), "drafts": len(values) - len(completed),
                "alternative": len(alternative), "genuine": len(genuine), "uncertain": len(uncertain), "error_counts": counts,
                "human_review_counts": {"yes": len(alternative), "no": len(genuine), "uncertain": human_uncertain},
                "error_counts_by_scope": {"genuine": counts, **model_counts},
                "model_positive": sum(any(c["model_same"] is True and not c["is_gt"] for c in r["retrieved"]) for r in self.records)}

    def snapshot(self):
        records = [{**r, "gt_title_retrieval": self.gt_title_results.get(r["gt"]["case_id"])} for r in self.records]
        return {"records": records, "state": self.state, "summary": self.summary(), "output": str(self.output)}

    def validate_annotation(self, key, raw):
        if key not in self.by_key or not isinstance(raw, dict):
            raise ValueError("无效badcase或标注内容")
        record = self.by_key[key]
        allowed = {c["case_id"] for c in record["retrieved"] if not c["is_gt"]}
        reviews = raw.get("reviews", {})
        if not isinstance(reviews, dict) or any(cid not in allowed for cid in reviews):
            raise ValueError("核验案例必须是top-10内的非GT案例")
        clean = {}
        candidates = {c["case_id"]: c for c in record["retrieved"]}
        for cid, review in reviews.items():
            if not isinstance(review, dict) or review.get("verdict", "") not in ("", "yes", "no", "uncertain"):
                raise ValueError("判断只能为是、否、不确定或未判断")
            clean[cid] = {"verdict": review.get("verdict", ""), "reason": str(review.get("reason", "")),
                          "owner": str(review.get("owner", ""))}
            candidate = candidates[cid]
            accepts_model = (clean[cid]["verdict"] == "yes" and not clean[cid]["reason"].strip()
                             and candidate["model_same"] is True and bool(candidate["model_reason"].strip()))
            clean[cid]["reason_source"] = "model" if accepts_model else "human"
            clean[cid]["effective_reason"] = candidate["model_reason"] if accepts_model else clean[cid]["reason"]
        error_type = str(raw.get("error_type", ""))
        if error_type and error_type not in self.state["error_types"]:
            raise ValueError("错误类型不存在，请刷新")
        annotation = {"record_key": key, "sample_index": record["sample_index"], "reviews": clean,
                      "error_type": error_type, "error_detail": str(raw.get("error_detail", "")),
                      "owner": str(raw.get("owner", "")), "completed": raw.get("completed") is True,
                      "updated_at": now()}
        if annotation["completed"]:
            required = {c["case_id"] for c in record["retrieved"] if c["model_same"] is True and not c["is_gt"]} | set(clean)
            for cid in required:
                r = clean.get(cid, {})
                if (r.get("verdict") not in ("yes", "no", "uncertain")
                    or (r.get("verdict") != "uncertain" and not r.get("effective_reason", "").strip())
                    or not r.get("owner", "").strip()):
                    raise ValueError("请选择判断和负责人；不确定可留空理由，选是可接受模型理由，选否必须填写理由")
            has_alternative = any(r["verdict"] == "yes" for r in clean.values())
            has_uncertain = any(r["verdict"] == "uncertain" for r in clean.values())
            if not has_alternative and not has_uncertain and (not error_type or not annotation["error_detail"].strip() or not annotation["owner"].strip()):
                raise ValueError("真实badcase需要填写错误类型、详细说明和负责人")
        annotation["outcome"] = ("alternative_found" if any(r["verdict"] == "yes" for r in clean.values()) else
                                 ("uncertain" if any(r["verdict"] == "uncertain" for r in clean.values()) else
                                  ("genuine_badcase" if annotation["completed"] else "pending")))
        return annotation

    def mutate(self, body):
        with self.lock:
            action = body.get("action")
            key = str(body.get("key"))
            if action == "annotation":
                current_revision = self.state["annotations"].get(key, {}).get("revision", 0)
                if body.get("record_revision") != current_revision:
                    raise RuntimeError("这条badcase已被其他人修改。你的草稿已保留，请比较最新标注后选择处理方式。")
            elif action in ("add_type", "delete_type") and body.get("types_revision") != self.state["types_revision"]:
                raise RuntimeError("错误类型已被其他人修改，请同步后重试。")
            old = copy.deepcopy(self.state)
            try:
                action = body.get("action")
                if action == "annotation":
                    key = str(body.get("key"))
                    annotation = self.validate_annotation(key, body.get("annotation"))
                    annotation["revision"] = current_revision + 1
                    self.state["annotations"][key] = annotation
                    self.collect_owners(annotation)
                elif action == "add_owner":
                    name = str(body.get("name", "")).strip()
                    if not name or len(name) > 80:
                        raise ValueError("姓名需要非空且不超过80字")
                    if name not in self.state["owners"]:
                        self.state["owners"].append(name)
                elif action == "add_type":
                    name = str(body.get("name", "")).strip()
                    if not name or len(name) > 80 or name in self.state["error_types"]:
                        raise ValueError("类型名称需要非空、不重复且不超过80字")
                    self.state["error_types"].insert(len(self.state["error_types"]) - 1, name)
                elif action == "delete_type":
                    name = body.get("name")
                    if name == "其他" or name not in self.state["error_types"]:
                        raise ValueError("不能删除其他或不存在的类型")
                    self.state["error_types"].remove(name)
                    for annotation in self.state["annotations"].values():
                        if annotation.get("error_type") == name:
                            annotation.update(error_type="其他", updated_at=now())
                            annotation["error_detail"] = f"[原错误类型：{name}]\n" + annotation["error_detail"]
                            annotation["revision"] += 1
                else:
                    raise ValueError("未知操作")
                if action in ("add_type", "delete_type"):
                    self.state["types_revision"] += 1
                self.state["revision"] += 1
                self.state["updated_at"] = now()
                self.state["summary"] = self.summary()
                atomic_write(self.output, self.state)
            except Exception:
                self.state = old
                raise
            return {"state": self.state, "summary": self.summary()}


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/sync":
                with store.lock:
                    self.respond(200, {"state": store.state, "summary": store.summary()})
                return
            if self.path == "/api/data":
                with store.lock:
                    self.respond(200, store.snapshot())
                return
            if self.path == "/api/export":
                with store.lock:
                    self.respond(200, store.state)
                return
            name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(self.path)
            if not name:
                self.respond(404, {"error": "Not found"})
                return
            data = (ROOT / "static" / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[name])
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path not in ("/api/save", "/api/retrieve"):
                self.respond(404, {"error": "Not found"})
                return
            # Only same-origin browser writes are accepted by this local service.
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.respond(403, {"error": "仅允许当前页面保存"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError("请求大小不合法")
                body = json.loads(self.rfile.read(length))
                self.respond(200, store.try_retrieval(body) if self.path == "/api/retrieve" else store.mutate(body))
            except RuntimeError as exc:
                self.respond(502 if self.path == "/api/retrieve" else 409, {"error": str(exc)})
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(400, {"error": str(exc)})
            except OSError as exc:
                self.respond(500, {"error": f"写盘失败，未保存：{exc}"})

        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else "") not in ("200",):
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description="本地badcase翻页标注器")
    parser.add_argument("--badcases", help="collect_badcases.py输出")
    parser.add_argument("--analysis", help="analyze_badcases.py输出")
    parser.add_argument("--cases", help="案例库case_id -> {case_name,text}")
    parser.add_argument("--output", default=str(ROOT / "annotations.json"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="使用独立的模拟数据体验界面")
    parser.add_argument("--retrieval-url", help="尝试检索接口地址，默认使用 search.py 中的地址")
    parser.add_argument("--gt-title-report", help="check_gt_retrieval.py 输出；在左栏展示所有GT标题检索结果")
    parser.add_argument("--retrieval-timeout", type=float, default=30, help="尝试检索超时秒数（默认30）")
    args = parser.parse_args()
    if not 0 < args.retrieval_timeout <= 300:
        parser.error("--retrieval-timeout 需要大于0且不超过300秒")
    if args.demo:
        demo = load(ROOT / "demo_data.json")
        records = build_dataset(demo["badcases"], demo["analysis"], demo["cases"])
        output = Path(args.output) if args.output != str(ROOT / "annotations.json") else ROOT / "demo_annotations.json"
        sources = {"demo": True}
    else:
        if not all((args.badcases, args.analysis, args.cases)):
            parser.error("需要--badcases、--analysis、--cases；或使用--demo")
        output = Path(args.output).resolve()
        sources = {k: str(Path(getattr(args, k)).resolve()) for k in ("badcases", "analysis", "cases")}
        if output in [Path(p) for p in sources.values()]:
            parser.error("输出路径不能覆盖输入文件")
        records = build_dataset(*(load(sources[k]) for k in ("badcases", "analysis", "cases")))
    gt_title_results = {}
    if args.gt_title_report:
        if Path(args.gt_title_report).resolve() == output.resolve():
            parser.error("输出路径不能覆盖GT标题检索报告")
        cases = demo["cases"] if args.demo else load(sources["cases"])
        gt_title_results = load_gt_title_results(load(args.gt_title_report), cases)
    store = Store(records, output.resolve(), sources, gt_title_results=gt_title_results,
                  retriever=lambda query: retrieve_cases(query, url=args.retrieval_url, timeout=args.retrieval_timeout))
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler(store))
    print(f"标注器已启动：http://0.0.0.0:{args.port}\n共{len(records)}条，实时保存到：{store.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止，已保存标注可在下次启动继续使用。")
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"启动失败：{exc}")
