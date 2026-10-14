"""将 GT 标题检索失败且模型认为没有其他正确案例的记录导入标注落盘文件。

运行前停止标注器，导入后用原来的启动命令重新启动。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

try:
    from .server import Store, atomic_write, build_dataset, load, now
except ImportError:
    from server import Store, atomic_write, build_dataset, load, now


ERROR_TYPE = "检索端问题"
ERROR_DETAIL = "即使用GT案例标题检索也检索不到"
OWNER = "唐锦康"


def import_report(store, report, *, overwrite=False):
    """Prepare imports in memory using the annotator's own validation and summary."""
    if not isinstance(report, dict) or report.get("artifact_type") != "gt_title_retrieval_check":
        raise ValueError("结果文件必须是 check_gt_retrieval.py 输出的 gt_title_retrieval_check")
    if not isinstance(report.get("misses"), list):
        raise ValueError("结果文件缺少 misses 数组")
    missed_ids = set()
    for row in report["misses"]:
        if not isinstance(row, dict) or row.get("outcome") != "miss" or not row.get("expected_case_id"):
            raise ValueError("misses 包含无效记录")
        missed_ids.add(str(row["expected_case_id"]))
    # The requested classification specifically means absent from all results,
    # even when the source report was produced with a top-K cutoff.
    absent_ids = {str(row["expected_case_id"]) for row in report["misses"]
                  if row.get("matched_rank") is None}
    stats = {"title_miss_cases": len(missed_ids), "outside_cutoff_only_cases": len(missed_ids - absent_ids),
             "matched_badcases": 0, "model_excluded": 0, "existing_skipped": 0,
             "review_conflicts": 0, "unchanged": 0, "imported": 0, "imported_keys": []}
    original = copy.deepcopy(store.state)
    for record in store.records:
        if record["gt"]["case_id"] not in absent_ids:
            continue
        stats["matched_badcases"] += 1
        if record["model_status"] != "success" or any(
            c["model_same"] is True and not c["is_gt"] for c in record["retrieved"]
        ):
            stats["model_excluded"] += 1
            continue
        key = record["key"]
        existing = store.state["annotations"].get(key)
        if existing and all(existing.get(field) == value for field, value in {
            "error_type": ERROR_TYPE, "error_detail": ERROR_DETAIL, "owner": OWNER,
            "completed": True, "outcome": "genuine_badcase",
        }.items()):
            stats["unchanged"] += 1
            continue
        if existing is not None and not overwrite:
            stats["existing_skipped"] += 1
            continue
        reviews = (existing or {}).get("reviews", {})
        if any(r.get("verdict") in ("yes", "uncertain", "") for r in reviews.values()):
            stats["review_conflicts"] += 1
            continue
        if ERROR_TYPE not in store.state["error_types"]:
            store.state["error_types"].append(ERROR_TYPE)
            store.state["types_revision"] += 1
        annotation = store.validate_annotation(key, {
            "reviews": reviews, "error_type": ERROR_TYPE, "error_detail": ERROR_DETAIL,
            "owner": OWNER, "completed": True,
        })
        annotation["revision"] = (existing or {}).get("revision", 0) + 1
        store.state["annotations"][key] = annotation
        store.collect_owners(annotation)
        stats["imported"] += 1
        stats["imported_keys"].append(key)
    if stats["imported"]:
        store.state["revision"] += 1
        store.state["updated_at"] = now()
        store.state["summary"] = store.summary()
        store.state.setdefault("import_history", []).append({
            "imported_at": now(), "kind": "gt_title_retrieval_check",
            "record_keys": stats["imported_keys"], "owner": OWNER,
        })
    else:
        store.state = original
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="GT 标题检索检查结果")
    parser.add_argument("--annotations", required=True, help="标注器落盘文件（原地更新，自动备份）")
    for name in ("badcases", "analysis", "cases"):
        parser.add_argument("--" + name, help="默认读取落盘文件 sources 中的路径；也可显式指定")
    parser.add_argument("--overwrite", action="store_true", help="覆盖符合条件的已有标注；仍保留人工核验且跳过是/不确定/未判断")
    parser.add_argument("--dry-run", action="store_true", help="仅预览数量，不写入或备份")
    args = parser.parse_args(argv)
    try:
        output = Path(args.annotations).resolve()
        before = output.read_bytes() if output.exists() else None
        previous = json.loads(before) if before is not None else {}
        sources = previous.get("sources", {}).copy()
        for name in ("badcases", "analysis", "cases"):
            source = getattr(args, name) or sources.get(name)
            if not source:
                raise ValueError(f"请提供 --{name}，落盘文件 sources 中没有该路径")
            sources[name] = str(Path(source).resolve())
        if output in [Path(p) for p in sources.values() if isinstance(p, str)] or output == Path(args.report).resolve():
            raise ValueError("标注落盘文件不能覆盖输入文件")
        records = build_dataset(*(load(sources[k]) for k in ("badcases", "analysis", "cases")))
        store = Store(records, output, sources)
        stats = import_report(store, load(args.report), overwrite=args.overwrite)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if args.dry_run:
            print("预览完成，未修改文件。")
        elif stats["imported"]:
            if (output.read_bytes() if output.exists() else None) != before:
                raise ValueError("落盘文件在导入期间已变化，请停止标注器后重试")
            if before is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(prefix=output.name + ".", suffix=".bak", dir=output.parent, delete=False) as backup:
                    backup.write(before)
                    print(f"原文件备份：{backup.name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(output, store.state)
            print(f"已导入 {stats['imported']} 条：{output}；请重新启动标注器。")
        else:
            print("没有需要导入的记录，未修改文件。")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"导入失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
