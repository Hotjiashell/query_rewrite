"""将人工确认的正确案例合并到测试数据的 m_caseID 数组中。"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

try:
    from .server import atomic_write, load
except ImportError:
    from server import atomic_write, load


def case_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("案例 ID 必须是非空字符串")
    return value.strip()


def export_test_data(dialogues, annotations, badcases, *, completed_only=False):
    if not isinstance(dialogues, list):
        raise ValueError("测试数据必须是 JSON 数组")
    if not isinstance(annotations, dict) or annotations.get("artifact_type") != "badcase_annotations":
        raise ValueError("请提供标注器输出的 badcase_annotations 文件")
    labels = annotations.get("annotations")
    if not isinstance(labels, dict):
        raise ValueError("标注文件缺少 annotations 对象")
    if not isinstance(badcases, dict) or not isinstance(badcases.get("records"), list):
        raise ValueError("badcases 文件必须包含 records 数组")
    records = {}
    for position, record in enumerate(badcases["records"]):
        if not isinstance(record, dict):
            raise ValueError("badcases 中包含无效记录")
        index = record.get("sample_index")
        key = str(index if index is not None else position)
        if key in records:
            raise ValueError(f"badcases 中存在重复记录键：{key}")
        records[key] = record

    result = copy.deepcopy(dialogues)
    for index, row in enumerate(result):
        if not isinstance(row, dict):
            raise ValueError(f"测试数据第 {index} 条不是对象")
        original = case_id(row.get("caseID"))
        existing = row.get("m_caseID", [])
        if not isinstance(existing, list):
            raise ValueError(f"测试数据第 {index} 条的 m_caseID 必须是数组")
        row["m_caseID"] = list(dict.fromkeys([original, *(case_id(cid) for cid in existing)]))

    added = 0
    updated = set()
    for key, annotation in labels.items():
        if not isinstance(annotation, dict) or not isinstance(annotation.get("reviews", {}), dict):
            raise ValueError(f"标注 {key} 格式无效")
        if completed_only and not annotation.get("completed"):
            continue
        positives = [case_id(cid) for cid, review in annotation.get("reviews", {}).items()
                     if isinstance(review, dict) and review.get("verdict") == "yes"]
        if not positives:
            continue
        record = records.get(key)
        if record is None:
            raise ValueError(f"人工确认记录 {key} 不在 badcases 文件中")
        index = record.get("sample_index")
        if type(index) is not int or not 0 <= index < len(result):
            raise ValueError(f"记录 {key} 的 sample_index 无效或超出测试数据范围：{index}")
        row = result[index]
        if annotation.get("sample_index") != index:
            raise ValueError(f"记录 {key} 的标注 sample_index 与 badcases 不一致")
        if case_id(record.get("expected_case_id")) != case_id(row["caseID"]) or record.get("chat_content") != row.get("chat_content"):
            raise ValueError(f"记录 {key} 与测试数据第 {index} 条的原正确案例或对话不匹配，请检查输入文件和数据顺序")
        if record.get("call_sno") is not None and str(record["call_sno"]) != str(row.get("call_sno")):
            raise ValueError(f"记录 {key} 的 call_sno 与测试数据不匹配")
        allowed = {case_id(c.get("case_id")) for c in record.get("retrieval_trace", [])
                   if isinstance(c, dict) and type(c.get("rank")) is int and 1 <= c["rank"] <= 10}
        for cid in positives:
            if cid not in allowed:
                raise ValueError(f"记录 {key} 的人工正确案例 {cid} 不在原 Top-10 召回列表中")
            if cid not in row["m_caseID"]:
                row["m_caseID"].append(cid)
                added += 1
                updated.add(index)
    return result, {"total": len(result), "updated_samples": len(updated), "added_case_ids": added}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="原测试数据，例如 data/dialog_example.json")
    parser.add_argument("--annotations", required=True, help="标注器保存的标注 JSON 文件")
    parser.add_argument("--badcases", help="原 badcases 文件；默认使用标注文件 sources.badcases")
    parser.add_argument("--output", required=True, help="新的测试数据路径，不能覆盖输入文件")
    parser.add_argument("--completed-only", action="store_true", help="仅合并已完成标注中的人工是判断")
    args = parser.parse_args(argv)
    try:
        annotation_path = Path(args.annotations).resolve()
        annotations = load(annotation_path)
        source = args.badcases or (annotations.get("sources", {}).get("badcases") if isinstance(annotations, dict) else None)
        if not source:
            raise ValueError("请提供 --badcases，标注文件 sources.badcases 中没有原 badcases 路径")
        badcase_path = Path(source).resolve()
        output = Path(args.output).resolve()
        if output in {Path(args.input).resolve(), annotation_path, badcase_path}:
            raise ValueError("输出路径不能覆盖原测试数据、标注文件或 badcases 文件")
        result, stats = export_test_data(load(args.input), annotations, load(badcase_path), completed_only=args.completed_only)
        atomic_write(output, result)
        print(f"已输出：{output}\n共 {stats['total']} 条数据，{stats['updated_samples']} 条补充了正确案例，新增 {stats['added_case_ids']} 个案例 ID。")
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"导出失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
