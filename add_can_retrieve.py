"""根据 check_gt_retrieval.py 的标题检索报告，为测试数据增加 can_retrieve。"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from label_badcase.server import atomic_write, load


def add_can_retrieve(samples, report):
    if not isinstance(samples, list):
        raise ValueError("原始测试数据必须是 JSON 数组")
    if not isinstance(report, dict) or report.get("artifact_type") != "gt_title_retrieval_check":
        raise ValueError("需要 check_gt_retrieval.py 输出的 gt_title_retrieval_check 报告，不能使用 badcase_analysis")
    if not isinstance(report.get("records"), list):
        raise ValueError("报告缺少完整 records 数组，不能仅通过 misses 推断命中情况")
    by_id = {}
    for record in report["records"]:
        if not isinstance(record, dict):
            raise ValueError("报告 records 中包含无效记录")
        cid = record.get("expected_case_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("报告记录缺少 expected_case_id")
        cid = cid.strip()
        if cid in by_id:
            raise ValueError(f"报告包含重复案例 ID：{cid}")
        outcome = record.get("outcome")
        if outcome not in ("hit", "miss", "missing_title", "retrieval_failed"):
            raise ValueError(f"案例 {cid} 的 outcome 无效：{outcome}")
        rank = record.get("matched_rank")
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError(f"案例 {cid} 的 matched_rank 无效")
        trace = record.get("retrieval_trace", [])
        if not isinstance(trace, list) or any(not isinstance(c, dict) for c in trace):
            raise ValueError(f"案例 {cid} 的 retrieval_trace 无效")
        returned = rank is not None or any(str(c.get("case_id", "")).strip() == cid for c in trace)
        if outcome == "hit" and not returned:
            raise ValueError(f"案例 {cid} 标为 hit，但没有命中排名或返回候选证据")
        if outcome in ("missing_title", "retrieval_failed") and returned:
            raise ValueError(f"案例 {cid} 的错误状态与命中结果矛盾")
        by_id[cid] = (returned, outcome)

    result = copy.deepcopy(samples)
    stats = {"total": len(result), "true": 0, "false": 0, "missing_title": 0, "retrieval_failed": 0}
    for index, sample in enumerate(result):
        if not isinstance(sample, dict):
            raise ValueError(f"测试数据第 {index} 条不是对象")
        cid = sample.get("caseID")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"测试数据第 {index} 条缺少有效 caseID")
        cid = cid.strip()
        if cid not in by_id:
            raise ValueError(f"测试数据第 {index} 条的案例 {cid} 未出现在报告中，请先完成标题检索检查")
        returned, outcome = by_id[cid]
        sample["can_retrieve"] = returned
        stats["true" if returned else "false"] += 1
        if outcome in ("missing_title", "retrieval_failed"):
            stats[outcome] += 1
    return result, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="原测试数据 JSON 数组，也支持已有 m_caseID 的数据")
    parser.add_argument("--report", required=True, help="check_gt_retrieval.py 输出的完整结果文件")
    parser.add_argument("--output", required=True, help="增加 can_retrieve 后的新数据文件")
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).resolve()
        if output in {Path(args.input).resolve(), Path(args.report).resolve()}:
            raise ValueError("输出文件不能覆盖输入数据或标题检索报告")
        result, stats = add_can_retrieve(load(args.input), load(args.report))
        atomic_write(output, result)
        print(f"已输出：{output}\n共 {stats['total']} 条，can_retrieve=true：{stats['true']} 条，false：{stats['false']} 条。")
        print(f"false 中缺少标题：{stats['missing_title']} 条，检索请求失败：{stats['retrieval_failed']} 条（可重试标题检索后重新导出）。")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"导出失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
