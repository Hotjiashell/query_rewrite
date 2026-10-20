"""离线重算已有 evaluate.py 结果在新标注数据上的四组 Recall@K。

python evaluate_annotated.py --results results/baseline.json \
    --dataset data/dialog_annotated.json --output results/baseline_annotated.json
"""
from __future__ import annotations

import argparse
from typing import Sequence

from evaluate import main as evaluate_main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", "--input", dest="results", required=True,
                        help="evaluate.py 输出的 retrieval_evaluation 结果文件")
    parser.add_argument("--dataset", required=True,
                        help="新标注数据 JSON 数组，包含 caseID、m_caseID 和 can_retrieve")
    parser.add_argument("--output", required=True, help="新结果文件，保留原结果并增加 annotated_metrics")
    args = parser.parse_args(argv)
    return evaluate_main([
        "--input", args.results, "--dataset", args.dataset, "--output", args.output,
    ], forced_stage="metrics")


if __name__ == "__main__":
    raise SystemExit(main())
