"""Train a DSPy program with GEPA against retrieval Recall@K."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common import (
    DialogueExample,
    GoldenReference,
    load_case_titles,
    load_examples,
    load_golden_references,
    load_settings,
    split_examples,
    write_json,
)
from query_program import make_lm, make_program
from retrieval import CaseRetriever, matched_rank


def _dspy_examples(rows: list[DialogueExample], golden_references: dict[int, GoldenReference]):
    import dspy

    examples = []
    for row in rows:
        reference = golden_references.get(row.sample_index)
        examples.append(
            dspy.Example(
                dialogue=row.dialogue,
                expected_case_id=row.expected_case_id,
                expected_case_title=row.expected_case_title,
                sample_index=row.sample_index,
                golden_query=reference.query if reference else None,
            ).with_inputs("dialogue")
        )
    return examples


def _metric(retriever: CaseRetriever, cutoff: int):
    def score(
        example: Any,
        prediction: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> Any:
        # DSPy calls this both for the program score and to ask the target
        # predictor for reflection feedback. Keep the retrieval trace in the
        # latter so GEPA can diagnose why an otherwise valid query missed.
        import dspy

        query = getattr(prediction, "query", "")
        if not isinstance(query, str) or not query.strip():
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"【用户对话】\n{example.dialogue}\n\n"
                    "【生成的 query】\n（空）\n\n"
                    f"【目标案例标题】\n{example.expected_case_title}\n\n"
                    "【是否命中目标案例】\n否\n\n"
                    "【实际检索到的案例标题】\n未发起检索，因为生成的 query 为空。"
                ),
            )
        try:
            cases = retriever.retrieve(query.strip())
        except Exception:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"【用户对话】\n{example.dialogue}\n\n"
                    f"【生成的 query】\n{query.strip()}\n\n"
                    f"【目标案例标题】\n{example.expected_case_title}\n\n"
                    "【是否命中目标案例】\n否\n\n"
                    "【实际检索到的案例标题】\n检索请求失败，未返回有效结果。"
                ),
            )
        rank = matched_rank(example.expected_case_id, cases)
        value = float(rank is not None and rank <= cutoff)
        trace_text = "\n".join(f"- {item.case_title}" for item in cases)
        feedback = (
            f"【用户对话】\n{example.dialogue}\n\n"
            f"【生成的 query】\n{query.strip()}\n\n"
            f"【目标案例标题】\n{example.expected_case_title}\n\n"
            f"【是否命中目标案例】\n{'是' if value else '否'}\n\n"
            f"【实际检索到的案例标题】\n{trace_text or '检索服务未返回有效结果。'}"
        )
        if not value:
            feedback += (
                "\n\n请分析：从上述用户对话中抽取的 query 未能检索到目标案例，实际检索结果如上。"
                "请据此总结如何改进 query 生成提示词。"
            )
            if getattr(example, "golden_query", None):
                feedback += (
                    "\n\n【Golden query】\n"
                    f"{example.golden_query}"
                    "\n\nGolden query 能够检索到目标案例，仅用于离线反思。"
                    "可以据此对比分析如何改进 query 生成提示词，不要直接照抄 Golden query。"
                )
        return dspy.Prediction(score=value, feedback=feedback)

    return score


def _program_instruction(program: Any) -> str:
    predictors = list(program.named_predictors())
    if len(predictors) != 1:
        raise RuntimeError("expected exactly one DSPy predictor")
    return predictors[0][1].signature.instructions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize the query-rewrite prompt using DSPy GEPA.")
    parser.add_argument("--config", default="gepa/config.json")
    parser.add_argument("--run-name", default="latest")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    rows = load_examples(settings.input_path, load_case_titles(settings.case_path))
    golden_references = load_golden_references(settings.golden_query_path)
    train_rows, val_rows = split_examples(rows, settings.train_ratio, settings.split_seed)
    run_dir = settings.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError("DSPy is not installed. Run: pip install -r gepa/requirements.txt") from exc

    task_lm = make_lm(settings.task_lm)
    reflection_lm = make_lm(settings.reflection_lm)
    dspy.configure(lm=task_lm)
    retriever = CaseRetriever(settings.retrieval.url, settings.retrieval.timeout, settings.retrieval.top_k)
    optimizer = dspy.GEPA(
        metric=_metric(retriever, settings.metric_cutoff),
        max_metric_calls=settings.max_metric_calls,
        reflection_minibatch_size=settings.reflection_minibatch_size,
        reflection_lm=reflection_lm,
        num_threads=settings.num_threads,
        log_dir=str(run_dir / "dspy_gepa_logs"),
    )
    optimized = optimizer.compile(
        student=make_program(),
        trainset=_dspy_examples(train_rows, golden_references),
        valset=_dspy_examples(val_rows, golden_references),
    )
    instruction = _program_instruction(optimized)
    (run_dir / "optimized_instruction.txt").write_text(instruction + "\n", encoding="utf-8")
    optimized.save(str(run_dir / "optimized_program.json"), save_program=False)
    write_json({
        "schema_version": 1,
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "golden_reference_samples": len(golden_references),
        "metric": f"recall_at_{settings.metric_cutoff}",
        "max_metric_calls": settings.max_metric_calls,
        "reflection_minibatch_size": settings.reflection_minibatch_size,
        "optimized_instruction": instruction,
    }, run_dir / "training_summary.json")
    print(json.dumps({"run_dir": str(run_dir), "train_samples": len(train_rows), "validation_samples": len(val_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Optimization failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
