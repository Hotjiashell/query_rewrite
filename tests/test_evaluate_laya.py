import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate import GeneratedQueryRecord, RetrievedCase, build_query_artifact
from evaluate_laya import (
    LayaMultiQueryEvaluator, LayaRelevanceFilter, analyse_saved_results, fuse_score, main, restore_dialogues,
)
from gen_query import MultiQueryGenerator, PromptMultiQueryGenerator


class Agent:
    def __init__(self, probabilities=None, truncated=False):
        self.probabilities = probabilities or {}
        self.truncated = truncated
        self.calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((states, questions, kwargs))
        return [
            {"answers": {"relevance": {
                "choice": "related" if self.probabilities.get(state["case_title"], 0.9) >= 0.5 else "unrelated",
                "probabilities": {"related": self.probabilities.get(state["case_title"], 0.9)},
            }}, "usage": {"truncated": self.truncated}}
            for state in states
        ]


class Retriever:
    def __init__(self, failures=()):
        self.failures = failures

    def retrieve(self, query):
        if query in self.failures:
            raise RuntimeError("offline")
        cases = {
            "q1": [("A", "irrelevant", 0.99), ("B", "useful", 0.2), ("B", "useful", 0.1), ("D", "extra", 0.4)],
            "q2": [("B", "useful", 0.9), ("C", "other", 0.8)],
        }[query]
        return {"retrieval_result": {
            f"top{i}": {"case_id": cid, "case_title": title, "score": score}
            for i, (cid, title, score) in enumerate(cases, 1)
        }}


def source():
    return GeneratedQueryRecord(0, "call", "B", "q1", "success", None,
                                chat_content="完整的客服对话", queries=["q1", "q2"])


class LayaEvaluationTests(unittest.TestCase):
    def run_pipeline(self, fusion="round_robin", agent=None, retriever=None, top_k=10):
        agent = agent or Agent({"irrelevant": 0.1})
        evaluator = LayaMultiQueryEvaluator(
            retriever or Retriever(), LayaRelevanceFilter(agent),
            fusion_method=fusion, top_k=top_k,
        )
        return evaluator.evaluate_query(source()), agent

    def test_dedup_before_judging_then_filter_and_interleave(self):
        result, agent = self.run_pipeline()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["unique_candidate_count"], 4)
        self.assertEqual(result["filtered_candidate_count"], 1)
        states = agent.calls[0][0]
        self.assertEqual(len(states), 4)
        self.assertTrue(all(set(state) == {"dialogue", "case_title"} for state in states))
        self.assertEqual([c["case_id"] for c in result["retrieval_trace"]], ["B", "C", "D"])
        self.assertEqual(result["matched_rank"], 1)
        self.assertEqual(result["prefilter_matched_rank"], 2)
        self.assertEqual([c["case_id"] for c in result["per_query_traces"][0]["trace"]], ["A", "B", "D"])

    def test_filter_before_top_k_so_lower_candidates_can_fill(self):
        result, _ = self.run_pipeline(top_k=2)
        self.assertEqual([c["case_id"] for c in result["retrieval_trace"]], ["B", "C"])

    def test_score_fusion_uses_best_retrieval_score_not_relevance(self):
        result, _ = self.run_pipeline("score", agent=Agent({"irrelevant": 0.1, "useful": 0.6, "other": 0.99}))
        self.assertEqual([c["case_id"] for c in result["retrieval_trace"]], ["B", "C", "D"])
        self.assertEqual(result["retrieval_trace"][0]["score"], 0.9)

    def test_zero_and_negative_scores_are_not_missing(self):
        traces = [[RetrievedCase(1, "A", "a", 0), RetrievedCase(2, "B", "b", None)],
                  [RetrievedCase(1, "A", "a", -0.1)]]
        result = fuse_score(traces, 10)
        self.assertEqual(result[0].score, 0)
        self.assertEqual(result[0].case_id, "A")

    def test_partial_retrieval_succeeds_and_keeps_query_alignment(self):
        result, _ = self.run_pipeline(retriever=Retriever(failures=("q1",)))
        self.assertEqual(result["retrieval_status"], "partial_success")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["per_query_traces"][0]["trace"], [])
        self.assertEqual(result["per_query_traces"][1]["query"], "q2")

    def test_all_retrievals_fail_without_calling_laya(self):
        result, agent = self.run_pipeline(retriever=Retriever(failures=("q1", "q2")))
        self.assertEqual(result["retrieval_status"], "failed")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(agent.calls, [])

    def test_all_filtered_is_success_with_empty_trace(self):
        result, _ = self.run_pipeline(agent=Agent({title: 0.1 for title in ("irrelevant", "useful", "extra", "other")}))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["retrieval_trace"], [])
        self.assertIsNone(result["matched_rank"])

    def test_truncation_and_invalid_probabilities_do_not_silently_filter(self):
        for agent in (Agent(truncated=True), Agent({"useful": float("nan")})):
            with self.subTest(agent=agent):
                result, _ = self.run_pipeline(agent=agent)
                self.assertEqual(result["filter_status"], "failed")
                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["prefilter_trace"])
                self.assertEqual(result["retrieval_trace"], [])

    def test_missing_dialogue_or_multi_query_is_explicit_failure(self):
        evaluator = LayaMultiQueryEvaluator(Retriever(), LayaRelevanceFilter(Agent()))
        for record in (replace(source(), chat_content=None), replace(source(), queries=None)):
            self.assertEqual(evaluator.evaluate_query(record)["status"], "failed")

    def test_restore_dialogue_rejects_mismatched_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogues.json"
            path.write_text(json.dumps([{"chat_content": "original", "caseID": "B", "call_sno": "call"}]))
            restored = restore_dialogues([replace(source(), chat_content=None)], str(path))
            self.assertEqual(restored[0].chat_content, "original")
            with self.assertRaises(ValueError):
                restore_dialogues([replace(source(), chat_content=None, call_sno="wrong")], str(path))

    def test_generate_uses_only_custom_multi_and_does_not_load_laya(self):
        class Generator(MultiQueryGenerator):
            def generate_queries(self, dialogue):
                return ["q1", "q2"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, dialogues, queries = root / "config.json", root / "dialogues.json", root / "queries.json"
            dialogues.write_text(json.dumps([{"chat_content": "对话", "caseID": "B", "call_sno": "call"}]))
            config.write_text(json.dumps({
                "llm": {"base_url": "http://model", "model_name": "test", "api_key": "test"},
                "query_generation": {"input_path": str(dialogues), "output_path": str(queries), "method": "baseline"},
            }))
            with patch("evaluate_laya.create_query_generator", return_value=Generator()) as create, \
                 patch.dict("sys.modules", {"laya": None}):
                status = main(["generate", "--config", str(config), "--prompt-file", "my-prompt.txt"])
            self.assertEqual(status, 0)
            self.assertEqual(create.call_args.args[1], "custom_multi")
            self.assertEqual(create.call_args.kwargs["prompt_file"], "my-prompt.txt")
            artifact = json.loads(queries.read_text())
            self.assertEqual(artifact["configuration"]["method"], "custom_multi")
            self.assertEqual(artifact["records"][0]["queries"], ["q1", "q2"])

    def test_filter_rejects_invalid_configuration_and_empty_titles(self):
        for options in ({"threshold": float("nan")}, {"threshold": True},
                        {"batch_size": 1.5}, {"head_max_len": 1024}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LayaRelevanceFilter(Agent(), **options)
        with self.assertRaises(ValueError):
            LayaRelevanceFilter(Agent()).judge("对话", [RetrievedCase(1, "A", "")])

    def test_included_prompt_and_real_multi_query_parser_work_together(self):
        from gen_query import load_prompt_template
        prompt = load_prompt_template(str(Path(__file__).resolve().parents[1] / "prompts" / "laya_multi_query.txt"))
        completions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"query": ["产品A登录失败", "产品A密码重置方法"]}'))],
        ))
        generator = PromptMultiQueryGenerator(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            "test", "custom_multi", prompt_template=prompt,
        )
        self.assertEqual(generator.generate_queries("产品A无法登录"), ["产品A登录失败", "产品A密码重置方法"])

    def test_existing_query_cli_bypasses_generator_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, output, config = root / "queries.json", root / "result.json", root / "config.json"
            queries.write_text(json.dumps(build_query_artifact([source()], input_path="dialogs.json", model_name="test", concurrency=1, method="custom_multi")))
            config.write_text(json.dumps({"retrieval": {"output_path": str(output)}}))
            with patch.dict("sys.modules", {"laya": SimpleNamespace(load=lambda *a, **k: Agent({"irrelevant": 0.1}))}), \
                 patch("evaluate_laya.SearchRetriever", return_value=Retriever()), \
                 patch("evaluate_laya.create_query_generator", side_effect=AssertionError("LLM must not be called")):
                status = main(["all", "--config", str(config), "--query-file", str(queries)])
            self.assertEqual(status, 0)
            artifact = json.loads(output.read_text())
            self.assertEqual(artifact["artifact_type"], "retrieval_evaluation")
            self.assertEqual(artifact["prefilter_metrics"]["recall_at_1"], 0)
            self.assertEqual(artifact["metrics"]["recall_at_1"], 1)
            self.assertEqual(artifact["union_top_k"]["recall_at_1"], 1)
            self.assertEqual(artifact["threshold_search"]["best"]["recall_at_10"], 1)


def saved_artifact(records):
    return {"artifact_type": "retrieval_evaluation", "configuration": {"top_k": 3}, "records": records}


def audit_record(cases, probabilities, expected="target"):
    return {
        "sample_index": 0, "expected_case_id": expected,
        "retrieval_status": "success", "filter_status": "success",
        "per_query_traces": [{"query": "q", "trace": [
            {"case_id": cid, "rank": rank, "case_title": cid, "score": score}
            for cid, rank, score in cases
        ]}],
        "laya_judgments": [{"case_id": cid, "related_probability": p} for cid, p in probabilities.items()],
    }


class SavedAnalysisTests(unittest.TestCase):
    def test_union_uses_each_query_original_rank_and_counts_each_sample_once(self):
        record = audit_record([("target", 12, 0.2)], {"target": 0.5})
        record["per_query_traces"].extend([
            {"query": "q2", "trace": [{"case_id": "target", "case_title": "t", "rank": 3, "score": 0.5}]},
            {"query": "q3", "trace": [{"case_id": "target", "case_title": "t", "rank": 1, "score": 0.5}]},
        ])
        failed = audit_record([], {})
        failed.update(filter_status="failed", retrieval_status="failed")
        artifact = analyse_saved_results(saved_artifact([record, failed]))
        self.assertEqual(record["union_matched_rank"], 1)
        self.assertTrue(record["union_top_k"]["1"])
        self.assertEqual(artifact["union_top_k"]["hits_at_10"], 1)
        self.assertEqual(artifact["union_top_k"]["recall_at_10"], 0.5)
        self.assertEqual(artifact["threshold_search"]["best"]["recall_at_10"], 0.5)
        self.assertEqual(artifact["threshold_search"]["evaluable_samples"], 1)

    def test_exact_search_finds_non_grid_optimum_and_fixed_top_ten(self):
        cases = [(f"bad{i}", i + 1, 1.0) for i in range(10)] + [("target", 11, 0.1)]
        probabilities = {f"bad{i}": 0.5001 for i in range(10)} | {"target": 0.5002}
        artifact = analyse_saved_results(saved_artifact([audit_record(cases, probabilities)]))
        best = artifact["threshold_search"]["best"]
        self.assertEqual(best["threshold"], 0.5002)
        self.assertEqual(best["recall_at_10"], 1)
        self.assertEqual(best["fusion_method"], "round_robin")
        self.assertEqual(artifact["union_top_k"]["recall_at_10"], 0)
        self.assertEqual(artifact["union_top_k"]["all_candidates_recall"], 1)
        self.assertEqual(artifact["threshold_search"]["fusion_top_k"], 10)

    def test_interval_accumulation_matches_brute_force_for_both_fusions(self):
        import random
        randomizer = random.Random(17)
        records = []
        for i in range(15):
            ids = [f"c{j}" for j in range(14)]
            probabilities = {cid: randomizer.choice([0, 0.2, 0.3, 0.55, 0.85, 1]) for cid in ids}
            randomizer.shuffle(ids)
            record = audit_record([(cid, j + 1, randomizer.random()) for j, cid in enumerate(ids)], probabilities, expected="c0")
            record["sample_index"] = i
            randomizer.shuffle(ids)
            record["per_query_traces"].append({"query": "q2", "trace": [
                {"case_id": cid, "case_title": cid, "rank": j + 1, "score": randomizer.random()}
                for j, cid in enumerate(ids[:8])
            ]})
            records.append(record)
        artifact = analyse_saved_results(saved_artifact(records))
        from evaluate_laya import FUSIONS
        for row in artifact["threshold_search"]["results"]:
            hits, kept_count = 0, 0
            for record in records:
                kept = {j["case_id"] for j in record["laya_judgments"] if j["related_probability"] >= row["threshold"]}
                kept_count += len(kept)
                traces = [[RetrievedCase(**case) for case in item["trace"] if case["case_id"] in kept]
                          for item in record["per_query_traces"]]
                hits += any(case.case_id == "c0" for case in FUSIONS[row["fusion_method"]](traces, 10))
            self.assertEqual(row["hits_at_10"], hits)
            self.assertAlmostEqual(row["average_kept_candidates"], kept_count / len(records))

    def test_search_selects_score_when_it_has_higher_recall(self):
        cases = [(f"bad{i}", i + 1, 0.1) for i in range(10)] + [("target", 11, 0.99)]
        record = audit_record(cases, {cid: 0.9 for cid, _, _ in cases})
        result = analyse_saved_results(saved_artifact([record]))["threshold_search"]
        self.assertEqual(result["best"]["fusion_method"], "score")
        self.assertEqual(result["best"]["recall_at_10"], 1)
        self.assertEqual(result["best_by_fusion"]["round_robin"]["recall_at_10"], 0)

    def test_offline_cli_needs_neither_config_nor_models_and_keeps_original_metrics(self):
        artifact = saved_artifact([audit_record([("target", 1, 0.9)], {"target": 1})])
        artifact["metrics"] = {"recall_at_10": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.json", Path(directory) / "output.json"
            source.write_text(json.dumps(artifact))
            with patch.dict("sys.modules", {"laya": None}), \
                 patch("evaluate_laya.SearchRetriever", side_effect=AssertionError("no retrieval")):
                status = main(["analyze", "--input", str(source), "--output", str(output), "--config", "does-not-exist.json"])
            self.assertEqual(status, 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["metrics"], artifact["metrics"])
            self.assertEqual(result["threshold_search"]["best"]["recall_at_10"], 1)
            self.assertNotIn("threshold_search", json.loads(source.read_text()))

    def test_empty_or_unavailable_probabilities_do_not_report_a_fake_optimum(self):
        for records in ([], [audit_record([("target", 1, 0.9)], {})]):
            result = analyse_saved_results(saved_artifact(records))
            self.assertIsNone(result["threshold_search"]["best"])
            self.assertEqual(result["threshold_search"]["evaluable_samples"], 0)


if __name__ == "__main__":
    unittest.main()
