import json
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    DialogueSample,
    Evaluator,
    build_artifact,
    calculate_metrics,
    extract_retrieval_trace,
    load_dialogue_samples,
    parse_args,
    resolve_run_config,
    write_json_atomically,
)
from gen_query import QueryGenerator


class _StaticGenerator(QueryGenerator):
    def generate(self, dialogue: str) -> str:
        return f"query: {dialogue}"


class _StaticRetriever:
    def retrieve(self, query: str):
        if "bad" in query:
            raise RuntimeError("service unavailable")
        return {
            "retrieval_result": {
                "top10": {"case_id": "KT10", "case_title": "第十名", "content": "hidden"},
                "top2": {"case_id": "KT2", "case_title": "第二名", "content": "hidden"},
                "top1": {"case_id": "KT1", "case_title": "第一名", "content": "hidden"},
                "top7": {"case_id": "KT7", "case_title": "第七名", "content": "hidden"},
            }
        }


class EvaluationTests(unittest.TestCase):
    def test_extracts_all_top_keys_in_numeric_order(self):
        trace = extract_retrieval_trace(_StaticRetriever().retrieve("normal"))
        self.assertEqual([case.rank for case in trace], [1, 2, 7, 10])
        self.assertEqual([case.case_id for case in trace], ["KT1", "KT2", "KT7", "KT10"])
        self.assertEqual(trace[0].case_title, "第一名")

    def test_metrics_use_all_inputs_and_failures_are_misses(self):
        samples = [
            DialogueSample(0, "1", "first", "KT1"),
            DialogueSample(1, "2", "second", "KT7"),
            DialogueSample(2, "3", "bad", "KT1"),
        ]
        records = Evaluator(_StaticGenerator(), _StaticRetriever()).evaluate(samples, concurrency=2)
        metrics = calculate_metrics(records)

        self.assertEqual([record["status"] for record in records], ["success", "success", "failed"])
        self.assertEqual(metrics["total_samples"], 3)
        self.assertEqual(metrics["successful_samples"], 2)
        self.assertEqual(metrics["hits_at_1"], 1)
        self.assertEqual(metrics["hits_at_3"], 1)
        self.assertEqual(metrics["hits_at_10"], 2)
        self.assertAlmostEqual(metrics["recall_at_10"], 2 / 3)
        self.assertEqual(records[0]["retrieval_trace"][0], {"rank": 1, "case_id": "KT1", "case_title": "第一名"})

    def test_load_and_write_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "nested" / "result.json"
            input_path.write_text(
                json.dumps([{"call_sno": "001", "chat_content": "dialogue", "caseID": "KT1"}]),
                encoding="utf-8",
            )
            samples = load_dialogue_samples(input_path)
            self.assertEqual(samples[0].expected_case_id, "KT1")
            records = Evaluator(_StaticGenerator(), _StaticRetriever()).evaluate(samples)
            artifact = build_artifact(
                records,
                input_path=input_path,
                model_name="test-model",
                retrieval_url="http://retriever",
                concurrency=1,
            )
            write_json_atomically(artifact, output_path)
            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["metrics"]["hits_at_1"], 1)
            self.assertNotIn("content", persisted["records"][0]["retrieval_trace"][0])

    def test_config_file_supplies_run_settings_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "llm": {
                            "base_url": "http://model",
                            "model_name": "model-a",
                            "api_key": "test-key",
                        },
                        "retrieval": {"url": "http://retrieval", "timeout": 12},
                        "evaluation": {
                            "input_path": "input.json",
                            "output_path": "output.json",
                            "concurrency": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--model",
                    "model-from-cli",
                    "--concurrency",
                    "5",
                ]
            )
            run_config = resolve_run_config(args)

            self.assertEqual(run_config.llm.base_url, "http://model")
            self.assertEqual(run_config.llm.model_name, "model-from-cli")
            self.assertEqual(run_config.input_path, "input.json")
            self.assertEqual(run_config.output_path, "output.json")
            self.assertEqual(run_config.retrieval_url, "http://retrieval")
            self.assertEqual(run_config.retrieval_timeout, 12.0)
            self.assertEqual(run_config.concurrency, 5)


if __name__ == "__main__":
    unittest.main()
