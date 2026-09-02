import json
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    QUERY_ARTIFACT_TYPE,
    DialogueSample,
    QueryGenerationRunner,
    RetrievalEvaluator,
    build_query_artifact,
    build_retrieval_artifact,
    calculate_metrics,
    extract_retrieval_trace,
    load_dialogue_samples,
    load_generated_query_records,
    parse_args,
    resolve_query_generation_config,
    resolve_retrieval_config,
    write_json_atomically,
)
from gen_query import QueryGenerator


class _StaticGenerator(QueryGenerator):
    def generate(self, dialogue: str) -> str:
        if dialogue == "generation_error":
            raise RuntimeError("model unavailable")
        return f"query: {dialogue}"


class _StaticRetriever:
    def retrieve(self, query: str):
        if "retrieval_error" in query:
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

    def test_two_stages_persist_queries_and_isolate_failures(self):
        samples = [
            DialogueSample(0, "1", "first", "KT1"),
            DialogueSample(1, "2", "generation_error", "KT7"),
            DialogueSample(2, "3", "retrieval_error", "KT1"),
        ]
        generated = QueryGenerationRunner(_StaticGenerator()).generate(samples, concurrency=2)
        self.assertEqual([record.status for record in generated], ["success", "failed", "success"])
        self.assertEqual(generated[1].error, "RuntimeError: model unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "queries.json"
            result_path = root / "results" / "retrieval.json"
            write_json_atomically(
                build_query_artifact(
                    generated,
                    input_path="dialogs.json",
                    model_name="test-model",
                    concurrency=2,
                ),
                query_path,
            )
            persisted_queries = json.loads(query_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted_queries["artifact_type"], QUERY_ARTIFACT_TYPE)
            self.assertNotIn("dialogue", persisted_queries["records"][0])

            loaded_queries = load_generated_query_records(query_path)
            records = RetrievalEvaluator(_StaticRetriever()).evaluate(loaded_queries, concurrency=3)
            self.assertEqual([record["status"] for record in records], ["success", "failed", "failed"])
            self.assertEqual([record["retrieval_status"] for record in records], ["success", "skipped", "failed"])
            self.assertIn("model unavailable", records[1]["query_error"])
            self.assertIn("not attempted", records[1]["retrieval_error"])
            self.assertIn("service unavailable", records[2]["retrieval_error"])
            self.assertEqual(
                records[0]["retrieval_trace"][0],
                {"rank": 1, "case_id": "KT1", "case_title": "第一名"},
            )

            artifact = build_retrieval_artifact(
                records,
                input_path=query_path,
                retrieval_url="http://retriever",
                timeout=12,
                concurrency=3,
            )
            write_json_atomically(artifact, result_path)
            persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted_result["metrics"]["total_samples"], 3)
            self.assertEqual(persisted_result["metrics"]["hits_at_1"], 1)
            self.assertNotIn("content", persisted_result["records"][0]["retrieval_trace"][0])

    def test_malformed_source_sample_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(
                json.dumps(
                    [
                        {"call_sno": "001", "chat_content": "valid", "caseID": "KT1"},
                        "not an object",
                        {"call_sno": "003", "chat_content": "", "caseID": "KT3"},
                    ]
                ),
                encoding="utf-8",
            )
            records = QueryGenerationRunner(_StaticGenerator()).generate(load_dialogue_samples(input_path))

        self.assertEqual([record.status for record in records], ["success", "failed", "failed"])
        self.assertIn("sample must be a JSON object", records[1].error)
        self.assertIn("chat_content", records[2].error)

    def test_malformed_query_record_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            query_path = Path(directory) / "queries.json"
            query_path.write_text(
                json.dumps(
                    {
                        "artifact_type": QUERY_ARTIFACT_TYPE,
                        "records": [
                            {
                                "sample_index": 0,
                                "expected_case_id": "KT1",
                                "query": "normal",
                                "status": "success",
                            },
                            "invalid record",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            records = RetrievalEvaluator(_StaticRetriever()).evaluate(load_generated_query_records(query_path))

        self.assertEqual([record["retrieval_status"] for record in records], ["success", "skipped"])
        self.assertIn("InvalidQueryArtifact", records[1]["query_error"])

    def test_metrics_use_all_query_records_as_the_denominator(self):
        records = [
            {"status": "success", "matched_rank": 1},
            {"status": "success", "matched_rank": 7},
            {"status": "failed", "matched_rank": None},
        ]
        metrics = calculate_metrics(records)

        self.assertEqual(metrics["total_samples"], 3)
        self.assertEqual(metrics["successful_samples"], 2)
        self.assertEqual(metrics["hits_at_1"], 1)
        self.assertEqual(metrics["hits_at_3"], 1)
        self.assertEqual(metrics["hits_at_10"], 2)
        self.assertAlmostEqual(metrics["recall_at_10"], 2 / 3)

    def test_each_stage_resolves_its_own_concurrency_and_paths(self):
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
                        "query_generation": {
                            "input_path": "dialogs.json",
                            "output_path": "queries.json",
                            "concurrency": 3,
                        },
                        "retrieval": {
                            "input_path": "queries.json",
                            "output_path": "results.json",
                            "url": "http://retrieval",
                            "timeout": 12,
                            "concurrency": 6,
                        },
                    }
                ),
                encoding="utf-8",
            )
            generation_args = parse_args(
                ["generate", "--config", str(config_path), "--model", "model-from-cli", "--concurrency", "5"]
            )
            retrieval_args = parse_args(
                ["retrieve", "--config", str(config_path), "--output", "other-results.json", "--concurrency", "8"]
            )
            generation_config = resolve_query_generation_config(generation_args)
            retrieval_config = resolve_retrieval_config(retrieval_args)

        self.assertEqual(generation_config.llm.base_url, "http://model")
        self.assertEqual(generation_config.llm.model_name, "model-from-cli")
        self.assertEqual(generation_config.input_path, "dialogs.json")
        self.assertEqual(generation_config.output_path, "queries.json")
        self.assertEqual(generation_config.concurrency, 5)
        self.assertEqual(retrieval_config.input_path, "queries.json")
        self.assertEqual(retrieval_config.output_path, "other-results.json")
        self.assertEqual(retrieval_config.url, "http://retrieval")
        self.assertEqual(retrieval_config.timeout, 12.0)
        self.assertEqual(retrieval_config.concurrency, 8)


if __name__ == "__main__":
    unittest.main()
