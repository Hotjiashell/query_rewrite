import json
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    QUERY_ARTIFACT_TYPE,
    DialogueSample,
    GeneratedQueryRecord,
    QueryGenerationRunner,
    RetrievalEvaluator,
    RetrievedCase,
    build_query_artifact,
    build_retrieval_artifact,
    calculate_metrics,
    extract_retrieval_trace,
    fuse_by_score,
    fuse_round_robin,
    load_dialogue_samples,
    load_generated_query_records,
    parse_args,
    resolve_query_generation_config,
    resolve_retrieval_config,
    write_json_atomically,
)
from gen_query import MultiQueryGenerator, QueryGenerator


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
                    method="method_v1",
                ),
                query_path,
            )
            persisted_queries = json.loads(query_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted_queries["artifact_type"], QUERY_ARTIFACT_TYPE)
            self.assertEqual(persisted_queries["configuration"]["method"], "method_v1")
            self.assertEqual(persisted_queries["configuration"]["generator"], "method_v1")
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
                {"rank": 1, "case_id": "KT1", "case_title": "第一名", "score": None},
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
                            "method": "method_v1",
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
        self.assertEqual(generation_config.method, "method_v1")
        self.assertEqual(generation_config.concurrency, 5)
        self.assertEqual(retrieval_config.input_path, "queries.json")
        self.assertEqual(retrieval_config.output_path, "other-results.json")
        self.assertEqual(retrieval_config.url, "http://retrieval")
        self.assertEqual(retrieval_config.timeout, 12.0)
        self.assertEqual(retrieval_config.concurrency, 8)

    def test_all_mode_accepts_independent_concurrency_overrides(self):
        args = parse_args(
            [
                "all",
                "--query-concurrency",
                "4",
                "--retrieval-concurrency",
                "9",
            ]
        )
        self.assertEqual(args.stage, "all")
        self.assertEqual(args.query_concurrency, 4)
        self.assertEqual(args.retrieval_concurrency, 9)

    def test_method_cli_override_takes_precedence_over_config(self):
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
                            "method": "baseline",
                            "input_path": "dialogs.json",
                            "output_path": "queries.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                ["generate", "--config", str(config_path), "--method", "method_v1"]
            )
            resolved = resolve_query_generation_config(args)

        self.assertEqual(resolved.method, "method_v1")

    def test_method_cli_accepts_prompt_constant_alias(self):
        args = parse_args(["generate", "--method", "METHOD_V1_PROMPT"])

        self.assertEqual(args.method, "method_v1")

    def test_custom_method_resolves_prompt_file_from_config_and_cli(self):
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
                            "method": "custom",
                            "input_path": "dialogs.json",
                            "output_path": "queries.json",
                            "prompt_file": "prompts/from_config.txt",
                        },
                    }
                ),
                encoding="utf-8",
            )
            from_config = resolve_query_generation_config(
                parse_args(["generate", "--config", str(config_path)])
            )
            overridden = resolve_query_generation_config(
                parse_args(
                    [
                        "generate",
                        "--config",
                        str(config_path),
                        "--prompt-file",
                        "prompts/from_cli.txt",
                    ]
                )
            )

        self.assertEqual(from_config.method, "custom")
        self.assertEqual(from_config.prompt_file, "prompts/from_config.txt")
        self.assertEqual(overridden.prompt_file, "prompts/from_cli.txt")

    def test_custom_method_without_prompt_file_raises(self):
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
                            "method": "custom",
                            "input_path": "dialogs.json",
                            "output_path": "queries.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(["generate", "--config", str(config_path)])
            with self.assertRaises(ValueError):
                resolve_query_generation_config(args)


def _case(case_id, rank, score=None, title=""):
    return RetrievedCase(rank=rank, case_id=case_id, case_title=title, score=score)


class FusionTests(unittest.TestCase):
    def test_round_robin_interleaves_and_dedupes(self):
        traces = [
            [_case("A", 1), _case("B", 2), _case("C", 3)],
            [_case("B", 1), _case("D", 2)],
        ]
        fused = fuse_round_robin(traces, top_k=10)

        # Round 1: trace0 contributes A, trace1 contributes B.
        # Round 2: trace0 skips the already-seen B and contributes C, trace1 contributes D.
        self.assertEqual([case.case_id for case in fused], ["A", "B", "C", "D"])
        self.assertEqual([case.rank for case in fused], [1, 2, 3, 4])

    def test_round_robin_respects_top_k(self):
        traces = [[_case("A", 1), _case("B", 2)], [_case("C", 1), _case("D", 2)]]
        fused = fuse_round_robin(traces, top_k=2)

        self.assertEqual([case.case_id for case in fused], ["A", "C"])

    def test_score_fusion_dedupes_keeping_highest_score_and_sorts_desc(self):
        traces = [
            [_case("A", 1, score=0.5), _case("B", 2, score=0.9)],
            [_case("A", 1, score=0.8), _case("C", 2, score=0.4)],
        ]
        fused = fuse_by_score(traces, top_k=10)

        self.assertEqual([case.case_id for case in fused], ["B", "A", "C"])
        self.assertEqual([case.rank for case in fused], [1, 2, 3])
        self.assertEqual(next(c for c in fused if c.case_id == "A").score, 0.8)

    def test_score_fusion_treats_missing_score_as_lowest(self):
        traces = [[_case("A", 1, score=None), _case("B", 2, score=0.1)]]
        fused = fuse_by_score(traces, top_k=10)

        self.assertEqual([case.case_id for case in fused], ["B", "A"])

    def test_score_fusion_respects_top_k(self):
        traces = [[_case("A", 1, score=0.9), _case("B", 2, score=0.5), _case("C", 3, score=0.1)]]
        fused = fuse_by_score(traces, top_k=2)

        self.assertEqual([case.case_id for case in fused], ["A", "B"])


class _MultiQueryRetriever:
    """Retriever whose response depends on which of the known queries was sent."""

    def __init__(self, responses, failing_queries=()):
        self._responses = responses
        self._failing_queries = set(failing_queries)

    def retrieve(self, query):
        if query in self._failing_queries:
            raise RuntimeError(f"retrieval failed for {query}")
        return self._responses[query]


def _retrieval_response(*cases):
    return {
        "retrieval_result": {
            f"top{rank}": {"case_id": case_id, "case_title": title, "score": score}
            for rank, (case_id, title, score) in enumerate(cases, start=1)
        }
    }


class MultiQueryRetrievalTests(unittest.TestCase):
    def test_evaluate_query_fuses_multiple_queries_with_round_robin(self):
        retriever = _MultiQueryRetriever(
            {
                "q1": _retrieval_response(("KT1", "案例1", 0.9), ("KT2", "案例2", 0.8)),
                "q2": _retrieval_response(("KT2", "案例2", 0.7), ("KT3", "案例3", 0.6)),
            }
        )
        record = GeneratedQueryRecord(
            sample_index=0,
            call_sno="1",
            expected_case_id="KT3",
            query="q1",
            status="success",
            error=None,
            queries=["q1", "q2"],
        )
        result = RetrievalEvaluator(retriever, fusion_method="round_robin", top_k=10).evaluate_query(record)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["retrieval_status"], "success")
        self.assertEqual([c["case_id"] for c in result["retrieval_trace"]], ["KT1", "KT2", "KT3"])
        self.assertEqual(result["matched_rank"], 3)

    def test_evaluate_query_fuses_multiple_queries_with_score(self):
        retriever = _MultiQueryRetriever(
            {
                "q1": _retrieval_response(("KT1", "案例1", 0.5)),
                "q2": _retrieval_response(("KT2", "案例2", 0.9)),
            }
        )
        record = GeneratedQueryRecord(
            sample_index=0,
            call_sno="1",
            expected_case_id="KT2",
            query="q1",
            status="success",
            error=None,
            queries=["q1", "q2"],
        )
        result = RetrievalEvaluator(retriever, fusion_method="score", top_k=10).evaluate_query(record)

        self.assertEqual([c["case_id"] for c in result["retrieval_trace"]], ["KT2", "KT1"])
        self.assertEqual(result["matched_rank"], 1)

    def test_evaluate_query_tolerates_partial_query_failure(self):
        retriever = _MultiQueryRetriever(
            {"q1": _retrieval_response(("KT1", "案例1", 0.5)), "q2": None},
            failing_queries=["q2"],
        )
        record = GeneratedQueryRecord(
            sample_index=0,
            call_sno="1",
            expected_case_id="KT1",
            query="q1",
            status="success",
            error=None,
            queries=["q1", "q2"],
        )
        result = RetrievalEvaluator(retriever).evaluate_query(record)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["retrieval_status"], "partial_success")
        self.assertIn("q2", result["retrieval_error"])
        self.assertEqual(result["matched_rank"], 1)

    def test_evaluate_query_fails_when_all_queries_fail(self):
        retriever = _MultiQueryRetriever({}, failing_queries=["q1", "q2"])
        record = GeneratedQueryRecord(
            sample_index=0,
            call_sno="1",
            expected_case_id="KT1",
            query="q1",
            status="success",
            error=None,
            queries=["q1", "q2"],
        )
        result = RetrievalEvaluator(retriever).evaluate_query(record)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["retrieval_status"], "failed")

    def test_single_query_record_behaves_like_before(self):
        retriever = _MultiQueryRetriever({"q1": _retrieval_response(("KT1", "案例1", 0.5))})
        record = GeneratedQueryRecord(
            sample_index=0,
            call_sno="1",
            expected_case_id="KT1",
            query="q1",
            status="success",
            error=None,
            queries=None,
        )
        result = RetrievalEvaluator(retriever).evaluate_query(record)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["retrieval_status"], "success")
        self.assertEqual(result["matched_rank"], 1)

    def test_generation_runner_uses_multi_query_generator_when_available(self):
        class _StaticMultiGenerator(MultiQueryGenerator):
            def generate_queries(self, dialogue):
                return ["q1", "q2"]

        sample = DialogueSample(0, "1", "dialogue", "KT1")
        record = QueryGenerationRunner(_StaticMultiGenerator()).generate_sample(sample)

        self.assertEqual(record.status, "success")
        self.assertEqual(record.query, "q1")
        self.assertEqual(record.queries, ["q1", "q2"])


class RetrievalConfigFusionTests(unittest.TestCase):
    def test_resolve_retrieval_config_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "retrieval": {
                            "input_path": "queries.json",
                            "output_path": "results.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(["retrieve", "--config", str(config_path)])
            config = resolve_retrieval_config(args)

        self.assertEqual(config.fusion_method, "round_robin")
        self.assertEqual(config.top_k, 10)

    def test_resolve_retrieval_config_reads_config_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "retrieval": {
                            "input_path": "queries.json",
                            "output_path": "results.json",
                            "fusion_method": "score",
                            "top_k": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            from_config = resolve_retrieval_config(
                parse_args(["retrieve", "--config", str(config_path)])
            )
            overridden = resolve_retrieval_config(
                parse_args(
                    [
                        "retrieve",
                        "--config",
                        str(config_path),
                        "--fusion-method",
                        "round_robin",
                        "--top-k",
                        "3",
                    ]
                )
            )

        self.assertEqual(from_config.fusion_method, "score")
        self.assertEqual(from_config.top_k, 5)
        self.assertEqual(overridden.fusion_method, "round_robin")
        self.assertEqual(overridden.top_k, 3)

    def test_resolve_retrieval_config_rejects_unknown_fusion_method(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "retrieval": {
                            "input_path": "queries.json",
                            "output_path": "results.json",
                            "fusion_method": "unknown",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(["retrieve", "--config", str(config_path)])
            with self.assertRaises(ValueError):
                resolve_retrieval_config(args)


if __name__ == "__main__":
    unittest.main()
