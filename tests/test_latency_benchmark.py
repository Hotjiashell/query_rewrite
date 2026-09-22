import json
import tempfile
import unittest
from pathlib import Path

from evaluate import DialogueSample, GeneratedQueryRecord, RetrievalEvaluator
from gen_query import LLMConfig, QueryGenerator
from latency_benchmark import (
    BenchmarkConfig,
    SerialLatencyBenchmark,
    build_benchmark_artifact,
    parse_args,
    resolve_config,
)
from rerank_cases import CaseReranker, RerankConfig


class _Generator(QueryGenerator):
    def generate(self, dialogue):
        return f"query for {dialogue}"


class _Retriever:
    def retrieve(self, query):
        return {
            "retrieval_result": {
                "top1": {"case_id": "KT2", "case_title": "其他案例", "score": 0.9},
                "top2": {"case_id": "KT1", "case_title": "目标案例", "score": 0.8},
            }
        }


class _Completions:
    def create(self, **kwargs):
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {
                                    "content": (
                                        '{"ranking": ['
                                        '{"case_id":"KT1","relevance":5,"reason":"相关"},'
                                        '{"case_id":"KT2","relevance":1,"reason":"无关"}'
                                        ']}'
                                    )
                                },
                            )()
                        },
                    )()
                ]
            },
        )()


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})()


class _SerialMultiRetriever:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []

    def retrieve(self, query):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(query)
        response = {
            "retrieval_result": {
                "top1": {"case_id": query, "case_title": query},
            }
        }
        self.active -= 1
        return response


class LatencyBenchmarkTests(unittest.TestCase):
    def test_multi_query_retrieval_can_be_forced_to_serial(self):
        retriever = _SerialMultiRetriever()
        evaluator = RetrievalEvaluator(retriever, parallel_queries=False)
        record = GeneratedQueryRecord(
            sample_index=0,
            call_sno="1",
            expected_case_id="q2",
            query="q1",
            status="success",
            error=None,
            chat_content="dialogue",
            queries=["q1", "q2"],
        )

        result = evaluator.evaluate_query(record)

        self.assertEqual(retriever.calls, ["q1", "q2"])
        self.assertEqual(retriever.max_active, 1)
        self.assertEqual(result["matched_rank"], 2)

    def test_benchmark_records_cumulative_timings_and_both_recalls(self):
        llm = LLMConfig("http://model", "model", "key")
        reranker = CaseReranker(
            _Client(),
            RerankConfig("", "", llm, concurrency=1),
        )
        samples = [DialogueSample(0, "1", "dialogue", "KT1")]
        records = SerialLatencyBenchmark(
            _Generator(),
            RetrievalEvaluator(_Retriever(), parallel_queries=False),
            reranker,
            samples,
        ).run()

        record = records[0]
        self.assertGreaterEqual(record["time_to_retrieval_sec"], 0)
        self.assertGreaterEqual(record["time_to_rerank_sec"], record["time_to_retrieval_sec"])
        self.assertEqual(record["matched_rank"], 2)
        self.assertEqual(record["reranked_matched_rank"], 1)
        config = BenchmarkConfig(
            input_path="input.json",
            output_path="output.json",
            test_num=1,
            llm=llm,
            method="baseline",
            prompt_file=None,
            retrieval_url="http://retrieval",
            retrieval_timeout=30,
            fusion_method="round_robin",
            top_k=10,
            candidate_limit=0,
        )
        artifact = build_benchmark_artifact(records, config=config, requested_test_num=1)
        self.assertEqual(artifact["metrics"]["retrieval"]["recall_at_1"], 0.0)
        self.assertEqual(artifact["metrics"]["rerank"]["recall_at_1"], 1.0)
        self.assertEqual(artifact["metrics"]["latency"]["time_to_rerank"]["completed_samples"], 1)

    def test_config_reads_test_num_and_benchmark_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "llm": {
                            "base_url": "http://model",
                            "model_name": "model",
                            "api_key": "key",
                        },
                        "query_generation": {
                            "input_path": "dialogs.json",
                            "method": "method_v1",
                        },
                        "retrieval": {"url": "http://retrieval"},
                        "benchmark": {
                            "test_num": 7,
                            "output_path": "results/latency.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = resolve_config(parse_args(["--config", str(config_path)]))

        self.assertEqual(config.test_num, 7)
        self.assertEqual(config.output_path, "results/latency.json")
        self.assertEqual(config.method, "method_v1")


if __name__ == "__main__":
    unittest.main()
