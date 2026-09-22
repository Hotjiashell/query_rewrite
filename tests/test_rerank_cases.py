import json
import tempfile
import unittest
from pathlib import Path

from gen_query import LLMConfig
from rerank_cases import (
    RERANK_ARTIFACT_TYPE,
    CaseReranker,
    RerankConfig,
    build_rerank_artifact,
    build_rerank_prompt,
    calculate_rerank_metrics,
    load_retrieval_records,
    write_json_atomically,
)


class _Completions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        message = type("Message", (), {"content": response})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _Client:
    def __init__(self, responses):
        completions = _Completions(responses)
        self.chat = type("Chat", (), {"completions": completions})()


def _record(index=0, expected="KT1", trace=None):
    return {
        "sample_index": index,
        "expected_case_id": expected,
        "chat_content": "用户：电脑连不上网\n客服：请描述故障",
        "retrieval_status": "success",
        "retrieval_trace": trace
        or [
            {"rank": 1, "case_id": "KT2", "case_title": "会议预订", "score": 0.9},
            {"rank": 2, "case_id": "KT1", "case_title": "电脑无法联网", "score": 0.8},
        ],
        "status": "success",
    }


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.config = RerankConfig(
            input_path="results.json",
            output_path="reranked.json",
            llm=LLMConfig("http://model", "test-model", "key"),
            concurrency=1,
        )

    def test_prompt_contains_dialogue_and_title_but_not_unavailable_content(self):
        prompt = build_rerank_prompt(
            "用户：无法联网",
            [{"rank": 1, "case_id": "KT1", "case_title": "网络故障"}],
        )
        self.assertIn("用户：无法联网", prompt)
        self.assertIn("case_id: KT1", prompt)
        self.assertIn("case_title: 网络故障", prompt)
        self.assertNotIn("score", prompt)
        self.assertIn("ranking", prompt)

    def test_reranks_and_recalculates_match_rank(self):
        client = _Client(
            [
                '{"ranking": ['
                '{"case_id":"KT1","relevance":5,"reason":"直接对应联网故障"},'
                '{"case_id":"KT2","relevance":1,"reason":"业务无关"}'
                ']}'
            ]
        )
        result = CaseReranker(client, self.config).rerank([_record()])[0]

        self.assertEqual(result["rerank_status"], "success")
        self.assertEqual([item["case_id"] for item in result["reranked_trace"]], ["KT1", "KT2"])
        self.assertEqual(result["reranked_matched_rank"], 1)
        self.assertEqual(result["reranked_trace"][0]["original_rank"], 2)
        self.assertNotIn("score", result["reranked_trace"][0])
        self.assertNotIn("score", result["retrieval_trace"][0])
        self.assertEqual(client.chat.completions.calls[0]["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})

    def test_invalid_model_ranking_falls_back_without_stopping(self):
        client = _Client(['{"ranking": [{"case_id":"KT1","relevance":5,"reason":"只返回一个"}]}'])
        result = CaseReranker(client, self.config).rerank([_record()])[0]

        self.assertEqual(result["rerank_status"], "failed")
        self.assertIn("exactly 2", result["rerank_error"])
        self.assertEqual([item["case_id"] for item in result["reranked_trace"]], ["KT2", "KT1"])
        self.assertEqual(result["reranked_matched_rank"], 2)

    def test_parallel_results_are_restored_to_input_order_and_failures_are_isolated(self):
        client = _Client(
            [
                '{"ranking":[{"case_id":"KT1","relevance":5,"reason":"相关"},{"case_id":"KT2","relevance":1,"reason":"不相关"}]}',
                '{"ranking":[{"case_id":"KT1","relevance":5,"reason":"相关"},{"case_id":"KT2","relevance":1,"reason":"不相关"}]}',
            ]
        )
        records = [_record(0), _record(1)]
        results = CaseReranker(
            client,
            RerankConfig("results.json", "out.json", self.config.llm, concurrency=2),
        ).rerank(records)

        self.assertEqual([result["sample_index"] for result in results], [0, 1])
        self.assertEqual([result["rerank_status"] for result in results], ["success", "success"])

    def test_skips_failed_retrieval_without_model_call(self):
        client = _Client([])
        record = _record()
        record["retrieval_status"] = "failed"
        result = CaseReranker(client, self.config).rerank([record])[0]

        self.assertEqual(result["rerank_status"], "skipped")
        self.assertEqual(result["reranked_matched_rank"], 2)
        self.assertEqual(client.chat.completions.calls, [])

    def test_skips_failed_retrieval_even_when_trace_is_empty(self):
        client = _Client([])
        record = _record()
        record["retrieval_status"] = "failed"
        record["retrieval_trace"] = []
        result = CaseReranker(client, self.config).rerank([record])[0]

        self.assertEqual(result["rerank_status"], "skipped")
        self.assertEqual(result["reranked_trace"], [])
        self.assertEqual(client.chat.completions.calls, [])

    def test_metrics_include_fallback_traces(self):
        records = [
            {"rerank_status": "success", "reranked_matched_rank": 1},
            {"rerank_status": "failed", "reranked_matched_rank": 7},
            {"rerank_status": "skipped", "reranked_matched_rank": None},
        ]
        metrics = calculate_rerank_metrics(records)

        self.assertEqual(metrics["successful_reranks"], 1)
        self.assertEqual(metrics["failed_reranks"], 1)
        self.assertEqual(metrics["hits_at_1"], 1)
        self.assertEqual(metrics["hits_at_10"], 2)
        self.assertAlmostEqual(metrics["recall_at_10"], 2 / 3)

    def test_loads_and_writes_rerank_artifact_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "out.json"
            source.write_text(
                json.dumps({"artifact_type": "retrieval_evaluation", "records": [_record()]}),
                encoding="utf-8",
            )
            records = load_retrieval_records(source)
            artifact = build_rerank_artifact(
                records,
                input_path=source,
                model_name="test-model",
                concurrency=2,
                candidate_limit=0,
            )
            write_json_atomically(artifact, output)
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(persisted["artifact_type"], RERANK_ARTIFACT_TYPE)
        self.assertEqual(persisted["configuration"]["candidate_limit"], 0)


if __name__ == "__main__":
    unittest.main()
