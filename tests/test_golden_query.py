import json
import tempfile
import unittest
from pathlib import Path

from evaluate import DialogueSample
from gen_query import LLMConfig
from get_goldenquery.__main__ import (
    DEFAULT_GOLDEN_TOP_K,
    GoldenCase,
    GoldenQueryConfig,
    GoldenQueryRunner,
    calculate_summary,
    load_cases,
)


class _Completions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _Client:
    def __init__(self, responses):
        completions = _Completions(responses)
        self.chat = type("Chat", (), {"completions": completions})()


class _Retriever:
    def __init__(self, responses):
        self.responses = responses
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.responses[query]


def _response(*case_ids):
    return {
        "retrieval_result": {
            f"top{index}": {
                "case_id": case_id,
                "case_title": f"标题 {case_id}",
                "content": f"内容 {case_id}",
                "score": 1.0 / index,
            }
            for index, case_id in enumerate(case_ids, start=1)
        }
    }


class GoldenQueryTests(unittest.TestCase):
    def setUp(self):
        self.config = GoldenQueryConfig(
            dialogue_path="dialogues.json",
            case_path="cases.json",
            output_path="results.json",
            llm=LLMConfig("http://model", "model", "key"),
            retrieval_url="http://retrieval",
            timeout=10,
            max_retries=2,
            top_k=10,
            concurrency=1,
        )
        self.sample = DialogueSample(0, "call-1", "用户：无法连接网络", "KT1")
        self.case = GoldenCase("KT1", "网络无法连接")

    def test_first_query_hit_returns_title_only_trace(self):
        client = _Client(['{"query": "网络连接配置"}'])
        retriever = _Retriever({"网络连接配置": _response("KT1", "KT2")})

        record = GoldenQueryRunner(client, self.config, retriever).run_sample(self.sample, self.case)

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["final_query"], "网络连接配置")
        self.assertEqual(record["matched_rank"], 1)
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(record["attempts"][0]["retrieval_trace"][0]["case_title"], "标题 KT1")
        self.assertNotIn("content", record["attempts"][0]["retrieval_trace"][0])

    def test_retry_receives_actual_top_results_and_can_hit(self):
        client = _Client(['{"query": "网络问题"}', '{"query": "网络连接配置"}'])
        retriever = _Retriever({
            "网络问题": _response("KT2", "KT3"),
            "网络连接配置": _response("KT2", "KT1"),
        })

        record = GoldenQueryRunner(client, self.config, retriever).run_sample(self.sample, self.case)

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["matched_rank"], 2)
        self.assertEqual([attempt["query"] for attempt in record["attempts"]], ["网络问题", "网络连接配置"])
        retry_prompt = client.chat.completions.calls[1]["messages"][0]["content"]
        initial_prompt = client.chat.completions.calls[0]["messages"][0]["content"]
        self.assertIn("标题 KT2", retry_prompt)
        self.assertIn("标题 KT3", retry_prompt)
        self.assertIn("网络问题", retry_prompt)
        self.assertNotIn("内容 KT2", retry_prompt)
        self.assertNotIn("内容 KT3", retry_prompt)
        self.assertNotIn("内容 KT1", initial_prompt)

    def test_load_cases_accepts_mapping_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({"KT1": {"case_name": "标题"}}), encoding="utf-8")
            cases = load_cases(path)

        self.assertEqual(cases["KT1"], GoldenCase("KT1", "标题"))

    def test_summary_reports_recall_at_all_requested_cutoffs(self):
        summary = calculate_summary([
            {"status": "success", "matched_rank": 1},
            {"status": "success", "matched_rank": 3},
            {"status": "failed", "matched_rank": 7},
            {"status": "failed", "matched_rank": None},
        ])

        self.assertEqual(summary["successful_samples"], 2)
        self.assertEqual(summary["hits_at_1"], 1)
        self.assertEqual(summary["hits_at_3"], 2)
        self.assertEqual(summary["hits_at_5"], 2)
        self.assertEqual(summary["hits_at_10"], 3)
        self.assertEqual(summary["recall_at_10"], 0.75)

    def test_default_feedback_and_pass_cutoff_is_top_five(self):
        self.assertEqual(DEFAULT_GOLDEN_TOP_K, 5)

        config = GoldenQueryConfig(
            dialogue_path="dialogues.json",
            case_path="cases.json",
            output_path="results.json",
            llm=LLMConfig("http://model", "model", "key"),
            retrieval_url="http://retrieval",
            timeout=10,
            max_retries=0,
            top_k=DEFAULT_GOLDEN_TOP_K,
            concurrency=1,
        )
        client = _Client(['{"query": "网络问题"}'])
        retriever = _Retriever({"网络问题": _response("KT2", "KT3", "KT4", "KT5", "KT6", "KT7", "KT1")})

        record = GoldenQueryRunner(client, config, retriever).run_sample(self.sample, self.case)

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["matched_rank"], 7)
        self.assertEqual(len(record["attempts"][0]["retrieval_trace"]), 5)


if __name__ == "__main__":
    unittest.main()
