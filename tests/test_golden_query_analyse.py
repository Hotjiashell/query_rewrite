import unittest

from evaluate import GeneratedQueryRecord
from gen_query import LLMConfig
from get_goldenquery.analyse import AnalysisConfig, GoldenMissAnalyser


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

    def retrieve(self, query):
        return self.responses[query]


def _response(*case_ids):
    return {
        "retrieval_result": {
            f"top{rank}": {
                "case_id": case_id,
                "case_title": f"标题 {case_id}",
                "content": f"不应传给模型的正文 {case_id}",
            }
            for rank, case_id in enumerate(case_ids, start=1)
        }
    }


class GoldenMissAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config = AnalysisConfig(
            golden_path="golden.json",
            query_path="queries.json",
            output_path="analysis.json",
            llm=LLMConfig("http://model", "test-model", "key"),
            retrieval_url="http://retrieval",
            timeout=10,
            top_k=10,
            concurrency=1,
        )
        self.golden = {
            "sample_index": 0,
            "call_sno": "call-1",
            "expected_case_id": "KT1",
            "golden_case": {"case_id": "KT1", "title": "企业微信会议室预订"},
            "final_query": "企业微信会议室预订流程",
            "matched_rank": 1,
            "status": "success",
        }
        self.ordinary = GeneratedQueryRecord(0, "call-1", "KT1", "会议预订", "success", None)

    def test_analyses_an_ordinary_query_miss_with_title_only_evidence(self):
        client = _Client([
            '{"reason":"普通 query 缺少企业微信关键词，且会议词使结果偏向通用会议。",'
            '"missing_keywords":["企业微信"],"noise_keywords":["会议"]}'
        ])
        retriever = _Retriever({"会议预订": _response("KT2", "KT3", "KT4")})

        record = GoldenMissAnalyser(client, self.config, retriever).analyse_sample(self.golden, self.ordinary)

        self.assertEqual(record["status"], "success")
        self.assertIsNone(record["ordinary_matched_rank"])
        self.assertEqual(record["analysis"]["missing_keywords"], ["企业微信"])
        self.assertEqual(record["analysis"]["noise_keywords"], ["会议"])
        self.assertEqual(record["ordinary_top_10"][0]["case_title"], "标题 KT2")
        self.assertNotIn("content", record["ordinary_top_10"][0])
        prompt = client.chat.completions.calls[0]["messages"][0]["content"]
        self.assertIn("标题 KT2", prompt)
        self.assertNotIn("不应传给模型的正文", prompt)

    def test_skips_model_analysis_when_ordinary_query_already_retrieves_target(self):
        client = _Client([])
        retriever = _Retriever({"会议预订": _response("KT2", "KT1")})

        record = GoldenMissAnalyser(client, self.config, retriever).analyse_sample(self.golden, self.ordinary)

        self.assertEqual(record["status"], "skipped")
        self.assertEqual(record["ordinary_matched_rank"], 2)
        self.assertEqual(client.chat.completions.calls, [])

    def test_summary_counts_only_successful_golden_records_as_candidates(self):
        client = _Client(['{"reason":"缺少企业微信。","missing_keywords":["企业微信"],"noise_keywords":[]}'])
        retriever = _Retriever({"会议预订": _response("KT2")})
        invalid_golden = dict(self.golden, sample_index=1, status="failed")

        records, summary = GoldenMissAnalyser(client, self.config, retriever).analyse(
            {0: self.golden, 1: invalid_golden},
            [self.ordinary],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(summary["golden_successful_samples"], 1)
        self.assertEqual(summary["candidate_samples"], 1)
        self.assertEqual(summary["analysed_misses"], 1)


if __name__ == "__main__":
    unittest.main()
