import json
import tempfile
import unittest
from pathlib import Path

from check_gt_retrieval import check_gt_retrieval, main


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        if query == "接口失败":
            raise TimeoutError("timed out")
        if query == "未命中":
            return {"retrieval_result": {"top1": {"case_id": "OTHER", "case_title": query}}}
        return {"retrieval_result": {"top12": {"case_id": "A", "case_title": query}}}


class CheckGTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.input = Path(self.temp.name) / "dialogs.json"
        self.cases = Path(self.temp.name) / "cases.json"
        self.input.write_text(json.dumps([
            {"caseID": "A", "call_sno": "1"}, {"caseID": "A", "call_sno": "2"},
            {"caseID": "B"}, {"caseID": "C"}, {"caseID": "D"}, {"caseID": "E"}, {},
        ]))
        self.cases.write_text(json.dumps({
            "A": {"case_name": "命中"}, "B": {"case_name": "未命中"},
            "D": {"case_name": "接口失败"}, "E": {"case_name": "   "},
        }))

    def test_deduplication_and_distinct_failure_categories(self):
        retriever = FakeRetriever()
        report = check_gt_retrieval(self.input, self.cases, retriever, concurrency=2)
        self.assertCountEqual(retriever.calls, ["命中", "未命中", "接口失败"])
        self.assertEqual([r["expected_case_id"] for r in report["misses"]], ["B"])
        self.assertEqual(report["records"][0]["sample_count"], 2)
        self.assertEqual(report["records"][0]["matched_rank"], 12)
        self.assertEqual(report["records"][0]["gt_case_title"], "命中")
        self.assertEqual(report["summary"]["missing_title_cases"], 2)
        self.assertEqual(report["summary"]["retrieval_failed_cases"], 1)
        self.assertEqual(report["summary"]["invalid_samples"], 1)
        self.assertEqual(report["summary"]["hit_rate_among_successful"], 0.5)

    def test_cutoff_uses_case_id_and_retains_full_trace(self):
        report = check_gt_retrieval(self.input, self.cases, FakeRetriever(), cutoff=10)
        self.assertEqual([r["expected_case_id"] for r in report["misses"]], ["A", "B"])
        self.assertEqual(report["summary"]["miss_samples"], 3)
        self.assertEqual(report["misses"][0]["retrieval_trace"][0]["rank"], 12)

    def test_cli_refuses_to_overwrite_source(self):
        self.assertEqual(main(["--input", str(self.input), "--case-summary", str(self.cases),
                               "--output", str(self.input)]), 2)


if __name__ == "__main__":
    unittest.main()
