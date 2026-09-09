import json
import tempfile
import unittest
from pathlib import Path

from compare_results import compare_result_artifacts


def _artifact(records):
    return {"artifact_type": "retrieval_evaluation", "records": records}


class CompareResultsTests(unittest.TestCase):
    def test_finds_first_hit_second_miss_at_requested_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.json"
            second_path = root / "second.json"
            first_path.write_text(
                json.dumps(
                    _artifact(
                        [
                            {
                                "sample_index": 0,
                                "call_sno": "a",
                                "expected_case_id": "C0",
                                "query": "first query",
                                "status": "success",
                                "retrieval_status": "success",
                                "matched_rank": 3,
                                "retrieval_trace": [
                                    {"rank": 3, "case_id": "C0", "case_title": "target", "content": "secret"}
                                ],
                            },
                            {
                                "sample_index": 1,
                                "call_sno": "b",
                                "expected_case_id": "C1",
                                "query": "both hit",
                                "status": "success",
                                "retrieval_status": "success",
                                "matched_rank": 1,
                                "retrieval_trace": [],
                            },
                            {
                                "sample_index": 2,
                                "expected_case_id": "C2",
                                "query": "first miss",
                                "status": "success",
                                "retrieval_status": "success",
                                "matched_rank": None,
                                "retrieval_trace": [],
                            },
                        ]
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    _artifact(
                        [
                            {
                                "sample_index": 0,
                                "call_sno": "a",
                                "expected_case_id": "C0",
                                "query": "second query",
                                "status": "success",
                                "retrieval_status": "success",
                                "matched_rank": None,
                                "retrieval_trace": [],
                            },
                            {
                                "sample_index": 1,
                                "call_sno": "b",
                                "expected_case_id": "C1",
                                "query": "both hit",
                                "status": "success",
                                "retrieval_status": "success",
                                "matched_rank": 2,
                                "retrieval_trace": [],
                            },
                        ]
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = compare_result_artifacts(first_path, second_path, cutoff=5)

        self.assertEqual(report["summary"]["aligned_samples"], 3)
        self.assertEqual(report["summary"]["first_hit_second_miss"], 1)
        self.assertEqual(report["summary"]["first_only_samples"], 1)
        self.assertEqual(len(report["records"]), 1)
        item = report["records"][0]
        self.assertEqual(item["sample_index"], 0)
        self.assertTrue(item["first"]["hit"])
        self.assertFalse(item["second"]["hit"])
        self.assertEqual(item["first"]["matched_rank"], 3)
        self.assertEqual(item["second"]["matched_rank"], None)
        self.assertNotIn("content", item["first"]["retrieval_trace"][0])

    def test_cutoff_excludes_rank_outside_window(self):
        record = {
            "sample_index": 0,
            "expected_case_id": "C0",
            "query": "q",
            "status": "success",
            "retrieval_status": "success",
            "matched_rank": 5,
            "retrieval_trace": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            first_path.write_text(json.dumps(_artifact([record])), encoding="utf-8")
            second_path.write_text(
                json.dumps(_artifact([{**record, "matched_rank": None}])), encoding="utf-8"
            )

            report = compare_result_artifacts(first_path, second_path, cutoff=3)

        self.assertEqual(report["summary"]["first_hit_second_miss"], 0)
        self.assertEqual(report["records"], [])

    def test_rejects_non_retrieval_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"artifact_type": "generated_queries", "records": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                compare_result_artifacts(path, path)


if __name__ == "__main__":
    unittest.main()
