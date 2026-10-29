import json
import tempfile
import unittest
from pathlib import Path

from collect_badcases import collect_badcases


def _artifact(records, *, source_query_path=None):
    artifact = {"artifact_type": "retrieval_evaluation", "records": records}
    if source_query_path is not None:
        artifact["configuration"] = {"source_query_path": source_query_path}
    return artifact


class CollectBadcasesTests(unittest.TestCase):
    def test_ranks_gt_cases_and_lists_associated_conversations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.json"
            cases_path = root / "cases.json"
            records = [
                {
                    "sample_index": index,
                    "call_sno": f"call-{index}",
                    "chat_content": f"对话{index}",
                    "expected_case_id": case_id,
                    "status": "success",
                    "matched_rank": rank,
                }
                for index, (case_id, rank) in enumerate(
                    [("unknown", None), ("KT2", None), ("KT2", 11), ("KT2", 1), (None, None)]
                )
            ]
            results_path.write_text(json.dumps(_artifact(records)), encoding="utf-8")
            cases_path.write_text(json.dumps({"KT2": {"case_name": "案例2"}}), encoding="utf-8")

            report = collect_badcases(results_path, cases_path)
            groups = report["gt_case_frequency"]
            self.assertEqual([item["gt_case_id"] for item in groups], ["KT2", "unknown", None])
            self.assertEqual([item["frequency"] for item in groups], [2, 1, 1])
            self.assertEqual(groups[0]["gt_case_title"], "案例2")
            self.assertEqual(groups[0]["conversations"], [
                {"sample_index": 1, "call_sno": "call-1", "chat_content": "对话1"},
                {"sample_index": 2, "call_sno": "call-2", "chat_content": "对话2"},
            ])
            self.assertIsNone(groups[1]["gt_case_title"])

            results_path.write_text(json.dumps(_artifact([])), encoding="utf-8")
            self.assertEqual(collect_badcases(results_path, cases_path)["gt_case_frequency"], [])

    def test_collects_only_successful_misses_within_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.json"
            case_summary_path = root / "cases.json"
            results_path.write_text(
                json.dumps(
                    _artifact(
                        [
                            {
                                "sample_index": 0,
                                "call_sno": "a",
                                "chat_content": "hit sample",
                                "expected_case_id": "KT1",
                                "query": "q1",
                                "status": "success",
                                "matched_rank": 3,
                                "retrieval_trace": [{"rank": 3, "case_id": "KT1", "case_title": "t", "score": 0.5}],
                            },
                            {
                                "sample_index": 1,
                                "call_sno": "b",
                                "chat_content": "miss sample",
                                "expected_case_id": "KT2",
                                "query": "q2",
                                "queries": ["q2", "q2 alternative"],
                                "status": "success",
                                "matched_rank": None,
                                "retrieval_trace": [],
                            },
                            {
                                "sample_index": 2,
                                "call_sno": "c",
                                "chat_content": "rank outside cutoff",
                                "expected_case_id": "KT3",
                                "query": "q3",
                                "status": "success",
                                "matched_rank": 11,
                                "retrieval_trace": [],
                            },
                            {
                                "sample_index": 3,
                                "call_sno": "d",
                                "chat_content": "generation failed",
                                "expected_case_id": "KT4",
                                "query": None,
                                "status": "failed",
                                "matched_rank": None,
                                "retrieval_trace": [],
                            },
                        ]
                    )
                ),
                encoding="utf-8",
            )
            case_summary_path.write_text(
                json.dumps(
                    {
                        "KT2": {"case_name": "案例2标题", "text": "..."},
                    }
                ),
                encoding="utf-8",
            )

            report = collect_badcases(results_path, case_summary_path, cutoff=10)

        self.assertEqual(report["summary"]["total_records"], 4)
        self.assertEqual(report["summary"]["badcase_count"], 2)
        self.assertEqual(report["summary"]["unresolved_gt_case_count"], 1)

        by_index = {item["sample_index"]: item for item in report["records"]}
        self.assertEqual(set(by_index), {1, 2})

        miss = by_index[1]
        self.assertEqual(miss["gt_case_title"], "案例2标题")
        self.assertTrue(miss["gt_case_found"])
        self.assertEqual(miss["queries"], ["q2", "q2 alternative"])

        outside_cutoff = by_index[2]
        self.assertIsNone(outside_cutoff["gt_case_title"])
        self.assertFalse(outside_cutoff["gt_case_found"])

    def test_rejects_non_retrieval_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "bad.json"
            case_summary_path = root / "cases.json"
            results_path.write_text(json.dumps({"artifact_type": "generated_queries", "records": []}), encoding="utf-8")
            case_summary_path.write_text(json.dumps({}), encoding="utf-8")

            with self.assertRaises(ValueError):
                collect_badcases(results_path, case_summary_path)

    def test_falls_back_to_queries_artifact_using_source_query_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.json"
            query_path = root / "queries.json"
            case_summary_path = root / "cases.json"
            results_path.write_text(
                json.dumps(
                    _artifact(
                        [
                            {
                                "sample_index": 4,
                                "expected_case_id": "KT4",
                                "query": "q1",
                                "status": "success",
                                "matched_rank": None,
                                "retrieval_trace": [],
                            }
                        ],
                        source_query_path="queries.json",
                    )
                ),
                encoding="utf-8",
            )
            query_path.write_text(
                json.dumps(
                    {
                        "artifact_type": "generated_queries",
                        "records": [{"sample_index": 4, "query": "q1", "queries": ["q1", "q2"]}],
                    }
                ),
                encoding="utf-8",
            )
            case_summary_path.write_text(json.dumps({"KT4": {"case_name": "案例4"}}), encoding="utf-8")

            report = collect_badcases(results_path, case_summary_path)

        self.assertEqual(report["records"][0]["queries"], ["q1", "q2"])

    def test_rejects_non_object_case_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.json"
            case_summary_path = root / "cases.json"
            results_path.write_text(json.dumps(_artifact([])), encoding="utf-8")
            case_summary_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

            with self.assertRaises(ValueError):
                collect_badcases(results_path, case_summary_path)


if __name__ == "__main__":
    unittest.main()
