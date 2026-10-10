import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from recall_at_threshold import evaluate_threshold, main


def artifact():
    return {
        "artifact_type": "retrieval_evaluation",
        "configuration": {"fusion_method": "score", "top_k": 1},
        "records": [{
            "sample_index": 0, "expected_case_id": "target", "filter_status": "success",
            "retrieval_trace": [],
            "per_query_traces": [{"query": "q", "trace": [
                {"case_id": "noise", "rank": 1, "score": 0.1},
                {"case_id": "target", "rank": 2, "score": 0.9},
            ]}],
            "startlux_judgments": [
                {"case_id": "noise", "related_probability": 0.3},
                {"case_id": "target", "related_probability": 0.7},
            ],
        }],
    }


class ThresholdRecallTests(unittest.TestCase):
    def test_replays_original_candidates_with_inclusive_threshold(self):
        report = evaluate_threshold(artifact(), 0.7)
        self.assertEqual(report["metrics"]["recall_at_1"], 1)
        self.assertEqual(report["records"][0]["kept_candidate_count"], 1)
        self.assertEqual(evaluate_threshold(artifact(), 0.8)["metrics"]["recall_at_10"], 0)

    def test_fusion_override_and_failed_samples_in_denominator(self):
        data = artifact()
        data["records"].append({"filter_status": "failed"})
        report = evaluate_threshold(data, 0, "round_robin")
        self.assertEqual(report["metrics"]["recall_at_1"], 0)
        for k in (3, 5, 10):
            self.assertEqual(report["metrics"][f"recall_at_{k}"], 0.5)
        self.assertEqual(report["metrics"]["failed_samples"], 1)

    def test_incomplete_judgments_are_failed_not_silent_drops(self):
        data = artifact()
        data["records"][0]["startlux_judgments"].pop()
        self.assertEqual(evaluate_threshold(data, 0.5)["metrics"]["failed_samples"], 1)

    def test_cli_writes_report_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "in.json", Path(directory) / "out.json"
            source.write_text(json.dumps(artifact()))
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(["--input", str(source), "--threshold", "0.7", "--output", str(output)]), 0)
            self.assertIn("Recall@10: 1.000000", stdout.getvalue())
            self.assertEqual(json.loads(output.read_text())["metrics"]["recall_at_1"], 1)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--input", str(source), "--threshold", "0.5", "--output", str(source)]), 1)

    def test_invalid_threshold_rejected(self):
        for threshold in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                evaluate_threshold(artifact(), threshold)


if __name__ == "__main__":
    unittest.main()
