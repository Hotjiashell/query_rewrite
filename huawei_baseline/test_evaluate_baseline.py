import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import DialogueSample, QueryGenerationRunner
from huawei_baseline.evaluate_baseline import HuaweiQueryGenerator, main


class BaselineTests(unittest.TestCase):
    def test_fixed_api_receives_dialogue_and_returns_single_query(self):
        with patch("huawei_baseline.api.get_query", return_value={"baseline_user_query": " 查询 "}) as api:
            record = QueryGenerationRunner(HuaweiQueryGenerator()).generate_sample(DialogueSample(0, "call", "完整对话", "A"))
        api.assert_called_once_with("完整对话")
        self.assertEqual(record.query, "查询")
        self.assertIsNone(record.queries)
        self.assertEqual(record.status, "success")

    def test_bad_responses_and_api_errors_are_isolated(self):
        for response in [None, {}, {"baseline_user_query": " "}, {"baseline_user_query": ["q"]}]:
            with self.subTest(response=response), patch("huawei_baseline.api.get_query", return_value=response):
                record = QueryGenerationRunner(HuaweiQueryGenerator()).generate_sample(DialogueSample(0, "call", "对话", "A"))
                self.assertEqual(record.status, "failed")
        with patch("huawei_baseline.api.get_query", side_effect=RuntimeError("timeout")):
            record = QueryGenerationRunner(HuaweiQueryGenerator()).generate_sample(DialogueSample(0, "call", "对话", "A"))
            self.assertIn("timeout", record.error)

    def test_all_and_retrieve_preserve_failure_denominator_and_saved_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, queries, result = [root / name for name in ("config.json", "source.json", "queries.json", "result.json")]
            config.write_text("{}")
            source.write_text(json.dumps([
                {"call_sno": "1", "chat_content": "对话1", "caseID": "A"},
                {"call_sno": "2", "chat_content": "对话2", "caseID": "B"},
            ]))

            class Search:
                def retrieve(self, query):
                    if query != "查询":
                        raise AssertionError("unexpected query")
                    return {"retrieval_result": {"top1": {"case_id": "A", "case_title": "案例A", "score": 0.9}}}

            with patch("huawei_baseline.api.get_query", side_effect=[{"baseline_user_query": "查询"}, RuntimeError("timeout")]) as api, \
                 patch("huawei_baseline.evaluate_baseline.SearchRetriever", return_value=Search()), patch("builtins.print"):
                self.assertEqual(main(["all", "--config", str(config), "--input", str(source), "--query-output", str(queries), "--output", str(result)]), 0)
                self.assertEqual(api.call_count, 2)
                self.assertEqual(main(["retrieve", "--config", str(config), "--input", str(queries), "--output", str(result)]), 0)
                self.assertEqual(api.call_count, 2)
            payload = json.loads(result.read_text())
            self.assertEqual(payload["metrics"]["recall_at_1"], 0.5)
            self.assertEqual(payload["records"][1]["retrieval_status"], "skipped")
            self.assertEqual(payload["records"][0]["retrieval_trace"][0]["case_id"], "A")
            self.assertEqual(json.loads(queries.read_text())["records"][0]["query"], "查询")

    def test_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text("{}")
            with patch("builtins.print"):
                self.assertEqual(main(["generate", "--config", str(config), "--input", "same.json", "--output", "same.json"]), 2)


if __name__ == "__main__":
    unittest.main()
