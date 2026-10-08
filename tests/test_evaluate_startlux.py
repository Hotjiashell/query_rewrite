import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate import build_query_artifact
from evaluate_startlux import StartLuxBatchAdapter, create_filter, main, parse_args
from test_evaluate_laya import Retriever, source


class StartLuxModel:
    def __init__(self):
        self.calls = []

    def decide_batch(self, requests):
        self.calls.append(requests)
        return [{"relevance": {
            "choice": "unrelated" if state["case_title"] == "irrelevant" else "related",
            "probabilities": {"related": 0.1 if state["case_title"] == "irrelevant" else 0.9},
        }} for state, questions in requests]


class StartLuxTests(unittest.TestCase):
    def test_adapter_chunks_and_preserves_answer_order(self):
        model = StartLuxModel()
        states = [{"case_title": title} for title in ["a", "irrelevant", "b", "c", "d"]]
        result = StartLuxBatchAdapter(model).predict_batch(states, {}, batch_size=2)
        self.assertEqual([len(call) for call in model.calls], [2, 2, 1])
        self.assertEqual(len(result), 5)
        self.assertEqual(result[1]["answers"]["relevance"]["probabilities"]["related"], 0.1)

    def test_factory_defaults_to_4b_and_total_budget_without_vision(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "StartLux-Decision-4B"
            model_path.mkdir()
            args = parse_args(["--startlux-model", str(model_path)])
            constructor = unittest.mock.Mock(return_value=StartLuxModel())
            with patch.dict("sys.modules", {"startlux_decision": SimpleNamespace(StartLuxDecision=constructor)}):
                filter_, metadata = create_filter(args, {})
            constructor.assert_called_once_with(str(model_path), device=None, max_length=4096,
                                                max_batch_tokens=65536, images=False)
            self.assertEqual(filter_.threshold, 0.5)
            self.assertEqual(metadata["model"], str(model_path))
        self.assertIsNone(parse_args([]).startlux_model)
        from evaluate_startlux import DEFAULT_MODEL
        self.assertEqual(DEFAULT_MODEL, "StartLux-Decision-4B")

    def test_existing_queries_produce_startlux_audits_and_offline_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, config, output, analysed = [root / name for name in ("queries.json", "config.json", "out.json", "analysed.json")]
            queries.write_text(json.dumps(build_query_artifact([source()], input_path="dialogs", model_name="test", concurrency=1, method="custom_multi")))
            config.write_text(json.dumps({"retrieval": {"output_path": str(output)}, "startlux_filter": {"model": directory}}))
            constructor = unittest.mock.Mock(return_value=StartLuxModel())
            with patch.dict("sys.modules", {"startlux_decision": SimpleNamespace(StartLuxDecision=constructor), "laya": None}), \
                 patch("evaluate_laya.SearchRetriever", return_value=Retriever()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["retrieve", "--config", str(config), "--query-file", str(queries)]), 0)
            artifact = json.loads(output.read_text())
            self.assertEqual(artifact["configuration"]["pipeline"], "multi_query_startlux_filter")
            record = artifact["records"][0]
            self.assertIn("startlux_judgments", record)
            self.assertNotIn("laya_judgments", record)
            self.assertIn("startlux_filter_seconds", record["timings"])
            self.assertEqual(artifact["metrics"]["recall_at_1"], 1)
            self.assertEqual(artifact["threshold_search"]["best"]["recall_at_10"], 1)
            with patch.dict("sys.modules", {"startlux_decision": None, "laya": None}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["analyze", "--input", str(output), "--output", str(analysed)]), 0)
            self.assertEqual(json.loads(analysed.read_text())["threshold_search"], artifact["threshold_search"])

    def test_invalid_batch_budget_and_missing_model_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "max_batch_tokens"):
            create_filter(parse_args(["--max-batch-tokens", "0"]), {})
        with self.assertRaisesRegex(ValueError, "hf download"):
            create_filter(parse_args(["--startlux-model", "/nonexistent/checkpoint"]), {})


if __name__ == "__main__":
    unittest.main()
