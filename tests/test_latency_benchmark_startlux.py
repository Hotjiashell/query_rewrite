import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate import DialogueSample
from evaluate_laya import LayaRelevanceFilter
from latency_benchmark_startlux import (
    SerialStartLuxBenchmark, build_benchmark_artifact, main, parse_args,
)
from test_evaluate_laya import Agent, Retriever, source


class StartLuxBenchmarkTests(unittest.TestCase):
    def test_serial_samples_and_filter_boundaries(self):
        events = []
        agent = Agent({"irrelevant": 0.1})
        inner = LayaRelevanceFilter(agent)

        class Filter:
            def judge(self, dialogue, cases):
                events.append("filter")
                return inner.judge(dialogue, cases)

        class Search(Retriever):
            def retrieve(self, query):
                events.append(query)
                return super().retrieve(query)

        records = SerialStartLuxBenchmark(Search(), Filter(), [source(), source()]).run()
        self.assertEqual(events[2], "filter")
        self.assertEqual(events[5], "filter")
        for record in records:
            self.assertEqual(record["status"], "success")
            self.assertIsNone(record["query_generation_time_sec"])
            self.assertLessEqual(record["time_to_retrieval_sec"], record["time_to_filter_sec"])
            self.assertLessEqual(record["time_to_filter_sec"], record["time_to_fusion_sec"])
            self.assertEqual(record["matched_rank"], 1)
            self.assertEqual(len(record["startlux_judgments"]), 4)
            self.assertNotIn("laya_judgments", record)

    def test_generation_included_and_analysis_excluded(self):
        benchmark = SerialStartLuxBenchmark(
            Retriever(), LayaRelevanceFilter(Agent()),
            [DialogueSample(0, "call", "对话", "B")], generator=object(),
        )
        # A deterministic clock isolates generation, filtering and offline analysis.
        now = [0.0]

        def generate(sample):
            now[0] += 2
            return source()

        inner = benchmark.timed_filter.filter

        class Filter:
            def judge(self, dialogue, cases):
                now[0] += 3
                return inner.judge(dialogue, cases)

        benchmark.timed_filter.filter = Filter()
        benchmark.query_runner.generate_sample = generate
        with patch("latency_benchmark_startlux.time", SimpleNamespace(perf_counter=lambda: now[0])):
            records = benchmark.run()
            self.assertEqual(records[0]["query_generation_time_sec"], 2)
            self.assertEqual(records[0]["filter_only_time_sec"], 3)
            self.assertEqual(records[0]["time_to_fusion_sec"], 5)

            def expensive_analysis(artifact):
                now[0] += 100

            with patch("latency_benchmark_startlux.analyse_saved_results", expensive_analysis):
                artifact = build_benchmark_artifact(
                    records, input_path="queries.json",
                    retrieval=SimpleNamespace(url="local", timeout=30, fusion_method="round_robin", top_k=10),
                    filter_metadata={}, generation_included=True, test_num=0, setup_seconds=20,
                )
        self.assertEqual(artifact["offline_analysis_seconds"], 100)
        self.assertEqual(artifact["latency"]["time_to_fusion"]["average_seconds"], 5)
        self.assertEqual(records[0]["total_attempt_time_sec"], 5)

    def test_failures_do_not_count_as_completed_pipeline(self):
        class BrokenFilter:
            def judge(self, dialogue, cases):
                raise RuntimeError("server unavailable")

        records = SerialStartLuxBenchmark(Retriever(), BrokenFilter(), [source()]).run()
        self.assertEqual(records[0]["status"], "failed")
        self.assertIsNone(records[0]["time_to_filter_sec"])
        self.assertIsNone(records[0]["time_to_fusion_sec"])
        self.assertIsNotNone(records[0]["filter_only_time_sec"])
        self.assertIsNotNone(records[0]["total_attempt_time_sec"])

    def test_cli_has_no_stage_and_keeps_startlux_options(self):
        args = parse_args(["--query-file", "queries.json", "--test-num", "20",
                           "--threshold", "0.7", "--startlux-endpoint", "http://localhost:8090"])
        self.assertEqual(args.test_num, 20)
        self.assertEqual(args.stage, "all")
        self.assertEqual(args.threshold, 0.7)

    def test_query_file_without_configured_input_writes_replayable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            output = Path(directory) / "benchmark.json"
            config.write_text("{}")
            with patch("latency_benchmark_startlux.load_generated_query_records", return_value=[source()]), \
                 patch("latency_benchmark_startlux.SearchRetriever", return_value=Retriever()), \
                 patch("latency_benchmark_startlux.create_filter", return_value=(LayaRelevanceFilter(Agent()), {})), \
                 patch("builtins.print"):
                result = main(["--config", str(config), "--query-file", "queries.json", "--output", str(output)])
            self.assertEqual(result, 0)
            artifact = json.loads(output.read_text())
            self.assertFalse(artifact["configuration"]["generation_included"])
            self.assertIn("union_top_k", artifact)
            self.assertIn("threshold_search", artifact)
            self.assertIn("startlux_judgments", artifact["records"][0])


if __name__ == "__main__":
    unittest.main()
