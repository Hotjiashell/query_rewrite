import json
import tempfile
import unittest
from pathlib import Path

from promptEov.engine import PromptEov


class PromptEovTests(unittest.TestCase):
    def test_optimizer_result_tag_is_extracted(self):
        prompt, tagged = PromptEov._extract_optimized_prompt(
            "分析过程\n<result>\n新的提示词\n</result>\n其他内容"
        )
        self.assertEqual(prompt, "新的提示词")
        self.assertTrue(tagged)

    def test_badcases_are_optimized_in_batches_of_fifty(self):
        calls = []

        def llm(text):
            calls.append(text)
            if "优化提示词" in text:
                return f"分析\n<result>prompt-{sum('优化提示词' in call for call in calls)}</result>"
            return "单条分析"

        dataset = [
            {"chat_content": f"dialogue-{index}", "caseID": f"case-{index}"}
            for index in range(55)
        ]

        def generate_query(prompt, dialogue):
            return dialogue

        def retrieve(query):
            return {"retrieval_result": {}}

        with tempfile.TemporaryDirectory() as directory:
            history = PromptEov(
                "initial",
                dataset,
                llm=llm,
                generate_query=generate_query,
                retrieve=retrieve,
                analysis_concurrency=1,
            ).run(iterations=1, output_dir=directory, progress=False)
            record = json.loads(
                (Path(directory) / "iteration_1.json").read_text(encoding="utf-8")
            )
            log = (Path(directory) / "evolution.log").read_text(encoding="utf-8")

        self.assertEqual(len(history), 1)
        self.assertEqual(len(record["optimization_batches"]), 2)
        self.assertEqual(
            [item["bad_case_count"] for item in record["optimization_batches"]],
            [50, 5],
        )
        self.assertEqual(record["new_prompt"], "prompt-2")
        self.assertIn("analysis_started iteration=1 batch=1 badcases=50", log)
        self.assertIn("analysis_started iteration=1 batch=2 badcases=5", log)
        self.assertIn("optimization_finished iteration=1 batch=2", log)

    def test_empty_analysis_skips_optimizer_when_there_are_no_badcases(self):
        calls = []

        def llm(text):
            calls.append(text)
            return "<result>should not be used</result>"

        dataset = [{"chat_content": "dialogue", "caseID": "case-1"}]

        with tempfile.TemporaryDirectory() as directory:
            history = PromptEov(
                "initial",
                dataset,
                llm=llm,
                generate_query=lambda prompt, dialogue: "query",
                retrieve=lambda query: {
                    "retrieval_result": {"top1": {"case_id": "case-1"}}
                },
            ).run(iterations=1, output_dir=directory, progress=False)

        self.assertEqual(history[0]["new_prompt"], "initial")
        self.assertEqual(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
