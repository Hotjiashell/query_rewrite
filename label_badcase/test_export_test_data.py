import copy
import json
import tempfile
import unittest
from pathlib import Path

from export_test_data import export_test_data, main


class ExportTestDataTests(unittest.TestCase):
    def setUp(self):
        self.dialogues = [
            {"call_sno": "001", "chat_content": "对话一", "caseID": "GT", "extra": 123},
            {"chat_content": "对话二", "caseID": "GT"},
        ]
        self.badcases = {"records": [{"sample_index": 0, "call_sno": "001", "chat_content": "对话一",
            "expected_case_id": "GT", "retrieval_trace": [
                {"rank": i + 1, "case_id": cid} for i, cid in enumerate(["A", "B", "C", "D", "GT"])]}]}
        self.annotations = {"artifact_type": "badcase_annotations", "annotations": {
            "0": {"sample_index": 0, "completed": False, "reviews": {
                "A": {"verdict": "yes"}, "B": {"verdict": "yes"},
                "C": {"verdict": "no"}, "D": {"verdict": "uncertain"}, "GT": {"verdict": "yes"}}}}}

    def test_merge_preserves_rows_and_does_not_spread_by_gt(self):
        before = copy.deepcopy(self.dialogues)
        result, stats = export_test_data(self.dialogues, self.annotations, self.badcases)
        self.assertEqual(result[0], {**before[0], "m_caseID": ["GT", "A", "B"]})
        self.assertEqual(result[1]["m_caseID"], ["GT"])
        self.assertEqual(stats, {"total": 2, "updated_samples": 1, "added_case_ids": 2})
        self.assertEqual(self.dialogues, before)
        repeated, stats = export_test_data(result, self.annotations, self.badcases)
        self.assertEqual(repeated, result)
        self.assertEqual(stats["added_case_ids"], 0)

    def test_completed_only_and_existing_ids(self):
        result, _ = export_test_data(self.dialogues, self.annotations, self.badcases, completed_only=True)
        self.assertEqual(result[0]["m_caseID"], ["GT"])
        self.annotations["annotations"]["0"]["completed"] = True
        self.dialogues[0]["m_caseID"] = ["OLD", "GT", "OLD"]
        result, _ = export_test_data(self.dialogues, self.annotations, self.badcases, completed_only=True)
        self.assertEqual(result[0]["m_caseID"], ["GT", "OLD", "A", "B"])

    def test_reject_mismatched_data_and_unknown_candidate(self):
        for field, value in (("caseID", "OTHER"), ("chat_content", "不同对话"), ("call_sno", "002")):
            rows = copy.deepcopy(self.dialogues)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                export_test_data(rows, self.annotations, self.badcases)
        self.annotations["annotations"]["0"]["reviews"]["UNKNOWN"] = {"verdict": "yes"}
        with self.assertRaisesRegex(ValueError, "Top-10"):
            export_test_data(self.dialogues, self.annotations, self.badcases)

    def test_cli_sources_and_protect_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / (name + '.json') for name in ('input', 'annotations', 'badcases', 'output')}
            self.annotations['sources'] = {'badcases': str(paths['badcases'])}
            for name, payload in [('input', self.dialogues), ('annotations', self.annotations), ('badcases', self.badcases)]:
                paths[name].write_text(json.dumps(payload), encoding='utf-8')
            args = ['--input', str(paths['input']), '--annotations', str(paths['annotations']), '--output']
            self.assertEqual(main(args + [str(paths['output'])]), 0)
            self.assertEqual(json.loads(paths['output'].read_text())[0]['m_caseID'], ['GT', 'A', 'B'])
            before = paths['input'].read_bytes()
            self.assertEqual(main(args + [str(paths['input'])]), 2)
            self.assertEqual(paths['input'].read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
