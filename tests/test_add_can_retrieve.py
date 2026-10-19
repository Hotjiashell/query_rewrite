import copy
import json
import tempfile
import unittest
from pathlib import Path

from add_can_retrieve import add_can_retrieve, main


class CanRetrieveTests(unittest.TestCase):
    def setUp(self):
        self.samples = [{"caseID": cid, "chat_content": "对话", "m_caseID": [cid, "OTHER"]}
                        for cid in ["A", "B", "C", "D", "E", "A"]]
        self.report = {"artifact_type": "gt_title_retrieval_check", "configuration": {"cutoff": 10}, "records": [
            {"expected_case_id": "A", "outcome": "hit", "matched_rank": 1},
            {"expected_case_id": "B", "outcome": "miss", "matched_rank": None},
            {"expected_case_id": "C", "outcome": "miss", "matched_rank": 12},
            {"expected_case_id": "D", "outcome": "missing_title"},
            {"expected_case_id": "E", "outcome": "retrieval_failed"},
        ]}

    def test_merge_and_cutoff_and_errors(self):
        before = copy.deepcopy(self.samples)
        result, stats = add_can_retrieve(self.samples, self.report)
        self.assertEqual([r['can_retrieve'] for r in result], [True, False, True, False, False, True])
        self.assertEqual(stats, {"total": 6, "true": 3, "false": 3, "missing_title": 1, "retrieval_failed": 1})
        self.assertEqual(self.samples, before)
        for old, new in zip(before, result):
            self.assertEqual(new, {**old, "can_retrieve": new['can_retrieve']})
        self.assertEqual(add_can_retrieve(result, self.report)[0], result)

    def test_trace_can_prove_retrieval(self):
        self.report['records'][1]['retrieval_trace'] = [{'case_id': 'B', 'rank': 20}]
        result, _ = add_can_retrieve(self.samples, self.report)
        self.assertTrue(result[1]['can_retrieve'])

    def test_invalid_or_incomplete_report(self):
        for mutate in (
            lambda r: r.update(artifact_type='badcase_analysis'),
            lambda r: r.pop('records'),
            lambda r: r['records'].pop(),
            lambda r: r['records'].append(r['records'][0]),
            lambda r: r['records'][0].update(matched_rank=None),
            lambda r: r['records'][4].update(matched_rank=1),
        ):
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.subTest(report=report), self.assertRaises(ValueError):
                add_can_retrieve(self.samples, report)

    def test_cli_and_input_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, report, output = (root / name for name in ['input.json', 'report.json', 'output.json'])
            source.write_text(json.dumps(self.samples))
            report.write_text(json.dumps(self.report))
            args = ['--input', str(source), '--report', str(report), '--output']
            self.assertEqual(main(args + [str(output)]), 0)
            self.assertIs(json.loads(output.read_text())[0]['can_retrieve'], True)
            before = source.read_bytes()
            self.assertEqual(main(args + [str(source)]), 2)
            self.assertEqual(source.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
