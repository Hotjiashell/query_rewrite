import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import calculate_annotated_metrics, main as evaluate_main
from evaluate_annotated import main


class AnnotatedMetricsTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"caseID": "GT", "m_caseID": ["GT", "ALT", "ALT"], "can_retrieve": available,
             "chat_content": f"dialogue-{i}", "call_sno": str(i)}
            for i, available in enumerate([True, False, True, True])
        ]
        traces = [[{"rank": 3, "case_id": "GT"}, {"rank": 1, "case_id": "ALT"}],
                  [{"rank": 1, "case_id": "ALT"}], [], [{"rank": 5, "case_id": "ALT"}]]
        self.records = [
            {"sample_index": i, "expected_case_id": "GT", "chat_content": f"dialogue-{i}",
             "call_sno": str(i), "status": "failed" if i == 2 else "success",
             "retrieval_trace": trace, "matched_rank": 3 if i == 0 else None}
            for i, trace in enumerate(traces)
        ]

    def test_four_metrics_denominators_and_failure(self):
        before = copy.deepcopy(self.records)
        metrics = calculate_annotated_metrics(self.records, self.samples)
        self.assertEqual(metrics['original']['hits_at_1'], 0)
        self.assertEqual(metrics['original']['recall_at_3'], 1/4)
        self.assertEqual(metrics['multi_case']['recall_at_1'], 2/4)
        self.assertEqual(metrics['multi_case']['recall_at_5'], 3/4)
        self.assertEqual(metrics['retrievable_only']['recall_at_3'], 1/3)
        self.assertEqual(metrics['multi_case_retrievable_only']['recall_at_1'], 1/3)
        self.assertEqual(metrics['multi_case_retrievable_only']['recall_at_5'], 2/3)
        self.assertEqual(metrics['multi_case_retrievable_only']['failed_samples'], 1)
        self.assertEqual(metrics['coverage']['excluded_can_retrieve_false'], 1)
        self.assertEqual(self.records, before)

    def test_original_gt_is_always_accepted_and_failed_hits_ignored(self):
        self.samples[0]['m_caseID'] = []
        self.records[2]['retrieval_trace'] = [{'rank': 1, 'case_id': 'ALT'}]
        metrics = calculate_annotated_metrics(self.records, self.samples)
        self.assertEqual(metrics['multi_case']['hits_at_1'], 1)
        self.assertEqual(metrics['multi_case']['hits_at_3'], 2)

    def test_empty_scope_and_partial_result_coverage(self):
        for sample in self.samples:
            sample['can_retrieve'] = False
        metrics = calculate_annotated_metrics(self.records[:2], self.samples)
        self.assertEqual(metrics['retrievable_only']['total_samples'], 0)
        self.assertEqual(metrics['retrievable_only']['recall_at_10'], 0)
        self.assertEqual(metrics['coverage']['dataset_samples_without_results'], 2)
        self.assertEqual(calculate_annotated_metrics([], [])['original']['total_samples'], 0)

    def test_reject_misalignment_and_invalid_annotations(self):
        for field, value in [('caseID', 'WRONG'), ('chat_content', 'wrong'), ('call_sno', 'wrong'),
                             ('m_caseID', 'ALT'), ('m_caseID', [None]), ('can_retrieve', 'false')]:
            samples = copy.deepcopy(self.samples)
            samples[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                calculate_annotated_metrics(self.records, samples)
        for index in [-1, 99, True]:
            records = copy.deepcopy(self.records)
            records[0]['sample_index'] = index
            with self.subTest(index=index), self.assertRaises(ValueError):
                calculate_annotated_metrics(records, self.samples)
        with self.assertRaises(ValueError):
            calculate_annotated_metrics(self.records + self.records[:1], self.samples)

    def test_both_cli_entry_points_are_offline_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, dataset, output = [root / name for name in ['results.json', 'data.json', 'out.json']]
            artifact = {'artifact_type': 'retrieval_evaluation', 'records': self.records,
                        'configuration': {}, 'metrics': {'old': 'preserved'}}
            results.write_text(json.dumps(artifact))
            dataset.write_text(json.dumps(self.samples))
            before = results.read_bytes()
            with patch('evaluate.SearchRetriever', side_effect=AssertionError('must stay offline')):
                self.assertEqual(main(['--results', str(results), '--dataset', str(dataset), '--output', str(output)]), 0)
                standalone = json.loads(output.read_text())
                self.assertEqual(evaluate_main(['metrics', '--input', str(results), '--dataset', str(dataset), '--output', str(output)]), 0)
            self.assertEqual(json.loads(output.read_text()), standalone)
            self.assertEqual(standalone['metrics'], artifact['metrics'])
            self.assertEqual(standalone['records'], artifact['records'])
            self.assertEqual(standalone['annotated_metrics']['multi_case']['recall_at_1'], .5)
            self.assertEqual(results.read_bytes(), before)
            self.assertEqual(main(['--results', str(results), '--dataset', str(dataset), '--output', str(results)]), 2)
            self.assertEqual(results.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
