"""Verify compatibility with evaluate.py's actual multi-query output."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from gen_query import MultiQueryGenerator
from test_syonyms.evaluate_synonyms import main


class FakeMultiGenerator(MultiQueryGenerator):
    def generate_queries(self, dialogue):
        return ['私密邮件如何发送', '卡片消息如何使用']


class MultiQueryCompatibilityTest(unittest.TestCase):
    def test_generated_artifact_replacement_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'queries.json'
            output = Path(tmp) / 'results.json'
            synonyms = Path(tmp) / 'synonyms.txt'
            synonyms.write_text('私密邮件, 加密邮件\n卡片消息, 卡片形式\n', encoding='utf-8')
            records = evaluate.QueryGenerationRunner(FakeMultiGenerator()).generate([
                evaluate.DialogueSample(0, 'call1', '私密邮件和卡片消息', 'case1')
            ])
            artifact = evaluate.build_query_artifact(records, input_path='unused',
                model_name='fake', concurrency=1, method='multi_query')
            evaluate.write_json_atomically(artifact, source)
            arguments = ['retrieve', '--synonym-mode', 'replace', '--synonyms', str(synonyms),
                         '--input', str(source), '--output', str(output)]
            response = {'retrieval_result': {'top1': {'case_id': 'case1', 'case_title': 'title'}}}
            with patch.object(evaluate.SearchRetriever, 'retrieve', return_value=response) as retrieve:
                self.assertEqual(main(arguments), 0)
                self.assertCountEqual([call.args[0] for call in retrieve.call_args_list],
                                      ['加密邮件如何发送', '卡片形式如何使用'])
            result = json.loads(output.read_text())
            self.assertEqual(result['records'][0]['queries'], ['加密邮件如何发送', '卡片形式如何使用'])
            result['records'][0]['retrieval_status'] = 'partial_success'
            evaluate.write_json_atomically(result, output)
            with patch.object(evaluate.SearchRetriever, 'retrieve', return_value=response) as retrieve:
                self.assertEqual(main(arguments + ['--retry-errors', str(output)]), 0)
                self.assertEqual(retrieve.call_count, 2)


if __name__ == '__main__':
    unittest.main()
