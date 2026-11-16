import tempfile
import unittest
from pathlib import Path
from .research import investigate, enrich, expand
from .pipeline import canonical, digest, save, parse_result, model_input


class ResearchTests(unittest.TestCase):
    def test_budget_and_evidence(self):
        calls, searches = [], []
        class Model:
            def task(self, stage, data):
                calls.append(data)
                if data['can_search']:
                    return {'status': 'success', 'result': {'action': 'search', 'query': '补充查询'}}
                return {'status': 'success', 'result': {'action': 'summarize', 'desc': '定义',
                    'relations': [{'source': 'A', 'target': 'B', 'description': 'A依赖B'}]}}
        def retrieve(q):
            searches.append(q)
            return {'retrieval_result': {f'top{i}': {'case_id': 'hidden', 'case_title': f'标题{i}', 'content': f'正文{i}'} for i in range(1, 4)}}
        result = investigate('A', [], ['A', 'B'], Model(), {'max_searches': 2, 'top_k': 2}, retrieve)
        self.assertEqual(searches, ['A', '补充查询'])
        self.assertEqual(result['status'], 'success')
        self.assertFalse(calls[-1]['can_search'])
        self.assertEqual(len(calls[-1]['cases']), 2)
        self.assertNotIn('hidden', str(calls))

    def test_research_threshold_uses_all_gt_records_for_same_word(self):
        tasks = [{'case_id': k, 'keyword': w, 'gt_case_title': '标题', 'gt_case_content': '正文'}
                 for k, w in [('a', 'A'), ('b', 'a'), ('c', 'C')]]
        groups = {k: [{'id': k + str(i)} for i in range(2)] for k in 'abc'}
        class Model:
            def task(self, stage, data):
                return {'status': 'success', 'result': {'action': 'summarize', 'desc': '定义', 'relations': []}}
        with tempfile.TemporaryDirectory() as directory:
            results = enrich(tasks, groups, Model(), {'min_dialogues': 2, 'max_searches': 1,
                'top_k': 1, 'concurrency': 2}, canonical, save, Path(directory),
                lambda q: {'retrieval_result': {'top1': {'case_title': '标题', 'content': '正文'}}})
        self.assertEqual(set(results), {'a'})

    def test_one_hop_does_not_take_transitive_closure(self):
        clusters = [{'id': w, 'keyword': w, 'desc': w, 'aliases': [w], 'concepts': [w],
                     'records': [{'id': w, 'expected_case_id': w}]} for w in 'ABC']
        relations = [{'source': 'A', 'target': 'B', 'description': 'A使用B'},
                     {'source': 'B', 'target': 'C', 'description': 'B依赖C'}]
        expanded = expand(clusters, relations, canonical, digest)
        self.assertEqual({tuple(c['aliases']) for c in expanded}, {('A', 'B'), ('B', 'C')})
        self.assertTrue(all(c['count'] == 2 for c in expanded))
        self.assertEqual(len(expanded[0]['relations']), 1)

    def test_research_model_validation_and_whitelist(self):
        data = {'keyword': 'A', 'known_keywords': ['A', 'B'], 'cases': [{'title': '标题', 'content': '正文', 'case_id': 'secret'}],
                'previous_queries': ['A'], 'can_search': False}
        payload = model_input('research', data)
        self.assertNotIn('secret', str(payload))
        with self.assertRaises(ValueError):
            parse_result('research', '{"action":"search","query":"继续"}', payload)
        with self.assertRaises(ValueError):
            parse_result('research', '{"action":"summarize","desc":"定义","relations":[{"source":"A","target":"虚构","description":"关联"}]}', payload)

    def test_budget_is_enforced_even_if_model_keeps_searching(self):
        searches = []
        class Model:
            def task(self, stage, data):
                return {'status': 'success', 'result': {'action': 'search', 'query': '继续'}}
        def retrieve(q):
            searches.append(q)
            return {'retrieval_result': {}}
        result = investigate('A', [], ['A'], Model(), {'max_searches': 1, 'top_k': 2}, retrieve)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(searches, ['A'])
