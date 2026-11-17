import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from .research import CandidateIndex, investigate, enrich, expand
from .pipeline import Model, HERE, canonical, digest, save, parse_result, model_input, load_prompts, read_json


class ResearchTests(unittest.TestCase):
    @staticmethod
    def candidate_tasks(words):
        return [{'keyword': w, 'gt_case_title': w + '案例', 'gt_case_content': '上下文' + w} for w in words]

    def test_direct_matching_normalization_and_ascii_boundaries(self):
        index = CandidateIndex(self.candidate_tasks(['当前词', 'VPN', 'VP', '企业账号', '企业账号', '无关']))
        selected, trace = index.select('当前词', [{'title': '当前词', 'content': '使用ｖｐｎ和企业账号，VPX也可用'}], [], {})
        self.assertEqual(selected, ['VPN', '企业账号'])
        self.assertTrue(all(row['method'] == 'direct' for row in trace['added']))

    def test_embedding_supplements_direct_match_using_gt_context(self):
        tasks = self.candidate_tasks(['主体', '直接词', '语义近', '语义远'])
        tasks[2]['gt_case_content'] = '认证功能'
        def embed(texts):
            return {t: ([1, 0] if '认证' in t else [0, 1]) for t in texts}
        index = CandidateIndex(tasks, embed)
        added, trace = index.select('主体', [{'title': '认证', 'content': '使用直接词'}], [], {})
        self.assertEqual(added, ['直接词', '语义近', '语义远'])
        self.assertEqual([r['method'] for r in trace['added']], ['direct', 'embedding', 'embedding'])
        self.assertGreater(trace['added'][1]['score'], trace['added'][2]['score'])

    def test_many_direct_matches_consume_round_budget_before_embedding(self):
        words = [f'词{i:02}' for i in range(45)]
        embed = Mock(side_effect=lambda texts: {t: [1, 0] for t in texts})
        index = CandidateIndex(self.candidate_tasks(words), embed)
        cases = [{'title': '词表', 'content': '、'.join(words)}]
        first, trace = index.select('主体', cases, [], {})
        second, _ = index.select('主体', cases, first, {})
        self.assertEqual(first, words[:20])
        self.assertEqual(second, words[20:40])
        self.assertTrue(all(row['method'] == 'direct' for row in trace['added']))
        self.assertEqual(embed.call_count, 1)  # No evidence embedding needed.

    def test_round_limit_total_limit_and_accumulation_through_retrieval(self):
        words = ['主体'] + [f'候选{i:02}' for i in range(61)]
        index = CandidateIndex(self.candidate_tasks(words), lambda texts: {t: [1, 0] for t in texts})
        seen = []
        class FakeModel:
            def task(self, stage, data):
                seen.append(data['known_keywords'])
                if data['can_search']:
                    return {'status': 'success', 'result': {'action': 'search', 'query': '下一轮'}}
                return {'status': 'success', 'result': {'action': 'summarize', 'desc': '定义', 'relations': []}}
        retrieve = Mock(side_effect=[
            {'retrieval_result': {}},
            {'retrieval_result': {'top1': {'case_title': '发现新词', 'content': '候选60'}}},
            {'retrieval_result': {'top1': {'case_title': '另一个新词', 'content': '候选59'}}},
        ])
        result = investigate('主体', [{'title': '主体', 'content': '初始材料'}], words, FakeModel(),
                             {'max_searches': 3, 'top_k': 1}, retrieve, index)
        self.assertEqual(result['status'], 'success')
        self.assertEqual([len(row) for row in seen], [20, 40, 40])
        self.assertEqual(seen[1][:20], seen[0])
        self.assertEqual(seen[1][20], '候选60')  # New direct match outranks semantic additions.
        self.assertEqual(seen[2], seen[1])
        self.assertNotIn('主体', seen[2])
        self.assertEqual(len(set(seen[2])), 40)
        self.assertEqual([len(s['added']) for s in result['candidate_selections']], [20, 20, 0])
        self.assertTrue(all(isinstance(w, str) for row in seen for w in row))
        with self.assertRaises(ValueError):
            parse_result('research', '{"action":"summarize","desc":"定义","relations":'
                         '[{"source":"主体","target":"候选59","description":"未入选的关系"}]}',
                         {'keyword': '主体', 'known_keywords': seen[-1], 'can_search': False})

    def test_long_case_tail_and_reuse_of_evidence_embeddings(self):
        tasks = self.candidate_tasks(['主体', '尾部相关'])
        tasks[1]['gt_case_content'] = 'x' * 1000 + '认证'
        embed = Mock(side_effect=lambda texts: {t: ([1, 0] if '认证' in t else [0, 1]) for t in texts})
        index = CandidateIndex(tasks, embed)
        cases, cache = [{'title': '主体', 'content': '认证'}], {}
        added, trace = index.select('主体', cases, [], cache)
        self.assertEqual(added, ['尾部相关'])
        self.assertAlmostEqual(trace['added'][0]['score'], 1)
        index.select('主体', cases, [], cache)
        self.assertEqual(embed.call_count, 2)  # Index once, evidence once.

    def test_embedding_failure_preserves_direct_matching_and_reports_degradation(self):
        for fail_index in (False, True):
            with self.subTest(fail_index=fail_index):
                calls = []
                def embed(texts):
                    calls.append(texts)
                    if fail_index or len(calls) > 1:
                        raise RuntimeError('offline')
                    return {t: [1, 0] for t in texts}
                index = CandidateIndex(self.candidate_tasks(['主体', '直接词', '语义词']), embed)
                model = Mock()
                model.task.return_value = {'status': 'success', 'result': {'action': 'summarize', 'desc': '定义', 'relations': []}}
                result = investigate('主体', [{'title': '主体', 'content': '使用直接词'}], [], model,
                                     {'max_searches': 1, 'top_k': 1}, lambda q: {'retrieval_result': {}}, index)
                self.assertEqual(result['status'], 'success')
                self.assertEqual(model.task.call_args.args[1]['known_keywords'], ['直接词'])
                self.assertTrue(result['candidate_embedding_failed'])

    def test_empty_candidates_do_not_call_embedding(self):
        embed = Mock()
        index = CandidateIndex(self.candidate_tasks(['主体']), embed)
        added, _ = index.select('主体', [{'title': '主体', 'content': '正文'}], [], {})
        self.assertEqual(added, [])
        embed.assert_not_called()

    def test_pipeline_wires_embedding_and_marks_selection_failure_partial(self):
        from . import pipeline as p
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                p.save(root / 'bad.json', {'artifact_type': 'badcase_report', 'records': [
                    {'expected_case_id': key, 'chat_content': f'对话{i}', 'gt_case_title': key,
                     'gt_case_content': '主体依赖直接词'} for key, count in [('主体', 2), ('直接词', 1)]
                    for i in range(count)]})
                (root / 'existing.txt').write_text('', encoding='utf-8')
                args = SimpleNamespace(output_dir=str(root / 'out'), knowledge=str(root / 'existing.txt'),
                    cases=None, dialogues=None, badcases=[str(root / 'bad.json')], keyword_batch_size=10,
                    concurrency=1, max_examples=10, min_cluster_size=0, precluster_similarity_threshold=0.85,
                    research={'enabled': True, 'min_dialogues': 1, 'max_searches': 1, 'top_k': 1,
                              'concurrency': 1, 'retrieval_url': 'http://fake', 'timeout': 1})
                class FakeModel:
                    identity, templates = {}, {}
                    def __init__(self, *args):
                        pass
                    def task(self, stage, data):
                        if stage == 'keywords':
                            result = {'keywords': [{'keyword': data['gt_case_title'], 'reason': '依据'}]}
                        elif stage == 'research':
                            self_test.assertEqual(data['known_keywords'], ['直接词'])
                            result = {'action': 'summarize', 'desc': '主体定义', 'relations': []}
                        elif stage == 'describe':
                            result = {'desc': '普通定义'}
                        else:
                            result = {'covered': True, 'reason': '已覆盖', 'matched_knowledge': '已有条目'}
                        return {'status': 'success', 'result': result}
                self_test = self
                def embed_texts(texts, *args, **kwargs):
                    self.assertEqual(kwargs['namespace'], 'research_embedding')
                    return ({}, [{'error_type': 'RuntimeError'}]) if failure else ({t: [1, 0] for t in texts}, [])
                with patch.object(p, 'Model', FakeModel), patch.object(p, 'precluster', return_value=([], [], 0)), \
                     patch('extract_knowledge.embedding.embed_texts', side_effect=embed_texts) as embeddings, \
                     patch('search.test_retrieval', return_value={'retrieval_result': {}}):
                    self.assertEqual(p.run(args, {}), int(failure))
                self.assertEqual(embeddings.call_count, 1 if failure else 2)
                self.assertEqual(read_json(root / 'out/summary.json')['partial'], failure)
                result = read_json(root / 'out/03_research_results.json')['主体']
                self.assertEqual(result['status'], 'success')
                self.assertEqual(result['candidate_embedding_failed'], failure)

    def test_react_search_observation_and_summary_with_real_parser(self):
        responses = [
            'Thought: 源案例提到A使用B，但B的用途还不明确，需要补充证据。\nAction:\n'
            '```json\n{"action":"search","query":"A B 用途"}\n```',
            'Thought: 新案例明确说明B负责认证，可以总结A的用途和依赖关系。\nAction:\n'
            '```json\n{"action":"summarize","desc":"A是使用B认证的内部系统",'
            '"relations":[{"source":"A","target":"B","description":"A使用B进行认证"}]}\n```',
        ]
        create = Mock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=raw))]) for raw in responses])
        retrieve = Mock(side_effect=[
            {'retrieval_result': {}},
            {'retrieval_result': {'top1': {'case_title': 'A认证', 'content': 'A使用B进行认证'}}},
        ])
        with tempfile.TemporaryDirectory() as directory:
            model = object.__new__(Model)
            model.args = SimpleNamespace(output_dir=directory, max_tokens=4096, resume=True, retries=0)
            model.model, model.identity = 'fake', {'model_name': 'fake', 'temperature': 0}
            model.templates = load_prompts(HERE / 'prompts.md')
            model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            result = investigate('A', [{'title': 'A入口', 'content': 'A使用B'}], ['A', 'B'],
                                 model, {'max_searches': 2, 'top_k': 1}, retrieve)
            self.assertEqual(result['status'], 'success')
            self.assertEqual([call.args[0] for call in retrieve.call_args_list], ['A', 'A B 用途'])
            first, second = [r['input'] for r in result['rounds']]
            self.assertEqual(len(first['cases']), 1)
            self.assertEqual(second['cases'][-1], {'title': 'A认证', 'content': 'A使用B进行认证'})
            self.assertEqual(second['previous_queries'], ['A', 'A B 用途'])
            self.assertFalse(second['can_search'])
            self.assertEqual(result['result']['relations'][0]['target'], 'B')
            traces = [read_json(path) for path in (Path(directory) / 'calls/research').glob('*.json')]
            self.assertCountEqual([trace['attempts'][0]['response'] for trace in traces], responses)
            self.assertTrue(model.task('research', second)['cached'])
            self.assertEqual(create.call_count, 2)

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
            parse_result('research', 'Thought: 还想查询。\nAction:\n```json\n'
                         '{"action":"search","query":"继续"}\n```', payload)
        with self.assertRaises(ValueError):
            parse_result('research', 'Thought: 准备总结。\nAction:\n```json\n'
                         '{"action":"summarize","desc":"定义","relations":[{"source":"A","target":"虚构","description":"关联"}]}\n```', payload)

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
