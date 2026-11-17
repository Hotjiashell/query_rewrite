import tempfile
import unittest
from pathlib import Path
from .feedback import refine
from .pipeline import model_input, save


class FeedbackTests(unittest.TestCase):
    def run_feedback(self, reject=False, broken=False, retrieval_miss=False, max_entries=0, entry_count=1):
        rows = [{'id': str(i), 'expected_case_id': 'target', 'chat_content': f'用户诉求{i}',
                 'gt_case_title': '目标标题', 'gt_case_content': '依据', 'queries': ['历史query']} for i in range(12)]
        calls = []
        class Model:
            def task(self, stage, data):
                calls.append(stage)
                if stage == 'feedback_query':
                    assert set(data) == {'knowledge', 'chat_content'}
                    return {'status': 'success', 'result': {'queries': [data['knowledge']]}}
                if stage == 'feedback_review':
                    assert all(set(e) == {'chat_content', 'gt_case_title'} for e in data['examples'])
                    return {'status': 'success', 'result': {'approved': not reject and (retrieval_miss or data['knowledge'] == '归纳规则'), 'reason': '请归纳共同语义'}}
                assert len(data['examples']) + len(data['retrieval_failures']) == 10
                assert data['existing_knowledge'] == '已有知识'
                assert ('review' in data) == (not retrieval_miss)
                if retrieval_miss:
                    assert len(data['retrieval_failures']) == 10
                assert data['keyword'] == '内部词'
                assert data['desc'] == '描述解释'
                assert all(set(e) == {'chat_content', 'gt_case_title', 'queries'} and e['queries'] for e in data['examples'])
                assert 'record_id' not in str(data)
                return {'status': 'success', 'result': {'knowledge': '归纳规则'}}
        def retrieve(query, **kwargs):
            calls.append('retrieve')
            if broken:
                raise RuntimeError('service unavailable')
            return {'retrieval_result': {'top1': {'case_id': 'target' if '归纳规则' in query else 'other', 'case_title': '标题'}}}
        with tempfile.TemporaryDirectory() as directory:
            prompt_file = Path(directory) / 'prompt.txt'
            knowledge_file = Path(directory) / 'existing.txt'
            prompt_file.write_text('知识：{knowledge}')
            knowledge_file.write_text('已有知识')
            result = refine([{'cluster_id': f'cluster{i}', 'knowledge': '罗列现象'} for i in range(entry_count)],
                            [{'id': f'cluster{i}', 'keyword': '内部词', 'desc': '描述解释', 'records': rows}
                             for i in range(entry_count)], Model(), Model(),
                            {'max_entries': max_entries, 'max_rounds': 2, 'top_k': 10, 'min_recall': 1, 'concurrency': 2, 'timeout': 1, 'prompt_file': str(prompt_file), 'knowledge_file': str(knowledge_file)},
                            '', save, Path(directory), retrieve)
            import json
            history = json.loads((Path(directory) / '08_feedback_results.json').read_text())
            skipped = json.loads((Path(directory) / 'feedback_skipped_knowledge.json').read_text())
            self.assertEqual(len(skipped), max(0, entry_count - max_entries) if max_entries else 0)
            if not retrieval_miss:
                self.assertEqual(history[0]['rounds'][0]['tests'], [])
                self.assertIsNone(history[0]['rounds'][0]['recall'])
                self.assertEqual(calls[:3], ['feedback_review', 'feedback_revise', 'feedback_review'])
            if reject:
                self.assertNotIn('feedback_query', calls)
                self.assertNotIn('retrieve', calls)
            return result, history

    def test_limit_reviews_prefix_and_preserves_unvalidated_remainder(self):
        (entries, rejected, errors), history = self.run_feedback(max_entries=1, entry_count=2)
        self.assertEqual((rejected, errors), (0, 0))
        self.assertEqual([h['cluster_id'] for h in history], ['cluster0'])
        self.assertEqual([e['cluster_id'] for e in entries], ['cluster0', 'cluster1'])
        self.assertEqual(entries[0]['knowledge'], '归纳规则')
        self.assertTrue(entries[0]['validation']['passed'])
        self.assertEqual(entries[1]['knowledge'], '罗列现象')
        self.assertTrue(entries[1]['validation']['skipped'])
        self.assertIsNone(entries[1]['validation']['passed'])

    def test_limit_does_not_replace_rejected_entries_with_lower_ranks(self):
        (entries, rejected, errors), history = self.run_feedback(reject=True, max_entries=1, entry_count=2)
        self.assertEqual((rejected, errors), (1, 0))
        self.assertEqual([h['cluster_id'] for h in history], ['cluster0'])
        self.assertEqual([e['cluster_id'] for e in entries], ['cluster1'])
        self.assertTrue(entries[0]['validation']['skipped'])

    def test_zero_or_large_limit_validates_all_entries(self):
        for limit in (0, 10):
            with self.subTest(limit=limit):
                (entries, rejected, errors), history = self.run_feedback(max_entries=limit, entry_count=2)
                self.assertEqual((len(entries), rejected, errors, len(history)), (2, 0, 0, 2))
                self.assertTrue(all(e['validation']['passed'] for e in entries))

    def test_revision_must_pass_both_gates_and_tests_full_cluster(self):
        (entries, rejected, errors), history = self.run_feedback()
        self.assertEqual((len(entries), rejected, errors), (1, 0, 0))
        self.assertEqual(entries[0]['knowledge'], '归纳规则')
        self.assertEqual(len(history[0]['rounds']), 2)
        self.assertEqual(len(history[0]['rounds'][1]['tests']), 12)

    def test_quality_rejection_skips_query_and_retrieval(self):
        (entries, rejected, errors), history = self.run_feedback(reject=True)
        self.assertEqual((entries, rejected, errors), ([], 1, 0))
        self.assertTrue(all(r['tests'] == [] and r['retrieval_status'] == 'skipped' for r in history[0]['rounds']))

    def test_service_failure_is_not_quality_feedback(self):
        (entries, rejected, errors), history = self.run_feedback(broken=True)
        self.assertEqual((entries, rejected, errors), ([], 1, 1))
        self.assertEqual(len(history[0]['rounds']), 2)

    def test_whitelist_strips_identifiers(self):
        payload = model_input('feedback_revise', {'knowledge': '规则', 'examples': [
            {'id': 'secret', 'expected_case_id': 'secret', 'chat_content': '诉求'}],
            'review': '评审', 'retrieval_failures': [{'record_id': 'secret', 'queries': ['q'], 'retrieved_titles': ['标题']}]})
        self.assertNotIn('secret', str(payload))

    def test_custom_prompt_is_exact_and_query_parser_matches_evaluate(self):
        from . import pipeline as p
        from types import SimpleNamespace
        from unittest.mock import Mock
        from gen_query import extract_queries
        raw = '分析后返回：\n```json\n{"query": ["无线网络故障", "企业无线接入"]}\n```'
        with tempfile.TemporaryDirectory() as directory:
            model = p.Model.__new__(p.Model)
            model.args = SimpleNamespace(output_dir=directory, resume=False, max_tokens=100, retries=0)
            model.templates = {}
            model.identity = {'model_name': 'fake', 'base_url': 'fake', 'temperature': 0}
            model.model = 'fake'
            model.feedback_query_template = '规则：{knowledge}\n对话：{dialogue}\n输出示例：{"query":"检索词"}'
            client = Mock()
            client.chat.completions.create.return_value = SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')])
            model.client = client
            trace = model.task('feedback_query', {'knowledge': '已有知识\n新知识',
                                'chat_content': '对话保留{knowledge}', 'expected_case_id': 'secret'})
            self.assertEqual(trace['status'], 'success')
            self.assertEqual(trace['result']['queries'], extract_queries(raw))
            self.assertEqual(trace['prompt'], '规则：已有知识\n新知识\n对话：对话保留{knowledge}\n输出示例：{"query":"检索词"}')
            self.assertNotIn('secret', trace['prompt'])
            model.args.resume = True
            model.task('feedback_query', {'knowledge': '已有知识\n新知识', 'chat_content': '对话保留{knowledge}'})
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_evaluate_metrics_retained_in_feedback(self):
        (_, _, _), history = self.run_feedback()
        test = history[0]['rounds'][-1]['tests'][0]
        self.assertEqual(test['evaluation']['matched_rank'], 1)
        self.assertTrue(test['evaluation']['union_top_k']['10'])
        self.assertEqual(test['evaluation']['retrieval_status'], 'success')

    def test_review_whitelist_excludes_case_body(self):
        payload = model_input('feedback_review', {'knowledge': '知识', 'examples': [
            {'chat_content': '对话', 'gt_case_title': '标题', 'gt_case_content': '正文', 'id': '编号'}]})
        self.assertEqual(payload, {'knowledge': '知识', 'examples': [
            {'chat_content': '对话', 'gt_case_title': '标题'}]})

    def test_feedback_templates_loaded_from_shared_markdown(self):
        from . import pipeline as p
        templates = p.load_prompts(p.HERE / 'prompts.md')
        self.assertIn('feedback_review', templates)
        self.assertIn('feedback_revise', templates)
        raw = '先归纳共同诉求。\n```json\n{"knowledge":"新归纳知识"}\n```'
        self.assertEqual(p.parse_result('feedback_revise', raw, {}), {'knowledge': '新归纳知识'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.md'
            original = (p.HERE / 'prompts.md').read_text()
            path.write_text(original.replace('概括、泛用性还不是很好', '泛化能力需要提升'))
            self.assertNotEqual(p.load_prompts(path)['feedback_revise'], templates['feedback_revise'])

    def test_retrieval_feedback_is_bounded_and_omits_approved_review(self):
        (entries, rejected, errors), history = self.run_feedback(retrieval_miss=True)
        self.assertEqual((len(entries), rejected, errors), (1, 0, 0))
        self.assertEqual(len(history[0]['rounds'][0]['tests']), 12)
