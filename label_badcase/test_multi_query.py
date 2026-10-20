"""Multi-query trials reuse generation/fusion without modifying annotations."""
import copy
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from server import ROOT, Store, build_dataset, handler


class MultiQueryTrialTests(unittest.TestCase):
    def setUp(self):
        demo = json.loads((ROOT / 'demo_data.json').read_text())
        records = build_dataset(demo['badcases'], demo['analysis'], demo['cases'])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(records, Path(self.tmp.name) / 'labels.json', {})
        self.gt = self.store.by_key['12']['gt']['case_id']
        self.request = {'key': '12', 'prompt': '生成查询：{dialogue}', 'fusion_method': 'round_robin', 'top_k': 3}
        self.calls = []
        self.store.query_generator = lambda prompt, dialogue: self.calls.append((prompt, dialogue)) or ['q1', 'q2']

        def retrieve(query):
            items = [('A', .4), (self.gt, .8)] if query == 'q1' else [('A', .9), ('B', .7)]
            return {'retrieval_result': {f'top{i}': {'case_id': cid, 'case_title': query + cid,
                'score': score, 'content': query + '-正文'} for i, (cid, score) in enumerate(items, 1)}}
        self.store.retriever = retrieve

    def test_round_robin_provenance_gt_and_no_writes(self):
        before = copy.deepcopy(self.store.state)
        result = self.store.try_multi_query(self.request)
        self.assertEqual(self.calls, [(self.request['prompt'], self.store.by_key['12']['chat_content'])])
        self.assertEqual(result['queries'], ['q1', 'q2'])
        self.assertEqual([r['query'] for r in result['query_results']], ['q1', 'q2'])
        self.assertEqual([c['case_id'] for c in result['fused']['cases']], ['A', 'B', self.gt])
        self.assertEqual([c['rank'] for c in result['fused']['cases']], [1, 2, 3])
        self.assertEqual(result['query_results'][0]['matched_rank'], 2)
        self.assertTrue(result['fused']['gt_hit'])
        self.assertEqual(result['fused']['matched_rank'], 3)
        self.assertEqual([s['query_index'] for s in result['fused']['cases'][0]['sources']], [1, 2])
        self.assertEqual(self.store.state, before)
        self.assertFalse(self.store.output.exists())

    def test_score_uses_best_score_and_matching_body(self):
        result = self.store.try_multi_query({**self.request, 'fusion_method': 'score', 'top_k': 2})
        self.assertEqual([c['case_id'] for c in result['fused']['cases']], ['A', self.gt])
        self.assertEqual(result['fused']['cases'][0]['score'], .9)
        self.assertEqual(result['fused']['cases'][0]['text'], 'q2-正文')

    def test_partial_all_failed_and_empty_results(self):
        def partial(query):
            if query == 'q1':
                raise TimeoutError('检索超时')
            return {'retrieval_result': {}}
        self.store.retriever = partial
        result = self.store.try_multi_query(self.request)
        self.assertEqual(result['status'], 'partial_success')
        self.assertIn('检索超时', result['query_results'][0]['error'])
        self.assertEqual(result['fused']['cases'], [])
        self.store.retriever = lambda query: None
        result = self.store.try_multi_query(self.request)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['queries'], ['q1', 'q2'])
        self.assertEqual(len(result['query_results']), 2)

    def test_validation_before_model_and_generated_query_limits(self):
        for body in [[], {**self.request, 'key': 'missing'}, {**self.request, 'prompt': 'no placeholder'},
                     {**self.request, 'top_k': True}, {**self.request, 'top_k': 101},
                     {**self.request, 'fusion_method': 'invalid'}]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.store.try_multi_query(body)
        self.assertFalse(self.calls)
        for queries in [[], [''], ['a'] * 4, 'query', [123]]:
            self.store.query_generator = lambda *args: queries
            with self.subTest(queries=queries), self.assertRaises(RuntimeError):
                self.store.try_multi_query(self.request)
        self.store.query_generator = lambda *args: [' q1 ', 'q1']
        self.assertEqual(self.store.try_multi_query(self.request)['queries'], ['q1'])

    def test_model_errors_hide_sensitive_exception_details(self):
        def fail(*args):
            raise RuntimeError('secret-token at private-url')
        self.store.query_generator = fail
        with self.assertRaises(RuntimeError) as error:
            self.store.try_multi_query(self.request)
        self.assertNotIn('secret-token', str(error.exception))
        self.assertIn('生成 query 失败', str(error.exception))

    def test_http_route_success_validation_and_model_failure(self):
        handler_type = handler(self.store)
        def request(body):
            instance = object.__new__(handler_type)
            instance.path = '/api/multi-query'
            payload = json.dumps(body).encode()
            instance.headers = {'Content-Length': str(len(payload))}
            instance.rfile = BytesIO(payload)
            instance.respond = Mock()
            instance.do_POST()
            return instance.respond.call_args.args
        code, result = request(self.request)
        self.assertEqual(code, 200)
        self.assertEqual(result['record_key'], '12')
        self.assertEqual(request({**self.request, 'prompt': 'invalid'})[0], 400)
        self.store.query_generator = Mock(side_effect=RuntimeError('failed'))
        self.assertEqual(request(self.request)[0], 502)

    def test_config_and_prompt_reuse_existing_generator(self):
        from label_badcase.multi_query import generate_queries
        config = Path(self.tmp.name) / 'config.json'
        config.write_text(json.dumps({'llm': {'base_url': 'http://test/v1', 'model_name': 'test-model',
            'api_key_env': 'TRIAL_TEST_KEY', 'temperature': .2}}))
        raw_client = Mock()
        raw_client.with_options.return_value = raw_client
        raw_client.chat.completions.create.return_value.choices = [Mock(message=Mock(content='{"query":["one","two","one","three","four"]}'))]
        with patch.dict('os.environ', {'TRIAL_TEST_KEY': 'test-key'}), patch('label_badcase.multi_query.build_openai_client', return_value=raw_client) as factory:
            result = generate_queries('对话：{dialogue} 输出 {"query":[]}', '当前对话', config)
        self.assertEqual(result, ['one', 'two', 'three'])
        self.assertEqual(factory.call_args.args[0].api_key, 'test-key')
        request = raw_client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['messages'][0]['content'], '对话：当前对话 输出 {"query":[]}')
        self.assertEqual(request['extra_body']['chat_template_kwargs']['enable_thinking'], False)
        self.assertEqual(request['temperature'], .2)
        self.assertEqual(raw_client.with_options.call_args.kwargs, {'timeout': 60, 'max_retries': 0})


if __name__ == '__main__':
    unittest.main()
