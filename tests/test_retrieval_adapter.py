import json
import io
from contextlib import redirect_stderr
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import requests

from evaluate import GeneratedQueryRecord, RetrievalEvaluator
from retrieval_adapter import RetrievalAdapter, convert_results, create_server
from search import RetrievalRequestError, test_retrieval as retrieve


CASES = [
    {'caseID': 'KT002', 'case_name': '无线接入', 'text': '第二个案例的完整正文', 'score': 0.8,
     'metadata': {'source': 'internal'}},
    {'caseID': 'KT001', 'case_name': '网络故障', 'text': '第一个案例的完整正文', 'score': 0.9},
]


class AdapterTests(unittest.TestCase):
    def test_keeps_ranking_and_content_without_sorting_scores(self):
        result = convert_results(CASES, 10)['retrieval_result']
        self.assertEqual(list(result), ['top1', 'top2'])
        self.assertEqual(result['top1'], {'case_id': 'KT002', 'case_title': '无线接入',
                                        'content': '第二个案例的完整正文', 'score': 0.8})
        self.assertEqual(result['top2']['case_id'], 'KT001')
        self.assertEqual(len(convert_results(CASES, 1)['retrieval_result']), 1)
        self.assertEqual(convert_results([], 10), {'retrieval_result': {}})

    def test_passes_backend_options(self):
        backend = Mock(return_value=CASES)
        adapter = RetrievalAdapter(backend, top_k=5, strategy='semantic', index='custom',
                                   use_similar_question=True, use_chat=True, chat_content='完整对话')
        adapter.retrieve('查询')
        backend.assert_called_once_with(query='查询', top_k=5, strategy='semantic', index='custom',
                                        use_similar_question=True, use_chat=True, chat_content='完整对话')

    def test_rejects_malformed_upstream_instead_of_reporting_empty_hits(self):
        for cases in (None, {}, [None], [{}], [{**CASES[0], 'caseID': ''}],
                      [{**CASES[0], 'text': None}], [{**CASES[0], 'score': float('nan')}],
                      [{**CASES[0], 'score': True}]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                convert_results(cases, 10)
        for limit in (0, -1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                RetrievalAdapter(Mock(), top_k=limit)


class AdapterHTTPTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock(return_value=CASES)
        self.server = create_server('127.0.0.1', 0, backend=self.backend, top_k=5)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        self.thread.start()
        self.root = f'http://127.0.0.1:{self.server.server_port}'
        self.url = self.root + '/run_case_retrieval'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_existing_client_and_evaluator_work_without_changes(self):
        outer = self
        class Retriever:
            def retrieve(self, query):
                return retrieve(query, url=outer.url, timeout=2)
        record = GeneratedQueryRecord(sample_index=0, call_sno=None, expected_case_id='KT001',
                                      query='无线网络', status='success', error=None)
        result = RetrievalEvaluator(Retriever()).evaluate_query(record)
        self.assertEqual(result['retrieval_status'], 'success')
        self.assertEqual(result['matched_rank'], 2)
        self.assertEqual(retrieve('正文测试', url=self.url)['retrieval_result']['top1']['content'], CASES[0]['text'])
        self.assertEqual(self.backend.call_args.kwargs['top_k'], 5)

    def test_invalid_requests_do_not_call_backend(self):
        for body in (None, [], {}, {'retrieval_query': ''}, {'retrieval_query': '  '}, {'retrieval_query': 1}):
            with self.subTest(body=body):
                response = requests.post(self.url, data=json.dumps(body), timeout=2)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.json())
        self.assertEqual(requests.post(self.url, data='{invalid', timeout=2).status_code, 400)
        self.backend.assert_not_called()

    def test_health_and_unknown_route(self):
        self.assertEqual(requests.get(self.root + '/health', timeout=2).json(), {'status': 'ok'})
        self.assertEqual(requests.post(self.root + '/wrong', json={}, timeout=2).status_code, 404)
        self.backend.assert_not_called()

    def test_upstream_errors_are_http_failures(self):
        self.backend.side_effect = RuntimeError('secret connection details')
        response = requests.post(self.url, json={'retrieval_query': '查询'}, timeout=2)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('secret', response.text)
        with self.assertRaises(RetrievalRequestError):
            retrieve('查询', url=self.url, timeout=2)
        self.backend.side_effect = None
        self.backend.return_value = {'unexpected': 'object'}
        self.assertEqual(requests.post(self.url, json={'retrieval_query': '查询'}, timeout=2).status_code, 502)

    def test_debug_logs_type_error_traceback_only_on_server(self):
        from retrieval_adapter import handler, RetrievalAdapter
        self.server.RequestHandlerClass = handler(RetrievalAdapter(self.backend), debug=True)
        self.backend.side_effect = TypeError("retrieve_case() got an unexpected keyword argument 'index'")
        log = io.StringIO()
        with redirect_stderr(log):
            response = requests.post(self.url, json={'retrieval_query': '查询'}, timeout=2)
        self.assertEqual(response.status_code, 502)
        self.assertIn('Traceback (most recent call last)', log.getvalue())
        self.assertIn("unexpected keyword argument 'index'", log.getvalue())
        self.assertNotIn('unexpected keyword', response.text)

    def test_multiple_requests_keep_queries_independent(self):
        self.backend.side_effect = lambda **kw: [{**CASES[0], 'caseID': kw['query']}]
        def call(i):
            return retrieve(str(i), url=self.url, timeout=2)['retrieval_result']['top1']['case_id']
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(call, range(8))), list(map(str, range(8))))


if __name__ == '__main__':
    unittest.main()
