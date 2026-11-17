"""Serve new_retrieve.retrieve_case through the evaluator's retrieval protocol."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import sys
import traceback
from typing import Any, Callable


def convert_results(cases: Any, top_k: int) -> dict[str, Any]:
    """Keep upstream ranking and case IDs while translating the field names."""
    if not isinstance(cases, list):
        raise ValueError('retrieve_case must return a list')
    results = {}
    for rank, case in enumerate(cases[:top_k], 1):
        if not isinstance(case, dict):
            raise ValueError('each retrieved case must be an object')
        case_id = case.get('caseID')
        if type(case_id) not in (str, int) or not str(case_id).strip():
            raise ValueError('each retrieved case requires a nonempty caseID')
        if any(not isinstance(case.get(field), str) for field in ('case_name', 'text')):
            raise ValueError('each retrieved case requires string case_name and text')
        score = case.get('score')
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('case score must be a finite number or null')
        results[f'top{rank}'] = {
            'case_id': str(case_id), 'case_title': case['case_name'],
            'content': case['text'], 'score': score,
        }
    return {'retrieval_result': results}


@dataclass
class RetrievalAdapter:
    backend: Callable[..., list[dict[str, Any]]]
    top_k: int = 10
    strategy: str = 'lexical&semantic'
    index: str = 'document_12'
    use_similar_question: bool = False
    use_chat: bool = False
    chat_content: str = ''

    def __post_init__(self):
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError('top_k must be a positive integer')
        if not self.strategy.strip() or not self.index.strip():
            raise ValueError('strategy and index must not be empty')

    def retrieve(self, query: str) -> dict[str, Any]:
        cases = self.backend(
            query=query, top_k=self.top_k, strategy=self.strategy, index=self.index,
            use_similar_question=self.use_similar_question,
            use_chat=self.use_chat, chat_content=self.chat_content,
        )
        return convert_results(cases, self.top_k)


def handler(adapter: RetrievalAdapter, *, debug: bool = False):
    class Handler(BaseHTTPRequestHandler):
        # Bound reads of incomplete HTTP requests; backend timeouts belong in retrieve_case.
        timeout = 30

        def send_json(self, status: int, payload: dict[str, Any]):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {'status': 'ok'})
            else:
                self.send_json(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/run_case_retrieval':
                self.send_json(404, {'error': 'not_found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_json(400, {'error': 'invalid_content_length'})
                return
            if not 0 < length <= 1024 * 1024:
                self.send_json(413 if length > 1024 * 1024 else 400, {'error': 'invalid_body_size'})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                self.send_json(400, {'error': 'invalid_json'})
                return
            query = payload.get('retrieval_query') if isinstance(payload, dict) else None
            if not isinstance(query, str) or not query.strip():
                self.send_json(400, {'error': 'retrieval_query must be a nonempty string'})
                return
            try:
                result = adapter.retrieve(query)
            except Exception as exc:
                # Keep connection details and credentials in upstream exceptions out of responses.
                print(f'[adapter] 检索失败 error_type={type(exc).__name__}', file=sys.stderr, flush=True)
                if debug:
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                self.send_json(502, {'error': 'upstream_retrieval_failed', 'error_type': type(exc).__name__})
                return
            self.send_json(200, result)

    return Handler


def create_server(host: str = '127.0.0.1', port: int = 8277, *, backend=None, debug=False, **options):
    if backend is None:
        from new_retrieve import retrieve_case
        backend = retrieve_case
    adapter = RetrievalAdapter(backend, **options)
    return ThreadingHTTPServer((host, port), handler(adapter, debug=debug))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8277)
    parser.add_argument('--top-k', type=int, default=10, help='新检索接口返回的案例数量')
    parser.add_argument('--strategy', default='lexical&semantic')
    parser.add_argument('--index', default='document_12')
    parser.add_argument('--use-similar-question', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use-chat', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--chat-content', default='', help='固定传给新接口的 chat_content；旧协议不携带对话')
    parser.add_argument('--debug', action='store_true', help='在服务端 stderr 打印完整异常堆栈，HTTP 响应仍仅返回错误类型')
    args = parser.parse_args(argv)
    try:
        server = create_server(**vars(args))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f'启动失败：{exc}\n')
    address, port = server.server_address
    print(f'检索转接服务：http://{address}:{port}/run_case_retrieval '
          f'index={args.index} top_k={args.top_k}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
