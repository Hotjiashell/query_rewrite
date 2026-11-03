"""Local knowledge review UI. Run: python3 -m extract_knowledge.annotate [run_dir]."""
from __future__ import annotations
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
import uuid

if __package__:
    from .pipeline import ROOT, HERE, read_json, save
else:
    from pipeline import ROOT, HERE, read_json, save


class Review:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.lock = threading.Lock()
        self.entries = read_json(self.directory / 'knowledge.json')
        if not isinstance(self.entries, list):
            raise ValueError('knowledge.json 必须为数组')
        snapshot = read_json(self.directory / 'existing_knowledge.json')
        existing = snapshot.get('content', '')
        if not isinstance(existing, str):
            raise ValueError('existing_knowledge.json 的 content 必须为字符串')
        self.fingerprint = hashlib.sha256(json.dumps([self.entries, existing], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        clusters = {}
        for name in ('04_keyword_clusters.json', '04_merged_keyword_clusters.json', '06_retained_clusters.json'):
            path = self.directory / name
            if path.exists():
                for cluster in read_json(path):
                    clusters[cluster['id']] = cluster
        self.candidates = []
        for i, entry in enumerate(self.entries):
            cluster = clusters.get(entry.get('cluster_id'), {})
            self.candidates.append({**entry, 'review_id': str(i), 'records': cluster.get('records', [])})
        self.path = self.directory / 'annotation_state.json'
        if self.path.exists():
            self.state = read_json(self.path)
            if self.state.get('fingerprint') != self.fingerprint:
                raise ValueError('run 的知识文件已变化，与 annotation_state.json 不一致；请换新 run 目录或备份并移走旧标注状态')
            self.validate(self.state)
        else:
            confirmed = []
            for line in existing.splitlines():
                value = re.sub(r'^\s*(?:\d+[.、．)]\s*|[-*]\s+)', '', line).strip()
                if value:
                    confirmed.append({'id': uuid.uuid4().hex, 'text': value, 'source': 'existing'})
            self.state = {'fingerprint': self.fingerprint, 'revision': 0, 'index': 0,
                          'confirmed': confirmed, 'drafts': {}, 'statuses': {}}
            save(self.path, self.state)

    def validate(self, state):
        if not isinstance(state, dict) or state.get('fingerprint') != self.fingerprint:
            raise ValueError('标注数据不属于当前 run')
        if type(state.get('revision')) is not int or type(state.get('index')) is not int:
            raise ValueError('revision / index 必须为整数')
        if not 0 <= state['index'] < max(1, len(self.entries)):
            raise ValueError('页码超出范围')
        if not isinstance(state.get('confirmed'), list):
            raise ValueError('confirmed 必须为数组')
        ids = set()
        for item in state['confirmed']:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str) or item['id'] in ids:
                raise ValueError('确定知识 ID 无效或重复')
            ids.add(item['id'])
            if not isinstance(item.get('text'), str) or not item['text'].strip():
                raise ValueError('确定知识内容不能为空')
        for field in ('drafts', 'statuses'):
            values = state.get(field)
            if not isinstance(values, dict):
                raise ValueError(f'{field} 必须为对象')
            for key, value in values.items():
                if key not in {str(i) for i in range(len(self.entries))} or not isinstance(value, str):
                    raise ValueError(f'{field} 包含无效条目')
                if field == 'statuses' and value not in ('added', 'skipped'):
                    raise ValueError('无效审核状态')

    def update(self, state, export=False):
        with self.lock:
            self.validate(state)
            if state['revision'] != self.state['revision']:
                raise ValueError('另一个页面已修改标注，请刷新后继续')
            updated = {**state, 'revision': state['revision'] + 1}
            save(self.path, updated)
            self.state = updated
            paths = []
            if export:
                # Text is directly reusable as the next run's existing knowledge.
                target = self.directory / 'final_knowledge.txt'
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile('w', encoding='utf-8', dir=self.directory, delete=False) as f:
                    f.write('\n'.join(f"{i}. {r['text'].strip()}" for i, r in enumerate(updated['confirmed'], 1)) + '\n')
                    temporary = Path(f.name)
                temporary.replace(target)
                save(self.directory / 'final_knowledge.json', {'knowledge': updated['confirmed'],
                      'fingerprint': self.fingerprint, 'revision': updated['revision']})
                paths = [str(target), str(self.directory / 'final_knowledge.json')]
            return {'revision': updated['revision'], 'paths': paths}


def handler(review):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            payload = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/api/data':
                with review.lock:
                    self.reply(200, {'run_dir': str(review.directory), 'candidates': review.candidates, 'state': review.state})
            elif self.path in ('/', '/index.html'):
                data = (HERE / 'annotation/index.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path not in ('/api/save', '/api/export'):
                return self.reply(404, {'error': 'Not found'})
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.reply(403, {'error': '不允许跨站请求'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(400, {'error': '需要 JSON 请求'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 20_000_000:
                    raise ValueError('请求大小无效')
                state = json.loads(self.rfile.read(length))
                self.reply(200, review.update(state, self.path == '/api/export'))
            except (ValueError, OSError, TypeError, KeyError) as exc:
                self.reply(400, {'error': str(exc)})

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', nargs='?', help='省略时读取 config.json 的 output_dir')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        directory = args.run_dir
        if not directory:
            directory = read_json(HERE / 'config.json')['extract_knowledge']['output_dir']
            directory = ROOT / directory
        review = Review(directory)
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(review))
        print(f'标注目录：{review.directory}\n打开 http://127.0.0.1:{args.port} （Ctrl+C 停止）', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'无法启动标注界面：{exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
