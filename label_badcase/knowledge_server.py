"""Independent knowledge-solvability annotation service."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from http.server import ThreadingHTTPServer
try:
    from . import server as base
except ImportError:
    import server as base

ROOT = Path(__file__).resolve().parent

class KnowledgeStore(base.Store):
    def __init__(self, records, output, sources, **kwargs):
        # Reuse retrieval only; never load model decisions or old annotations.
        self.records, self.by_key = records, {r['key']: r for r in records}
        self.output, self.lock = output, base.threading.Lock()
        self.retriever = kwargs.get('retriever') or base.retrieve_cases
        self.llm_config = kwargs.get('llm_config') or ROOT.parent / 'config.json'
        self.query_generator = kwargs.get('query_generator')
        fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.state = dict(artifact_type='knowledge_badcase_annotations', schema_version=1,
                          dataset_fingerprint=fingerprint, sources=sources, revision=0, annotations={},
                          settings={'prompt': Path(kwargs.get('prompt_file') or ROOT.parent/'extract_knowledge/eval_prompt_template.txt').read_text(encoding='utf-8'),
                                    'knowledge': Path(kwargs.get('knowledge_file') or ROOT.parent/'extract_knowledge/existing_knowledge.txt').read_text(encoding='utf-8')})
        if output.exists():
            previous = base.load(output)
            if previous.get('artifact_type') != self.state['artifact_type'] or previous.get('dataset_fingerprint') != fingerprint:
                raise ValueError('标注文件类型或数据集不匹配，请使用独立的 --output')
            self.state = previous

    def summary(self):
        values = list(self.state['annotations'].values())
        return {'total': len(self.records), 'completed': sum(a['completed'] for a in values),
                'yes': sum(a['can_solve'] is True for a in values), 'no': sum(a['can_solve'] is False for a in values)}

    def snapshot(self):
        return {'records': self.records, 'state': self.state, 'summary': self.summary(), 'output': str(self.output)}

    def mutate(self, body):
        with self.lock:
            old = copy.deepcopy(self.state)
            try:
                if body.get('action') == 'settings':
                    if body.get('revision') != self.state['revision']:
                        raise RuntimeError('数据已更新，请刷新后重试；当前输入尚未保存')
                    settings = body['settings']
                    if any(not isinstance(settings.get(k), str) for k in ('prompt', 'knowledge')):
                        raise ValueError('提示词和知识必须是文本')
                    self.state['settings'] = {k: settings[k] for k in ('prompt', 'knowledge')}
                elif body.get('action') == 'annotation':
                    key = str(body.get('key'))
                    if key not in self.by_key:
                        raise ValueError('记录不存在')
                    current = self.state['annotations'].get(key, {})
                    if body.get('record_revision') != current.get('revision', 0):
                        raise RuntimeError('该记录已被修改，请刷新后重试；当前输入尚未保存')
                    raw = body['annotation']
                    if type(raw.get('can_solve')) is not bool:
                        raise ValueError('必须选择能否通过知识解决')
                    annotation = {k: raw.get(k, '') for k in ('knowledge_entry', 'error_type', 'reason')}
                    if any(not isinstance(v, str) for v in annotation.values()):
                        raise ValueError('选填项必须是文本')
                    annotation.update(record_key=key, sample_index=self.by_key[key]['sample_index'],
                                      can_solve=raw['can_solve'], completed=True,
                                      revision=current.get('revision', 0)+1, updated_at=base.now())
                    self.state['annotations'][key] = annotation
                else:
                    raise ValueError('未知操作')
                self.state['revision'] += 1
                self.state['updated_at'] = base.now()
                self.state['summary'] = self.summary()
                base.atomic_write(self.output, self.state)
            except Exception:
                self.state = old
                raise
            return {'state': self.state, 'summary': self.summary()}

    def try_multi_query(self, body):
        prompt, knowledge = body.get('prompt'), body.get('knowledge')
        if not isinstance(prompt, str) or not isinstance(knowledge, str) or not prompt.strip():
            raise ValueError('请填写基础提示词和知识文本')
        # Replace only known placeholders: literal JSON braces remain untouched.
        prompt = prompt.replace('{knowledge}', knowledge)
        if '{dialogue}' not in prompt:
            prompt += '\n当前对话：\n{dialogue}'
        prompt += '\n请返回 JSON：{"query":["检索query"]}，可生成1–3条query。'
        return super().try_multi_query({**body, 'prompt': prompt})


def main():
    p = argparse.ArgumentParser(description='知识能否解决 badcase 标注器')
    p.add_argument('--badcases'); p.add_argument('--cases'); p.add_argument('--demo', action='store_true')
    p.add_argument('--output'); p.add_argument('--port', type=int, default=8766)
    p.add_argument('--llm-config', default=str(ROOT.parent/'config.json'))
    p.add_argument('--prompt-file', default=str(ROOT.parent/'extract_knowledge/eval_prompt_template.txt'), help='基础提示词初始文件（UTF-8）')
    p.add_argument('--knowledge-file', default=str(ROOT.parent/'extract_knowledge/existing_knowledge.txt'), help='测试知识初始文件（UTF-8）')
    p.add_argument('--retrieval-url'); p.add_argument('--retrieval-timeout', type=float, default=30)
    args = p.parse_args()
    if not 0 < args.retrieval_timeout <= 300:
        p.error('检索超时需在0–300秒之间')
    if args.demo:
        demo = base.load(ROOT/'demo_data.json'); badcases, cases = demo['badcases'], demo['cases']; sources = {'demo': True}
    else:
        if not args.badcases or not args.cases:
            p.error('需要 --badcases 和 --cases，或 --demo；不需要模型分析文件')
        sources = {k: str(Path(getattr(args, k)).resolve()) for k in ('badcases', 'cases')}
        badcases, cases = base.load(args.badcases), base.load(args.cases)
    output = Path(args.output or ROOT/('knowledge_demo_annotations.json' if args.demo else 'knowledge_annotations.json')).resolve()
    protected = [Path(v).resolve() for k,v in sources.items() if k != 'demo'] + [Path(args.llm_config).resolve(), Path(args.prompt_file).resolve(), Path(args.knowledge_file).resolve()]
    if output in protected:
        p.error('输出不能覆盖输入文件或配置')
    records = base.build_dataset(badcases, {'records': []}, cases)
    for r in records:
        for field in ('model_status', 'model_error', 'stage1_reason'): r.pop(field, None)
        for c in r['retrieved']:
            for field in ('model_same', 'model_reason'): c.pop(field, None)
    store = KnowledgeStore(records, output, sources, llm_config=Path(args.llm_config),
        prompt_file=args.prompt_file, knowledge_file=args.knowledge_file,
        retriever=lambda q: base.retrieve_cases(q, url=args.retrieval_url, timeout=args.retrieval_timeout))
    Handler = base.handler(store)
    class KnowledgeHandler(Handler):
        def do_GET(self):
            name = {'/':'index.html', '/app.js':'app.js'}.get(self.path)
            if name:
                data = (ROOT/'knowledge_static'/name).read_bytes()
                self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8' if name.endswith('html') else 'text/javascript; charset=utf-8')
                self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
            else: super().do_GET()
    server = ThreadingHTTPServer(('0.0.0.0', args.port), KnowledgeHandler)
    print(f'知识标注器：http://0.0.0.0:{args.port}；保存至 {output}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == '__main__':
    main()
