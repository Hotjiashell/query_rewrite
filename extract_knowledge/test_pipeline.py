"""Offline regression tests: python3 -m unittest extract_knowledge.test_pipeline."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extract_knowledge import pipeline as p


class PipelineTests(unittest.TestCase):
    def test_cluster_description_uses_largest_case_and_random_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, knowledge = root / 'bad.json', root / 'existing.txt'
            knowledge.write_text('', encoding='utf-8')
            args = argparse.Namespace(output_dir=str(root / 'out'), knowledge=str(knowledge), cases=None,
                                      dialogues=None, badcases=[str(source)], keyword_batch_size=1,
                                      concurrency=2, max_examples=10, min_cluster_size=0,
                                      precluster_similarity_threshold=0.85)
            class FakeModel:
                identity = {}
                templates = {}
                def __init__(self, *args):
                    pass
                def task(self, stage, data):
                    if stage == 'keywords':
                        return {'status': 'success', 'result': {'keywords': [
                            {'keyword': '内部词', 'desc': '描述' + data['case_id'], 'reason': '依据'}]}}
                    return {'status': 'failed'}
            for counts, allowed in [([1, 3, 2], {'描述B'}), ([1, 3, 3], {'描述B', '描述C'})]:
                rows = [{'expected_case_id': case_id, 'chat_content': f'对话{i}',
                         'gt_case_title': '内部词', 'gt_case_content': '正文'}
                        for case_id, count in zip('ABC', counts) for i in range(count)]
                p.save(source, {'artifact_type': 'badcase_report', 'records': rows})
                with patch.object(p, 'Model', FakeModel), patch.object(p, 'precluster', return_value=([], [], 0)):
                    p.run(args, {})
                    cluster = p.read_json(root / 'out/04_keyword_clusters.json')[0]
                    self.assertIn(cluster['desc'], allowed)
                    self.assertEqual(cluster['concept'], '内部词：' + cluster['desc'])
                    self.assertEqual(cluster['description_source']['dialogue_count'], 3)
                    self.assertEqual(cluster['count'], sum(counts))
                    self.assertEqual([e['dialogue_count'] for e in cluster['keyword_evidence']], counts)
                    p.run(args, {})
                    self.assertEqual(p.read_json(root / 'out/04_keyword_clusters.json')[0]['desc'], cluster['desc'])

    def test_one_keyword_task_per_case_with_reproducible_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'expected_case_id': case_id, 'chat_content': f'对话{i}',
                     'gt_case_title': '内部产品', 'gt_case_content': '产品正文'}
                    for case_id, count in [('large', 25), ('small', 2)] for i in range(count)]
            source = root / 'bad.json'
            p.save(source, {'artifact_type': 'badcase_report', 'records': rows})
            knowledge = root / 'existing.txt'
            knowledge.write_text('', encoding='utf-8')
            args = argparse.Namespace(output_dir=str(root / 'out'), knowledge=str(knowledge), cases=None,
                                      dialogues=None, badcases=[str(source)], keyword_batch_size=10,
                                      concurrency=2, max_examples=10, min_cluster_size=1,
                                      precluster_similarity_threshold=0.85)
            class FakeModel:
                identity = {'model_name': 'fake'}
                templates = {}
                def __init__(self, *args):
                    pass
                def task(self, stage, data):
                    self_test.assertEqual(stage, 'keywords')
                    return {'status': 'failed'}
            self_test = self
            with patch.object(p, 'Model', FakeModel), patch.object(p, 'precluster', return_value=([], [], 0)):
                p.run(args, {})
                tasks = p.read_json(root / 'out/02_keyword_tasks.json')
                self.assertEqual([(t['case_id'], len(t['records'])) for t in tasks], [('large', 10), ('small', 2)])
                self.assertEqual(len({r['id'] for r in tasks[0]['records']}), 10)
                self.assertEqual(len(p.read_json(root / 'out/03_keyword_results.json')), 2)
                self.assertEqual(len(p.read_json(root / 'out/01_records.json')), 27)
                p.run(args, {})
                self.assertEqual(p.read_json(root / 'out/02_keyword_tasks.json'), tasks)

    def test_end_to_end_dedup_filter_failure_and_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, keyword in enumerate(['Alpha', 'Alpha', 'Beta', 'Existing', 'Broken']):
                rows.append({'expected_case_id': str(i), 'chat_content': f'用户需求{i}',
                             'gt_case_title': keyword + ' 登录', 'gt_case_content': '内部产品使用说明', 'query': '登录不了'})
            source = root / 'bad.json'
            p.save(source, {'artifact_type': 'badcase_report', 'records': rows + [rows[0], None]})
            knowledge = root / 'existing.txt'
            knowledge.write_text('Existing：当用户登录时可考虑', encoding='utf-8')
            class FakeModel:
                identity = {'model_name': 'fake'}
                templates = {}
                def __init__(self, *args):
                    pass
                def task(self, stage, data):
                    if stage == 'keywords':
                        result = {'keywords': [{'keyword': data['records'][0]['gt_case_title'].split()[0], 'desc': '企业登录产品', 'reason': 'query遗漏'}]}
                    elif stage == 'filter':
                        if data['keyword'] == 'Broken':
                            return {'status': 'failed'}
                        result = {'covered': data['keyword'] == 'Existing', 'reason': '比较', 'matched_knowledge': '已有'}
                    else:
                        assert len(data['examples']) <= 10
                        result = p.validate('extract', {'knowledge': data['keyword'] + '：登录时可考虑', 'reason': '依据'})
                    return {'status': 'success', 'result': result}
            args = argparse.Namespace(output_dir=str(root / 'out'), knowledge=str(knowledge), cases=None,
                                      dialogues=None, badcases=[str(source)], keyword_batch_size=1,
                                      concurrency=4, max_examples=10, min_cluster_size=0, precluster_similarity_threshold=0.85)
            with patch.object(p, 'Model', FakeModel), patch.object(p, 'precluster', return_value=([], [], 0)):
                self.assertEqual(p.run(args, {}), 1)
            entries = p.read_json(root / 'out/knowledge.json')
            self.assertEqual([r['keyword'] for r in entries], ['Alpha', 'Beta'])
            self.assertEqual([r['cluster_count'] for r in entries], [2, 1])
            self.assertEqual((root / 'out/knowledge.txt').read_text(),
                             '1. Alpha：登录时可考虑\n2. Beta：登录时可考虑\n')
            self.assertEqual(p.read_json(root / 'out/existing_knowledge.json')['content'], knowledge.read_text())
            self.assertEqual(len(p.read_json(root / 'out/01_records.json')), 5)
            self.assertTrue(p.read_json(root / 'out/summary.json')['partial'])
            # At the inclusive boundary count=2 is dropped, as are singleton clusters.
            args.min_cluster_size = 2
            with patch.object(p, 'Model', FakeModel), patch.object(p, 'precluster', return_value=([], [], 0)):
                self.assertEqual(p.run(args, {}), 1)  # malformed input still yields partial
            self.assertEqual(p.read_json(root / 'out/knowledge.json'), [])
            self.assertEqual(p.read_json(root / 'out/05_filter_results.json'), [])
            self.assertEqual(p.read_json(root / 'out/07_extraction_results.json'), [])
            discarded = p.read_json(root / 'out/low_frequency_clusters.json')
            self.assertEqual(sorted(c['count'] for c in discarded), [1, 1, 1, 2])
            # A larger cluster passes, while the singleton clusters never reach filtering.
            args.min_cluster_size = 1
            with patch.object(p, 'Model', FakeModel), patch.object(p, 'precluster', return_value=([], [], 0)):
                self.assertEqual(p.run(args, {}), 1)
            self.assertEqual([c['keyword'] for c in p.read_json(root / 'out/knowledge.json')], ['Alpha'])
            self.assertEqual(len(p.read_json(root / 'out/05_filter_results.json')), 1)

    def test_retry_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output_dir=directory, max_tokens=512, resume=True, retries=1)
            model = object.__new__(p.Model)
            model.args, model.model, model.identity = args, 'fake', {'model_name': 'fake', 'temperature': 0}
            responses = iter(['invalid', '{"keywords":[]}', '{"keywords":[]}'])
            def create(**kwargs):
                self.assertFalse(kwargs['extra_body']['chat_template_kwargs']['enable_thinking'])
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=next(responses)))])
            model.templates = p.load_prompts(p.HERE / 'prompts.md')
            model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with patch.object(p.time, 'sleep'):
                result = model.task('keywords', {'records': []})
            self.assertEqual(result['status'], 'success')
            self.assertEqual(len(result['attempts']), 2)
            self.assertTrue(model.task('keywords', {'records': []})['cached'])
            self.assertEqual(len(list(Path(directory).rglob('*.json'))), 1)
            model.templates['keywords'] += '\n新增人工调整'
            self.assertNotIn('cached', model.task('keywords', {'records': []}))
            self.assertEqual(len(list(Path(directory).rglob('*.json'))), 2)

    def test_model_requests_exclude_metadata(self):
        row = {'id': 'secret-id', 'expected_case_id': 'secret-case',
               'call_sno': 'secret-call', 'sources': [{'file': '/secret/path', 'record_index': 9}],
               'chat_content': '用户：外网无法访问', 'gt_case_title': 'WebG 访问问题', 'gt_case_content': '访问外网的安全网关',
               'queries': ['外网无法访问'], 'future_metadata': 'secret-future'}
        expected = {key: row[key] for key in ('chat_content', 'gt_case_title', 'queries')}
        tasks = {
            'keywords': ({'case_id': 'secret-case', 'records': [row, row]},
                         {'gt_case_title': row['gt_case_title'], 'gt_case_content': row['gt_case_content'],
                          'records': [{'chat_content': row['chat_content'], 'queries': row['queries']}] * 2}),
            'merge': ({'keywords': ['华为wifi', '华为wi-fi'], 'source': '/secret/path'},
                      {'keywords': ['华为wifi', '华为wi-fi']}),
            'filter': ({'keyword': 'WebG', 'desc': '外网网关', 'existing_knowledge': '已有条目', 'source': '/secret/path'},
                       {'keyword': 'WebG：外网网关', 'desc': '外网网关', 'existing_knowledge': '已有条目'}),
            'extract': ({'cluster_id': 'secret-cluster', 'keyword': 'WebG', 'desc': '外网网关', 'examples': [row]},
                        {'keyword': 'WebG', 'desc': '外网网关', 'examples': [expected]})}
        outputs = {'keywords': {'keywords': []},
                   'merge': {'groups': [{'keyword': '华为wifi', 'keywords': ['华为wifi', '华为wi-fi'], 'reason': '同义'}]},
                   'filter': {'covered': False, 'reason': '未覆盖'},
                   'extract': {'knowledge': 'WebG：外网访问时考虑', 'reason': '依据'}}
        with tempfile.TemporaryDirectory() as directory:
            model = object.__new__(p.Model)
            model.args = argparse.Namespace(output_dir=directory, max_tokens=512, resume=True, retries=0)
            model.model, model.identity = 'fake', {'model_name': 'fake', 'temperature': 0}
            model.templates = {stage: '{input}' for stage in tasks}
            for stage, (task, payload) in tasks.items():
                def create(**kwargs):
                    actual = json.loads(kwargs['messages'][0]['content'])
                    self.assertEqual(actual, payload)
                    self.assertNotIn('secret-', kwargs['messages'][0]['content'])
                    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                        message=SimpleNamespace(content=json.dumps(outputs[stage])))])
                model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                result = model.task(stage, task)
                self.assertEqual(result['status'], 'success')
                self.assertEqual(result['input'], payload)
                # Metadata changes do not invalidate semantic model-call caches.
                task['irrelevant_metadata'] = 'changed'
                self.assertTrue(model.task(stage, task)['cached'])
            self.assertEqual(row['sources'][0]['file'], '/secret/path')

    def test_candidate_fusion_dedup_and_failure(self):
        from extract_knowledge.clustering import candidates, fuse, validate_partition, similarity
        words = ['华为wifi', '华为wi-fi', '华为VPN']
        shared = {'id': 'shared', 'expected_case_id': 'A'}
        clusters = [{'keyword': word, 'records': [shared, {'id': str(i), 'expected_case_id': str(i)}],
                     'keyword_evidence': [], 'count': 2} for i, word in enumerate(words)]
        groups, edges = candidates(clusters)
        self.assertEqual(groups, [words[:2]])
        self.assertTrue(edges)
        self.assertTrue(similarity('华为wifi', '华为 wi‑fi')['candidate'])
        self.assertFalse(similarity('AB', 'AC')['candidate'])
        self.assertTrue(similarity('abcde', 'abcdf', 0.8, 1)['candidate'])
        self.assertFalse(similarity('abcde', 'abcdf', 0.9, 1)['candidate'])
        self.assertTrue(similarity('abcde', 'abcdf', 1, 0.75)['candidate'])
        self.assertTrue(similarity('AB', 'AC', 0.5, 1, 0.5)['candidate'])
        task = {'keywords': words[:2]}
        result = {'groups': [{'keyword': words[0], 'keywords': words[:2], 'reason': '连接符差异'}]}
        merged = fuse(clusters, [task], [{'status': 'success', 'result': result}], p.digest)
        self.assertEqual(merged[0]['count'], 3)
        self.assertEqual(merged[0]['aliases'], words[:2])
        self.assertEqual(merged[0]['gt_case_count'], 3)
        self.assertEqual(len(fuse(clusters, [task], [{'status': 'failed'}], p.digest)), 3)
        for bad in ({'groups': []}, {'groups': [{'keyword': words[0], 'keywords': words, 'reason': '新增'}]},
                    {'groups': [{'keyword': words[0], 'keywords': [words[0], words[0]], 'reason': '重复'}]}):
            with self.assertRaises(ValueError):
                validate_partition(bad, words[:2])
        # Chain-like candidates can be split by the model instead of unioned blindly.
        split = {'groups': [{'keyword': w, 'keywords': [w], 'reason': '不同'} for w in words[:2]]}
        self.assertEqual(len(fuse(clusters, [task], [{'status': 'success', 'result': split}], p.digest)), 3)

    def test_merge_analysis_and_json_block(self):
        words = ['华为wifi', '华为wi-fi']
        answer = {'groups': [{'keyword': words[0], 'keywords': words, 'reason': '连接符差异'}]}
        raw = '分析：检查每个词，两个名称核心概念一致。\n```json\n' + json.dumps(answer) + '\n```'
        self.assertEqual(p.parse_result('merge', raw, {'keywords': words}), answer)
        self.assertEqual(p.parse_result('merge', json.dumps(answer), {'keywords': words}), answer)
        with self.assertRaises(ValueError):
            p.parse_result('merge', '分析\n```json\n{"groups": []}\n```', {'keywords': words})

    def test_size_filter_runs_after_model_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'bad.json'
            rows = [{'expected_case_id': str(i), 'chat_content': f'用户{i}',
                     'gt_case_title': word, 'gt_case_content': '企业无线网络服务', 'query': '无法访问'}
                    for i, word in enumerate(['华为wifi', '华为wi-fi', '其他词'])]
            p.save(source, {'artifact_type': 'badcase_report', 'records': rows})
            knowledge = root / 'existing.txt'
            knowledge.write_text('')
            stages = []
            class FakeModel:
                identity, templates = {}, {}
                def __init__(self, *args):
                    pass
                def task(self, stage, data):
                    stages.append(stage)
                    if stage == 'keywords':
                        result = {'keywords': [{'keyword': data['records'][0]['gt_case_title'], 'desc': '企业无线网络', 'reason': '遗漏'}]}
                    elif stage == 'filter':
                        result = {'covered': False, 'reason': '无已有知识'}
                    elif stage == 'extract':
                        result = {'knowledge': '知识条目', 'reason': '依据'}
                    else:
                        result = {'groups': [{'keyword': data['keywords'][0], 'keywords': data['keywords'], 'reason': '同义'}]}
                    return {'status': 'success', 'result': result}
            args = argparse.Namespace(output_dir=str(root / 'out'), knowledge=str(knowledge),
                                      cases=None, dialogues=None, badcases=[str(source)],
                                      keyword_batch_size=10, concurrency=2, max_examples=10, min_cluster_size=1, precluster_similarity_threshold=0.85)
            with patch.object(p, 'Model', FakeModel), patch.object(p, 'precluster', return_value=(
                    [['华为wifi：企业无线网络', '华为wi-fi：企业无线网络']], [], 0)) as candidates:
                self.assertEqual(p.run(args, {}), 0)
                self.assertEqual(len(candidates.call_args.args[0]), 3)
            self.assertIn('merge', stages)
            self.assertEqual(p.read_json(root / 'out/low_frequency_clusters.json')[0]['keyword'], '其他词')
            self.assertEqual(p.read_json(root / 'out/knowledge.json')[0]['cluster_count'], 2)

    def test_single_keyword_analysis_and_semantic_candidates(self):
        from .embedding import semantic_groups, vectors_valid
        result = p.parse_result('keywords', '分析文字\n```json\n{"keywords":"内部产品", "desc":"企业内部产品描述", "reason":"依据"}\n```', {})
        self.assertEqual(result['keywords'][0]['keyword'], '内部产品')
        self.assertEqual(p.validate('keywords', {'keywords':'', 'reason':'无符合项'}), {'keywords':[]})
        clusters = [{'keyword': w} for w in ['华为wifi','无线网络','邮箱']]
        groups, edges = semantic_groups(clusters, [[1,0],[0.99,0.01],[0,1]], 0.9)
        self.assertEqual(groups, [['华为wifi','无线网络']])
        self.assertEqual(len(edges), 1)
        self.assertEqual(semantic_groups(clusters, [[1,0],[0.99,0.01],[0,1]], 1)[0], [])
        for vectors in ([[0,0]], [[float('nan'),1]], [[1,0],[1,0,0]]):
            with self.assertRaises(ValueError):
                vectors_valid(vectors,len(vectors))

    def test_embedding_batches_retry_cache_and_failure(self):
        from .embedding import precluster
        import sys
        calls = []
        attempts = {}
        def create(**kwargs):
            words = kwargs['input']
            calls.append(words)
            attempts[words[0]] = attempts.get(words[0], 0) + 1
            if words[0] == 'A' and attempts['A'] == 1:
                raise RuntimeError('temporary')
            data = [SimpleNamespace(index=i,embedding=[1,0] if w != 'C' else [0,1]) for i,w in enumerate(words)]
            return SimpleNamespace(data=list(reversed(data)))
        fake = SimpleNamespace(OpenAI=lambda **kwargs:SimpleNamespace(embeddings=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output_dir=Path(directory),resume=True,precluster_similarity_threshold=0.9,
                embedding={'base_url':'http://fake','model_name':'embed','api_key':'secret',
                           'timeout':10,'retries':1,'batch_size':2,'concurrency':2})
            clusters = [{'keyword':w} for w in ['A','B','C']]
            with patch.dict(sys.modules, {'openai':fake}), patch('extract_knowledge.embedding.time.sleep'):
                groups,edges,failed = precluster(clusters,args,p.save,p.read_json,p.digest)
                self.assertEqual(groups,[['A','B']])
                self.assertEqual(failed,0)
                first = len(calls)
                precluster(clusters,args,p.save,p.read_json,p.digest)
                self.assertEqual(len(calls),first)
            self.assertNotIn('secret',(Path(directory)/'04_embeddings.json').read_text())
            args.resume=False
            broken=SimpleNamespace(OpenAI=lambda **kwargs:SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError('offline')))))
            with patch.dict(sys.modules,{'openai':broken}), patch('extract_knowledge.embedding.time.sleep'):
                groups,edges,failed=precluster(clusters,args,p.save,p.read_json,p.digest)
                self.assertEqual(groups,[])
                self.assertEqual(failed,2)

    def test_case_content_and_concept_fusion(self):
        from .clustering import fuse
        from .embedding import semantic_groups
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'bad.json'
            p.save(source, {'artifact_type':'badcase_report','records':[
                {'expected_case_id':'GT','chat_content':'用户对话','query':'query'}]})
            rows,issues = p.load_records([source], {'GT':{'case_name':'内部词标题','text':'内部系统正文'}})
            self.assertEqual(issues,[])
            self.assertEqual(p.model_input('keywords',{'records':rows})['gt_case_content'],'内部系统正文')
            self.assertEqual(len(p.load_records([source], {'GT':{'case_name':'标题'}})[1]),1)
        with self.assertRaises(ValueError):
            p.validate('keywords',{'keywords':'内部词','reason':'依据'})
        clusters=[{'keyword':w,'desc':d,'concept':w+'：'+d,'records':[{'id':str(i),'expected_case_id':str(i)}],
                   'keyword_evidence':[]} for i,(w,d) in enumerate([('内部A','外网网关'),('内部B','网络出口')])]
        groups,_=semantic_groups(clusters,[[1,0],[1,0]],0.9)
        self.assertEqual(groups,[['内部A：外网网关','内部B：网络出口']])
        task={'keywords':groups[0]}
        answer={'groups':[{'keyword':groups[0][0],'keywords':groups[0],'reason':'同义'}]}
        result=fuse(clusters,[task],[{'status':'success','result':answer}],p.digest)[0]
        self.assertEqual(result['keyword'],'内部A')
        self.assertEqual(result['desc'],'外网网关')
        self.assertEqual(result['concepts'],groups[0])
        self.assertEqual(result['count'],2)

    def test_extract_new_schema(self):
        result = {'knowledge': 'WebG：外网网关；访问外网时考虑 WebG', 'reason': '对话依据'}
        self.assertEqual(p.validate('extract', result), result)
        self.assertEqual(p.parse_result('extract', '分析说明\n```json\n{"knowledge":"完整知识"}\n```', {}), {'knowledge': '完整知识'})
        for invalid in ({'reason': '依据'}, {'knowledge': '', 'reason': '依据'},
                        {'knowledge': '条目', 'reason': ''}, {'knowledge': [], 'reason': '依据'}):
            with self.assertRaises(ValueError):
                p.validate('extract', invalid)

    def test_prompt_file_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.md'
            path.write_text('## keywords\n{input}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '缺少阶段'):
                p.load_prompts(path)
            path.write_text('## keywords\n{input}\n## filter\n无占位符\n## extract\n{input}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '占位符'):
                p.load_prompts(path)

    def test_config_only_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p.save(root / 'bad1.json', {})
            config = root / 'config.json'
            p.save(config, {'embedding': {'base_url': 'http://fake/v1', 'model_name': 'embed'}, 'llm': {'base_url': 'http://fake/v1', 'model_name': 'fake', 'api_key': 'test'},
                            'extract_knowledge': {'badcases': [str(root / 'bad*.json')],
                                                  'output_dir': 'extract_knowledge/outputs/test',
                                                  'concurrency': 7, 'resume': True, 'precluster_similarity_threshold': 0.85}})
            with patch.object(p, 'run', return_value=0) as run:
                self.assertEqual(p.main(['--config', str(config)]), 0)
                args, llm = run.call_args.args
                self.assertEqual(args.concurrency, 7)
                self.assertEqual(args.precluster_similarity_threshold, 0.85)
                self.assertEqual(args.min_cluster_size, 1)
                self.assertEqual(args.knowledge, str(p.HERE / 'existing_knowledge.txt'))
                self.assertTrue(args.resume)
                self.assertEqual(args.badcases, [str(root / 'bad1.json')])
                self.assertEqual(args.output_dir, str(p.ROOT / 'extract_knowledge/outputs/test'))
                self.assertEqual(llm['model_name'], 'fake')
                self.assertEqual(p.main(['--config', str(config), '--concurrency', '3', '--no-resume']), 0)
                args = run.call_args.args[0]
                self.assertEqual(args.concurrency, 3)
                self.assertFalse(args.resume)
            with patch.object(p, 'HERE', root), patch.object(p, 'run', return_value=0) as run:
                self.assertEqual(p.main([]), 0)
                self.assertTrue(run.called)

    def test_diverse_sampling_and_validation(self):
        rows = [{'expected_case_id': 'A', 'n': i} for i in range(15)] + [{'expected_case_id': 'B', 'n': 99}]
        chosen = p.diverse(rows, 10)
        self.assertEqual(len(chosen), 10)
        self.assertEqual(chosen[1]['expected_case_id'], 'B')
        with self.assertRaises(ValueError):
            p.validate('filter', {'covered': 'false', 'reason': 'bad boolean'})
        self.assertEqual(p.canonical('ＷｅｂＧ '), p.canonical('WebG'))


if __name__ == '__main__':
    unittest.main()
