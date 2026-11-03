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
    def test_end_to_end_dedup_filter_failure_and_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, keyword in enumerate(['Alpha', 'Alpha', 'Beta', 'Existing', 'Broken']):
                rows.append({'expected_case_id': str(i), 'chat_content': f'用户需求{i}',
                             'gt_case_title': keyword + ' 登录', 'query': '登录不了'})
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
                        result = {'keywords': [{'keyword': data['records'][0]['gt_case_title'].split()[0], 'reason': 'query遗漏'}]}
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
                                      concurrency=4, max_examples=10, min_cluster_size=0, precluster_edit_threshold=0.8,
                                      precluster_dice_threshold=0.8, precluster_short_threshold=0.9)
            with patch.object(p, 'Model', FakeModel):
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
            with patch.object(p, 'Model', FakeModel):
                self.assertEqual(p.run(args, {}), 1)  # malformed input still yields partial
            self.assertEqual(p.read_json(root / 'out/knowledge.json'), [])
            self.assertEqual(p.read_json(root / 'out/05_filter_results.json'), [])
            self.assertEqual(p.read_json(root / 'out/07_extraction_results.json'), [])
            discarded = p.read_json(root / 'out/low_frequency_clusters.json')
            self.assertEqual(sorted(c['count'] for c in discarded), [1, 1, 1, 2])
            # A larger cluster passes, while the singleton clusters never reach filtering.
            args.min_cluster_size = 1
            with patch.object(p, 'Model', FakeModel):
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
               'chat_content': '用户：外网无法访问', 'gt_case_title': 'WebG 访问问题',
               'queries': ['外网无法访问'], 'future_metadata': 'secret-future'}
        expected = {key: row[key] for key in ('chat_content', 'gt_case_title', 'queries')}
        tasks = {
            'keywords': ({'case_id': 'secret-case', 'records': [row]}, {'records': [expected]}),
            'merge': ({'keywords': ['华为wifi', '华为wi-fi'], 'source': '/secret/path'},
                      {'keywords': ['华为wifi', '华为wi-fi']}),
            'filter': ({'keyword': 'WebG', 'existing_knowledge': '已有条目', 'source': '/secret/path'},
                       {'keyword': 'WebG', 'existing_knowledge': '已有条目'}),
            'extract': ({'cluster_id': 'secret-cluster', 'keyword': 'WebG', 'examples': [row]},
                        {'keyword': 'WebG', 'examples': [expected]})}
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

    def test_size_filter_runs_before_candidate_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'bad.json'
            rows = [{'expected_case_id': str(i), 'chat_content': f'用户{i}',
                     'gt_case_title': word, 'query': '无法访问'}
                    for i, word in enumerate(['华为wifi', '华为wifi', '华为wi-fi'])]
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
                        result = {'keywords': [{'keyword': data['records'][0]['gt_case_title'], 'reason': '遗漏'}]}
                    elif stage == 'filter':
                        result = {'covered': False, 'reason': '无已有知识'}
                    elif stage == 'extract':
                        result = {'knowledge': '知识条目', 'reason': '依据'}
                    else:
                        raise AssertionError('singleton should not reach merge')
                    return {'status': 'success', 'result': result}
            args = argparse.Namespace(output_dir=str(root / 'out'), knowledge=str(knowledge),
                                      cases=None, dialogues=None, badcases=[str(source)],
                                      keyword_batch_size=10, concurrency=2, max_examples=10, min_cluster_size=1, precluster_edit_threshold=0.8,
                                      precluster_dice_threshold=0.8, precluster_short_threshold=0.9)
            with patch.object(p, 'Model', FakeModel), patch.object(p, 'candidates', wraps=p.candidates) as candidates:
                self.assertEqual(p.run(args, {}), 0)
                self.assertEqual([c['keyword'] for c in candidates.call_args.args[0]], ['华为wifi'])
            self.assertNotIn('merge', stages)
            self.assertEqual(p.read_json(root / 'out/low_frequency_clusters.json')[0]['keyword'], '华为wi-fi')
            self.assertEqual(p.read_json(root / 'out/knowledge.json')[0]['cluster_count'], 2)

    def test_extract_new_schema(self):
        result = {'knowledge': 'WebG：外网网关；访问外网时考虑 WebG', 'reason': '对话依据'}
        self.assertEqual(p.validate('extract', result), result)
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
            p.save(config, {'llm': {'base_url': 'http://fake/v1', 'model_name': 'fake', 'api_key': 'test'},
                            'extract_knowledge': {'badcases': [str(root / 'bad*.json')],
                                                  'output_dir': 'extract_knowledge/outputs/test',
                                                  'concurrency': 7, 'resume': True, 'precluster_edit_threshold': 0.85}})
            with patch.object(p, 'run', return_value=0) as run:
                self.assertEqual(p.main(['--config', str(config)]), 0)
                args, llm = run.call_args.args
                self.assertEqual(args.concurrency, 7)
                self.assertEqual(args.precluster_edit_threshold, 0.85)
                self.assertEqual(args.precluster_dice_threshold, 0.8)
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
