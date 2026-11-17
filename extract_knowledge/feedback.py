"""Retrieval and quality gates for extracted knowledge; IDs stay in local traces."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


def refine(entries, clusters, model, reviewer, settings, existing, save, out, retrieve=None):
    from evaluate import DialogueSample, QueryGenerationRunner, RetrievalEvaluator, SearchRetriever
    from gen_query import MultiQueryGenerator
    if __package__:
        from .pipeline import diverse
    else:
        from pipeline import diverse
    from pathlib import Path
    limit = settings.get('max_entries', 0)
    selected = entries[:limit] if limit else entries
    skipped = [{**entry, 'validation': {'passed': None, 'skipped': True, 'reason': 'outside_feedback_limit'}}
               for entry in entries[len(selected):]]
    save(out / 'feedback_skipped_knowledge.json', skipped)
    template = Path(settings['prompt_file']).read_text(encoding='utf-8')
    if '{knowledge}' not in template:
        raise ValueError('feedback.prompt_file 必须包含 {knowledge}')
    if '{dialogue}' not in template:
        template += '\n当前对话：\n{dialogue}'
    model.feedback_query_template = template
    existing = Path(settings['knowledge_file']).read_text(encoding='utf-8')
    save(out / 'feedback_evaluation_settings.json', {
        'prompt_template': template, 'existing_knowledge': existing,
        'fusion_method': settings.get('fusion_method', 'round_robin'), 'top_k': settings['top_k'],
        'max_entries': limit, 'selected_count': len(selected), 'skipped_count': len(skipped)})
    class Retriever:
        def retrieve(self, query):
            return retrieve(query)
    retriever = Retriever() if retrieve is not None else SearchRetriever(settings['retrieval_url'], settings['timeout'])
    evaluator = RetrievalEvaluator(retriever, fusion_method=settings.get('fusion_method', 'round_robin'),
                                   top_k=settings['top_k'], parallel_queries=False)
    lookup = {c['id']: c for c in clusters}
    accepted, rejected, histories = [], [], []
    errors = 0
    print(f'[feedback] 0/{len(selected)} 待验证知识={len(selected)} 跳过={len(skipped)} '
          f'最多轮数={settings["max_rounds"]}',
          file=sys.stderr, flush=True)
    for entry_index, entry in enumerate(selected, 1):
        cluster = lookup[entry['cluster_id']]
        prefix = f'[feedback {entry_index}/{len(selected)} {cluster["keyword"]!r}]'
        current = dict(entry)
        history = {'cluster_id': entry['cluster_id'], 'rounds': []}
        for iteration in range(settings['max_rounds']):
            print(f'{prefix} 第 {iteration + 1}/{settings["max_rounds"]} 轮 评审中',
                  file=sys.stderr, flush=True)
            review_examples = [{k: r[k] for k in ('chat_content', 'gt_case_title')} for r in cluster['records']]
            review = reviewer.task('feedback_review', {'knowledge': current['knowledge'], 'examples': review_examples})
            class Generator(MultiQueryGenerator):
                def generate_queries(self, dialogue):
                    response = model.task('feedback_query', {'knowledge': '\n'.join(part for part in (existing, current['knowledge']) if part.strip()),
                                                             'chat_content': dialogue})
                    if response['status'] != 'success':
                        raise ValueError('query_generation_failed')
                    return response['result']['queries']
            runner = QueryGenerationRunner(Generator())
            def test(row):
                generated = runner.generate_sample(DialogueSample(
                    index=0, call_sno=None, dialogue=row['chat_content'], expected_case_id=row['expected_case_id']))
                evaluation = evaluator.evaluate_query(generated)
                rank = evaluation['matched_rank']
                result = {'record_id': row['id'], 'expected_case_id': row['expected_case_id'],
                          'queries': generated.queries or [],
                          'hit': rank is not None and rank <= settings['top_k'],
                          'retrieved': evaluation['retrieval_trace'], 'evaluation': evaluation}
                if evaluation['retrieval_status'] != 'success':
                    result['error'] = 'query_or_retrieval_failed'
                return result
            tests = []
            retrieval_status = 'skipped'
            if review['status'] == 'success' and review['result']['approved']:
                total = len(cluster['records'])
                print(f'{prefix} 评审通过，开始生成 query 并检索 0/{total}', file=sys.stderr, flush=True)
                tests = [None] * total
                completed = hits = test_errors = 0
                with ThreadPoolExecutor(max_workers=settings['concurrency']) as pool:
                    futures = {pool.submit(test, row): i for i, row in enumerate(cluster['records'])}
                    for future in as_completed(futures):
                        result = future.result()
                        tests[futures[future]] = result
                        completed += 1
                        hits += result['hit']
                        test_errors += 'error' in result
                        if completed % max(1, total // 10) == 0 or completed == total:
                            print(f'{prefix} 检索 {completed}/{total} 命中={hits} 错误={test_errors}',
                                  file=sys.stderr, flush=True)
                retrieval_status = 'completed'
            elif review['status'] != 'success':
                print(f'{prefix} 评审调用失败，停止本条验证', file=sys.stderr, flush=True)
            else:
                print(f'{prefix} 评审拒绝，跳过检索', file=sys.stderr, flush=True)
            failed = review['status'] != 'success' or any('error' in t for t in tests)
            errors += int(failed)
            recall = sum(t['hit'] for t in tests) / len(tests) if tests else None
            passed = not failed and review['result']['approved'] and recall >= settings['min_recall']
            if recall is not None:
                print(f'{prefix} 命中比例={recall:.2%} 要求>={settings["min_recall"]:.2%} '
                      f'本轮通过={passed}', file=sys.stderr, flush=True)
            history['rounds'].append({'knowledge': current['knowledge'], 'review': review, 'tests': tests,
                                      'recall': recall, 'retrieval_status': retrieval_status, 'passed': passed})
            save(out / '08_feedback_results.json', histories + [history])
            if passed:
                accepted.append({**current, 'validation': {'passed': True, 'recall': recall, 'rounds': iteration + 1}})
                break
            if failed or iteration + 1 == settings['max_rounds']:
                reason = '调用失败' if failed else '达到验证轮数上限'
                print(f'{prefix} {reason}，本条未通过', file=sys.stderr, flush=True)
                rejected.append({**current, 'validation': {'passed': False, 'recall': recall}})
                break
            # Share a ten-dialogue budget across examples and failure feedback.
            by_id = {row['id']: test for row, test in zip(cluster['records'], tests)}
            failed_rows = [row for row in cluster['records']
                           if row['id'] in by_id and not by_id[row['id']]['hit']]
            selected_failures = diverse(failed_rows, 10) if failed_rows else []
            selected_ids = {row['id'] for row in selected_failures}
            remaining = [row for row in cluster['records'] if row['id'] not in selected_ids]
            budget = 10 - len(selected_failures)
            selected_examples = diverse(remaining, budget) if budget else []
            misses = [{'chat_content': row['chat_content'], 'gt_case_title': row['gt_case_title'],
                       'queries': by_id[row['id']].get('queries', []),
                       'retrieved_titles': [c['case_title'] for c in by_id[row['id']].get('retrieved', [])]}
                      for row in selected_failures]
            examples = [{'chat_content': row['chat_content'], 'gt_case_title': row['gt_case_title'],
                         'queries': by_id[row['id']].get('queries', []) if tests else row.get('queries', [])}
                        for row in selected_examples]
            revision_input = {**{k: cluster[k] for k in ('relations', 'relation_members') if k in cluster}, 'keyword': cluster['keyword'], 'desc': cluster['desc'],
                              'knowledge': current['knowledge'], 'existing_knowledge': existing,
                              'examples': examples, 'retrieval_failures': misses}
            if not review['result']['approved']:
                revision_input['review'] = review['result']['reason']
            print(f'{prefix} 修改知识中', file=sys.stderr, flush=True)
            revised = model.task('feedback_revise', revision_input)
            history['rounds'][-1]['revision'] = revised
            if revised['status'] != 'success':
                print(f'{prefix} 修改调用失败，本条未通过', file=sys.stderr, flush=True)
                errors += 1
                rejected.append({**current, 'validation': {'passed': False, 'error': 'revision_failed'}})
                break
            current.pop('reason', None)
            current.update({k: revised['result'][k] for k in ('knowledge', 'reason') if k in revised['result']})
        histories.append(history)
        save(out / '08_feedback_results.json', histories)
        print(f'[feedback] {entry_index}/{len(selected)} 通过={len(accepted)} 未通过={len(rejected)} '
              f'调用错误={errors}', file=sys.stderr, flush=True)
    save(out / 'feedback_rejected_knowledge.json', rejected)
    return accepted + skipped, len(rejected), errors
