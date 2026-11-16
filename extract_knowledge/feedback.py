"""Retrieval and quality gates for extracted knowledge; IDs stay in local traces."""
from concurrent.futures import ThreadPoolExecutor


def refine(entries, clusters, model, reviewer, settings, existing, save, out, retrieve=None):
    from evaluate import DialogueSample, QueryGenerationRunner, RetrievalEvaluator, SearchRetriever
    from gen_query import MultiQueryGenerator
    from pathlib import Path
    template = Path(settings['prompt_file']).read_text(encoding='utf-8')
    if '{knowledge}' not in template:
        raise ValueError('feedback.prompt_file 必须包含 {knowledge}')
    if '{dialogue}' not in template:
        template += '\n当前对话：\n{dialogue}'
    model.feedback_query_template = template
    existing = Path(settings['knowledge_file']).read_text(encoding='utf-8')
    save(out / 'feedback_evaluation_settings.json', {
        'prompt_template': template, 'existing_knowledge': existing,
        'fusion_method': settings.get('fusion_method', 'round_robin'), 'top_k': settings['top_k']})
    class Retriever:
        def retrieve(self, query):
            return retrieve(query)
    retriever = Retriever() if retrieve is not None else SearchRetriever(settings['retrieval_url'], settings['timeout'])
    evaluator = RetrievalEvaluator(retriever, fusion_method=settings.get('fusion_method', 'round_robin'),
                                   top_k=settings['top_k'], parallel_queries=False)
    lookup = {c['id']: c for c in clusters}
    accepted, rejected, histories = [], [], []
    errors = 0
    for entry in entries:
        cluster = lookup[entry['cluster_id']]
        current = dict(entry)
        history = {'cluster_id': entry['cluster_id'], 'rounds': []}
        for iteration in range(settings['max_rounds']):
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
            with ThreadPoolExecutor(max_workers=settings['concurrency']) as pool:
                tests = list(pool.map(test, cluster['records']))
            failed = review['status'] != 'success' or any('error' in t for t in tests)
            errors += int(failed)
            recall = sum(t['hit'] for t in tests) / len(tests)
            passed = not failed and review['result']['approved'] and recall >= settings['min_recall']
            history['rounds'].append({'knowledge': current['knowledge'], 'review': review, 'tests': tests,
                                      'recall': recall, 'passed': passed})
            save(out / '08_feedback_results.json', histories + [history])
            if passed:
                accepted.append({**current, 'validation': {'passed': True, 'recall': recall, 'rounds': iteration + 1}})
                break
            if failed or iteration + 1 == settings['max_rounds']:
                rejected.append({**current, 'validation': {'passed': False, 'recall': recall}})
                break
            # Whitelist semantic feedback: no record IDs, GT IDs, ranks or source metadata.
            misses = [{'chat_content': row['chat_content'], 'gt_case_title': row['gt_case_title'],
                       'queries': test.get('queries', []),
                       'retrieved_titles': [c['case_title'] for c in test.get('retrieved', [])]}
                      for row, test in zip(cluster['records'], tests) if not test['hit']]
            examples = [{'chat_content': row['chat_content'], 'gt_case_title': row['gt_case_title'],
                         'queries': test.get('queries', [])} for row, test in zip(cluster['records'], tests)]
            revised = model.task('feedback_revise', {'keyword': cluster['keyword'], 'desc': cluster['desc'],
                                 'knowledge': current['knowledge'], 'examples': examples,
                                 'review': review['result']['reason'], 'retrieval_failures': misses})
            history['rounds'][-1]['revision'] = revised
            if revised['status'] != 'success':
                errors += 1
                rejected.append({**current, 'validation': {'passed': False, 'error': 'revision_failed'}})
                break
            current.pop('reason', None)
            current.update({k: revised['result'][k] for k in ('knowledge', 'reason') if k in revised['result']})
        histories.append(history)
        save(out / '08_feedback_results.json', histories)
    save(out / 'feedback_rejected_knowledge.json', rejected)
    return accepted, len(rejected), errors
