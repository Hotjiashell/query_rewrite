"""Budgeted keyword research and non-transitive one-hop relation expansion."""
from concurrent.futures import ThreadPoolExecutor
import re


def investigate(keyword, seeds, known_keywords, model, settings, retrieve):
    evidence = list(seeds)
    queries, rounds = [], []
    query = keyword
    try:
        for step in range(settings['max_searches'] + 1):
            if query is not None:
                if len(queries) >= settings['max_searches']:
                    raise ValueError('research retrieval budget exhausted')
                payload = retrieve(query).get('retrieval_result')
                if not isinstance(payload, dict):
                    raise ValueError('invalid retrieval result')
                ranked = sorted((int(str(k)[3:]), v) for k, v in payload.items()
                                if re.fullmatch(r'top\d+', str(k), re.I) and isinstance(v, dict))
                for _, case in ranked[:settings['top_k']]:
                    item = {'title': str(case.get('case_title') or ''), 'content': str(case.get('content') or '')}
                    if not item['title'] or not item['content']:
                        raise ValueError('research case requires title and content')
                    if item not in evidence:
                        evidence.append(item)
                queries.append(query)
            data = {'keyword': keyword, 'known_keywords': known_keywords, 'cases': evidence,
                    'previous_queries': queries, 'can_search': len(queries) < settings['max_searches']}
            response = model.task('research', data)
            rounds.append(response)
            if response['status'] != 'success':
                return {'status': 'failed', 'rounds': rounds, 'queries': queries}
            result = response['result']
            if result['action'] == 'summarize':
                return {'status': 'success', 'result': result, 'rounds': rounds, 'queries': queries}
            query = result['query']
        raise ValueError('research did not summarize within budget')
    except Exception as exc:
        return {'status': 'failed', 'error_type': type(exc).__name__, 'rounds': rounds, 'queries': queries}


def enrich(tasks, groups, model, settings, canonical, save, out, retrieve=None):
    if retrieve is None:
        from search import test_retrieval
        retrieve = lambda q: test_retrieval(q, url=settings['retrieval_url'], timeout=settings['timeout'])
    buckets = {}
    for task in tasks:
        buckets.setdefault(canonical(task['keyword']), []).append(task)
    known = [items[0]['keyword'] for items in buckets.values()]
    eligible = []
    for key, items in buckets.items():
        records = {r['id']: r for task in items for r in groups[task['case_id']]}
        if len(records) > settings['min_dialogues']:
            eligible.append((key, items))
    def research(item):
        key, items = item
        seeds = []
        for task in items:
            case = {'title': task['gt_case_title'], 'content': task['gt_case_content']}
            if case not in seeds:
                seeds.append(case)
        return key, investigate(items[0]['keyword'], seeds, known, model, settings, retrieve)
    with ThreadPoolExecutor(max_workers=settings['concurrency']) as pool:
        results = dict(pool.map(research, eligible))
    save(out / '03_research_results.json', results)
    return results


def expand(clusters, relations, canonical, digest):
    """Outgoing direct neighborhoods; never compute a transitive closure."""
    owners = {canonical(alias): i for i, c in enumerate(clusters) for alias in c['aliases']}
    neighbors = {i: {i} for i in range(len(clusters))}
    for relation in relations:
        a, b = owners.get(canonical(relation['source'])), owners.get(canonical(relation['target']))
        if a is not None and b is not None and a != b:
            neighbors[a].add(b)
    # Keep maximal direct neighborhoods, removing redundant singleton/subset groups.
    sets = {frozenset(v) for v in neighbors.values()}
    sets = [s for s in sets if not any(s < other for other in sets)]
    result = []
    for members in sorted(sets, key=lambda s: sorted(s)):
        selected = [clusters[i] for i in sorted(members)]
        row = dict(selected[0])
        records = {r['id']: r for c in selected for r in c['records']}
        aliases = list(dict.fromkeys(a for c in selected for a in c['aliases']))
        row.update(id=digest(sorted(c['id'] for c in selected))[:16], records=list(records.values()),
                   count=len(records), gt_case_count=len({r['expected_case_id'] for r in records.values()}),
                   aliases=aliases, concepts=list(dict.fromkeys(x for c in selected for x in c['concepts'])),
                   relations=[r for r in relations if canonical(r['source']) in {canonical(a) for a in aliases}
                              and canonical(r['target']) in {canonical(a) for a in aliases}],
                   relation_members=[{'keyword': c['keyword'], 'desc': c['desc']} for c in selected])
        result.append(row)
    return sorted(result, key=lambda c: (-c['count'], canonical(c['keyword'])))
