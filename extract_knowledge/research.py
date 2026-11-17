"""Budgeted keyword research and non-transitive one-hop relation expansion."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import sys
import math
import unicodedata


def normalized(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip().casefold()


def case_chunks(case):
    """Bound embedding input length without dropping the end of long cases."""
    title, content = case['title'], case['content']
    return [title + '\n' + content[i:i + 1000] for i in range(0, max(1, len(content)), 1000)]


class CandidateIndex:
    """Shared GT context index; selection state stays local to each investigation."""
    def __init__(self, tasks, embed=None):
        self.embed = embed
        self.documents = {}
        self.words = {}
        for task in tasks:
            key = normalized(task['keyword'])
            if not key:
                continue
            self.words.setdefault(key, task['keyword'])
            docs = self.documents.setdefault(key, [])
            for chunk in case_chunks({'title': task['gt_case_title'], 'content': task['gt_case_content']}):
                value = task['keyword'] + '\n' + chunk
                if value not in docs:
                    docs.append(value)
        self.vectors, self.error = {}, None
        if embed and len(self.words) > 1:
            try:
                texts = list(dict.fromkeys(t for docs in self.documents.values() for t in docs))
                self.vectors = self.unit_vectors(embed(texts))
            except Exception as exc:
                self.error = type(exc).__name__

    @staticmethod
    def unit_vectors(values):
        if __package__:
            from .embedding import vectors_valid
        else:
            from embedding import vectors_valid
        vectors_valid(list(values.values()), len(values))
        result = {}
        for text, vector in values.items():
            norm = math.hypot(*vector)
            result[text] = [x / norm for x in vector]
        return result

    def select(self, keyword, cases, selected, cache):
        # Each round appends at most 20; old candidates never get replaced.
        capacity = min(20, 40 - len(selected))
        if capacity <= 0:
            return [], {'added': [], 'embedding_error': None}
        excluded = {normalized(keyword), *(normalized(w) for w in selected)}
        remaining = [k for k in self.words if k not in excluded]
        if not remaining or not cases:
            return [], {'added': [], 'embedding_error': self.error}
        texts = [normalized(c[field]) for c in cases for field in ('title', 'content')]
        direct = []
        for key in remaining:
            pattern = re.escape(key)
            if key[0].isascii() and (key[0].isalnum() or key[0] == '_'):
                pattern = r'(?<![a-z0-9_])' + pattern
            if key[-1].isascii() and (key[-1].isalnum() or key[-1] == '_'):
                pattern += r'(?![a-z0-9_])'
            if any(re.search(pattern, text) for text in texts):
                direct.append(key)
        scores, error = {}, self.error
        if len(direct) < capacity and self.embed and self.vectors:
            try:
                chunks = list(dict.fromkeys(t for c in cases for t in case_chunks(c)))
                missing = [t for t in chunks if t not in cache]
                if missing:
                    cache.update(self.unit_vectors(self.embed(missing)))
                for key in remaining:
                    if key in direct:
                        continue
                    if any(len(cache[t]) != len(self.vectors[d]) for t in chunks for d in self.documents[key]):
                        raise ValueError('candidate embedding dimensions differ')
                    scores[key] = max(sum(a*b for a, b in zip(cache[t], self.vectors[d]))
                                      for t in chunks for d in self.documents[key])
            except Exception as exc:
                error = type(exc).__name__
                scores = {}
        ranked = direct + sorted(scores, key=lambda key: (-scores[key], key))
        added = [self.words[key] for key in ranked[:capacity]]
        trace = {'added': [{'keyword': self.words[key], 'method': 'direct' if key in direct else 'embedding',
                            **({'score': scores[key]} if key in scores else {})} for key in ranked[:capacity]],
                 'embedding_error': error}
        return added, trace


def investigate(keyword, seeds, known_keywords, model, settings, retrieve, candidates=None):
    prefix = f'[research {keyword!r}]'
    evidence = list(seeds)
    queries, rounds = [], []
    query = keyword
    selected, selections, vector_cache = [], [], {}
    if candidates is None:
        candidates = CandidateIndex([{'keyword': w, 'gt_case_title': '', 'gt_case_content': ''}
                                     for w in known_keywords])
    def output(**values):
        return {**values, 'candidate_selections': selections,
                'candidate_embedding_failed': any(s['embedding_error'] for s in selections)}
    try:
        for step in range(settings['max_searches'] + 1):
            if query is not None:
                if len(queries) >= settings['max_searches']:
                    raise ValueError('research retrieval budget exhausted')
                print(f'{prefix} 检索 {len(queries) + 1}/{settings["max_searches"]} query={query!r}',
                      file=sys.stderr, flush=True)
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
            added, selection = candidates.select(keyword, evidence, selected, vector_cache)
            selected.extend(added)
            selections.append({**selection, 'known_keywords': list(selected)})
            if selection['embedding_error']:
                print(f'{prefix} 候选 embedding 失败，仅使用直接匹配并保留已有候选', file=sys.stderr, flush=True)
            data = {'keyword': keyword, 'known_keywords': list(selected), 'cases': list(evidence),
                    'previous_queries': list(queries), 'can_search': len(queries) < settings['max_searches']}
            print(f'{prefix} 模型分析中 证据={len(evidence)} 候选新增={len(added)} '
                  f'累计候选={len(selected)}/40 可继续检索={data["can_search"]}',
                  file=sys.stderr, flush=True)
            response = model.task('research', data)
            rounds.append(response)
            if response['status'] != 'success':
                print(f'{prefix} 模型调用失败，回退普通定义生成', file=sys.stderr, flush=True)
                return output(status='failed', rounds=rounds, queries=queries)
            result = response['result']
            if result['action'] == 'summarize':
                print(f'{prefix} 总结完成 关系={len(result["relations"])}', file=sys.stderr, flush=True)
                return output(status='success', result=result, rounds=rounds, queries=queries)
            print(f'{prefix} 模型选择继续检索', file=sys.stderr, flush=True)
            query = result['query']
        raise ValueError('research did not summarize within budget')
    except Exception as exc:
        print(f'{prefix} 调研失败（{type(exc).__name__}），回退普通定义生成', file=sys.stderr, flush=True)
        return output(status='failed', error_type=type(exc).__name__, rounds=rounds, queries=queries)


def enrich(tasks, groups, model, settings, canonical, save, out, retrieve=None, embed=None):
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
    candidates = CandidateIndex(tasks if eligible else [], embed)
    def research(item):
        key, items = item
        seeds = []
        for task in items:
            case = {'title': task['gt_case_title'], 'content': task['gt_case_content']}
            if case not in seeds:
                seeds.append(case)
        return key, investigate(items[0]['keyword'], seeds, known, model, settings, retrieve, candidates)
    print(f'[research] 0/{len(eligible)} 待调研={len(eligible)} 跳过={len(buckets) - len(eligible)} '
          f'并发={settings["concurrency"]}', file=sys.stderr, flush=True)
    completed, failed = 0, 0
    by_key = {}
    with ThreadPoolExecutor(max_workers=settings['concurrency']) as pool:
        futures = [pool.submit(research, item) for item in eligible]
        for future in as_completed(futures):
            key, result = future.result()
            by_key[key] = result
            completed += 1
            failed += result['status'] != 'success'
            print(f'[research] {completed}/{len(eligible)} 成功={completed - failed} 失败={failed}',
                  file=sys.stderr, flush=True)
    # Retain input order so downstream relation and cluster ordering stays stable.
    results = {key: by_key[key] for key, _ in eligible}
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
